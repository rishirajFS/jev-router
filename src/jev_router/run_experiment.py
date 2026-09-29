"""CLI: Jev vs no-Jev routing on RouterBench + MMLU-Pro, fully cached, hard-capped spend."""

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from jev_router.anthropic_runner import (
    MODELS,
    RunRecord,
    ledger_spent,
    load_cached_records,
    make_tracker,
    run,
)
from jev_router.budget import BudgetExceeded, jev_cost_usd
from jev_router.config import load_api_key
from jev_router.dataset import LabeledPrompt
from jev_router.experiment import dumps, evaluate_routing, format_report
from jev_router.feasibility import collect, load_cached_features, load_cached_jev_tokens
from jev_router.mmlu_pro import build_mmlu_pro_sample
from jev_router.routing_eval import build_outcomes
from jev_router.sampling import BenchItem

ROOT = Path(__file__).resolve().parents[2]
UNKNOWN_FAMILY_SAFETY = 2.0


def load_items(path: Path) -> list[BenchItem]:
    return [BenchItem(**json.loads(line)) for line in path.read_text().splitlines() if line]


def subsample(items: Sequence[BenchItem], limit: int | None, seed: int) -> list[BenchItem]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled if limit is None else shuffled[:limit]


def projected_cost(
    todo: Sequence[BenchItem],
    cached: Sequence[RunRecord],
    families: Mapping[str, str],
    models: Sequence[str],
) -> float:
    """Estimate spend for uncached items from cached per-family mean cost per model."""
    def mean_cost(model: str, family: str | None) -> float | None:
        rows = [r.cost_usd for r in cached if r.model == model and (family is None or families.get(r.item_id) == family)]
        return sum(rows) / len(rows) if rows else None

    total = 0.0
    for item in todo:
        for model in models:
            own = mean_cost(model, item.family)
            if own is not None:
                total += own
            else:
                overall = mean_cost(model, None)
                total += (overall or 0.0) * UNKNOWN_FAMILY_SAFETY
    return total


def load_all_items(datasets: Sequence[str], mmlu_pro_n: int, seed: int) -> list[BenchItem]:
    items: list[BenchItem] = []
    if "routerbench" in datasets:
        items += load_items(ROOT / "data" / "routerbench_sample.jsonl")
    if "mmlu-pro" in datasets:
        path = ROOT / "data" / "mmlu_pro_sample.jsonl"
        if not path.exists():
            frame = pd.read_parquet(ROOT / "data" / "raw" / "mmlu_pro_test.parquet")
            sample = build_mmlu_pro_sample(frame, mmlu_pro_n, seed)
            path.write_text("".join(json.dumps(i.__dict__) + "\n" for i in sample))
        items += load_items(path)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="routerbench,mmlu-pro")
    parser.add_argument("--limit", type=int, default=None, help="items per dataset (after a seeded shuffle)")
    parser.add_argument("--mmlu-pro-n", type=int, default=420)
    parser.add_argument("--cap", type=float, default=12.0, help="CUMULATIVE Anthropic spend cap in USD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--floors", default="0.99,0.97,0.95")
    parser.add_argument("--jev-cost", type=float, default=None, help="override Jev price per call in USD (default: real per-token price)")
    parser.add_argument("--offline", action="store_true", help="use cached results only; no API calls")
    parser.add_argument("--dry-run", action="store_true", help="print projected cost and exit")
    args = parser.parse_args(argv)

    datasets = args.datasets.split(",")
    items: list[BenchItem] = []
    for name in datasets:
        chunk = load_all_items([name], args.mmlu_pro_n, args.seed)
        items += subsample(chunk, args.limit, args.seed)
    families = {i.id: i.family for i in items}
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in items)
    anthropic_cache = ROOT / "data" / "cache" / "anthropic.jsonl"
    jev_cache = ROOT / "data" / "cache" / "jev.jsonl"

    cached_items, cached_records = load_cached_records(items, MODELS, anthropic_cache)
    todo = [i for i in items if i.id not in {c.id for c in cached_items}]
    print(f"items={len(items)} cached_for_all_models={len(cached_items)} to_run={len(todo)}")
    print(f"anthropic spent so far (ledger): ${ledger_spent(anthropic_cache):.4f} of ${args.cap:.2f} cap")
    if args.dry_run or args.offline:
        est = projected_cost(todo, cached_records, {**families}, MODELS)
        print(f"projected extra spend for uncached items: ~${est:.2f}")
        if args.dry_run:
            return 0

    if not args.offline:
        import anthropic

        from jev_router.jev_client import JevClient

        jev = JevClient(load_api_key(ROOT / ".env"))
        try:
            collect(prompts, jev, jev_cache, workers=6)
        finally:
            jev.close()
        client = anthropic.Anthropic(
            api_key=load_api_key(ROOT / ".env", name="ANTHROPIC_API_KEY"),
            max_retries=4,
            default_headers={"anthropic-workspace-id": load_api_key(ROOT / ".env", name="ANTHROPIC_WORKSPACE_ID")},
        )
        tracker = make_tracker(args.cap, anthropic_cache)
        spent_before = tracker.spent
        try:
            run(items, MODELS, client, anthropic_cache, tracker)
        except BudgetExceeded as exc:
            print(f"STOPPED AT BUDGET: {exc}")
            return 2
        print(f"spent this session: ${tracker.spent - spent_before:.4f} (ledger total ${tracker.spent:.4f})")

    ready, records = load_cached_records(items, MODELS, anthropic_cache)
    feature_map = load_cached_features(tuple(p for p in prompts if p.id in {i.id for i in ready}), jev_cache)
    jev_tokens = load_cached_jev_tokens(tuple(p for p in prompts if p.id in feature_map), jev_cache)
    ready = [i for i in ready if i.id in feature_map]
    records = [r for r in records if r.item_id in {i.id for i in ready}]
    outcomes = build_outcomes(ready, records)
    features = [feature_map[i.id] for i in ready]
    floors = tuple(float(f) for f in args.floors.split(","))

    groups = {"combined": list(range(len(ready)))}
    for name in ("routerbench", "mmlu-pro"):
        idx = [k for k, i in enumerate(ready) if (i.family == "mmlu-pro") == (name == "mmlu-pro")]
        if idx and len(idx) != len(ready):
            groups[name] = idx
    results = {}
    for name, idx in groups.items():
        if len(idx) < 20:
            print(f"skipping {name}: only {len(idx)} complete items")
            continue
        mean_jev = sum(jev_cost_usd(jev_tokens[ready[k].id]) for k in idx) / len(idx)
        per_call = mean_jev if args.jev_cost is None else args.jev_cost
        report = evaluate_routing([outcomes[k] for k in idx], [features[k] for k in idx], args.seed, floors, per_call)
        report["jev_total_cost_all_calls"] = mean_jev * len(idx)
        results[name] = report
        print(format_report(name, report))
    out = ROOT / "data" / "results" / ("experiment_offline.json" if args.offline else "experiment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
