"""Robustness check: re-run the learned-router pipeline across many dev/test splits (cached, no API calls)."""

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.dataset import LabeledPrompt
from jev_router.feasibility import load_cached_features, load_cached_jev_tokens
from jev_router.questions_v2 import FEATURE_NAMES, build_questions_v2, extract_features
from jev_router.router_v2_experiment import JEV_V2_CACHE, load_cached_v2, score_v2
from jev_router.routing_eval import build_outcomes
from jev_router.run_experiment import ROOT, load_all_items


def _stats(gaps: list[float], savings: list[float], acc: list[float], vs_random: list[float]) -> dict[str, Any]:
    return {
        "n_seeds": len(gaps),
        "gap_mean": statistics.fmean(gaps),
        "gap_std": statistics.pstdev(gaps),
        "share_cheaper_than_frontier": sum(g < 0 for g in gaps) / len(gaps),
        "savings_vs_opus_mean": statistics.fmean(savings),
        "accuracy_mean": statistics.fmean(acc),
        "acc_vs_random_mean_pp": 100 * statistics.fmean(vs_random),
    }


def aggregate(reports: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (method, floor): frontier gap (negative = cheaper than the best single-model mix)."""
    if not reports:
        raise ValueError("aggregate needs at least one report")
    cells: dict[str, list[tuple[float, float, float, float]]] = {}

    def add(key: str, gap: float, savings: float, acc: float, vs_random: float) -> None:
        cells.setdefault(key, []).append((gap, savings, acc, vs_random))

    for report in reports:
        for row in report["old_threshold"]:
            add(f"old_threshold:{row['quality_floor']}", row["frontier_gap_pct"], row["savings_vs_opus_pct"],
                row["jev"]["accuracy"], row["accuracy_vs_random_matched"]["mean"])
        for set_name, block in report["learned"].items():
            for row in block["rows"]:
                args = (row["frontier_gap_pct"], row["savings_vs_opus_pct"], row["result"]["accuracy"],
                        row["accuracy_vs_random_matched"]["mean"])
                add(f"learned:{set_name}:{row['rule']}:{row['quality_floor']}", *args)
                if set_name == report["primary_feature_set"]:
                    add(f"primary:{row['rule']}:{row['quality_floor']}", *args)
    return {key: _stats(*[list(col) for col in zip(*rows)]) for key, rows in cells.items()}


def _one(args: tuple) -> dict[str, Any]:
    outcomes, X, v2_cost, v1_features, v1_cost, seed, floors = args
    return score_v2(outcomes, X, v2_cost, v1_features, v1_cost, seed, floors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--floors", default="0.99,0.97,0.95")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    items = load_all_items(("routerbench", "mmlu-pro"), 420, 42)
    ready, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    rows = load_cached_v2(prompts, JEV_V2_CACHE, build_questions_v2())
    outcomes = build_outcomes(ready, records)
    X = np.array([[extract_features(rows[p.id]["answers"], len(p.text))[n] for n in FEATURE_NAMES] for p in prompts])
    v2_cost = [jev_cost_usd(rows[p.id]["usage"]["input_tokens"]) for p in prompts]
    v1 = load_cached_features(prompts, ROOT / "data" / "cache" / "jev.jsonl")
    v1_tokens = load_cached_jev_tokens(prompts, ROOT / "data" / "cache" / "jev.jsonl")
    v1_cost = float(np.mean([jev_cost_usd(v1_tokens[p.id]) for p in prompts]))
    floors = tuple(float(f) for f in args.floors.split(","))
    jobs = [(outcomes, X, v2_cost, [v1[p.id] for p in prompts], v1_cost, seed, floors) for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(_one, jobs))
    summary = aggregate(reports)
    out = ROOT / "data" / "results" / "router_v2_multi_seed.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"{'method':38s} {'gap mean':>9s} {'std':>6s} {'cheaper':>8s} {'saves':>7s} {'acc':>6s} {'vs rand':>8s}")
    for key, v in sorted(summary.items()):
        if key.startswith(("primary:threshold", "old_threshold", "learned:difficulty_only:threshold")) or key.startswith("primary:loss"):
            print(f"{key:38s} {v['gap_mean']:+8.1f}% {v['gap_std']:5.1f} {100*v['share_cheaper_than_frontier']:7.0f}% "
                  f"{v['savings_vs_opus_mean']:6.1f}% {100*v['accuracy_mean']:5.1f}% {v['acc_vs_random_mean_pp']:+7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
