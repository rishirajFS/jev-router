"""Untouched held-out check of the frozen learned Jev router.

PRE-REGISTRATION (written before any paid call for this check was made)
-----------------------------------------------------------------------
The question wording (questions_v2), the feature list, the regularization grid, the rule grids and
the routing code are FROZEN as they were when the seed-42 results were produced. The router is
refit exactly as router_v2_experiment.score_v2 does for seed 42: dev half of the ORIGINAL 1,220
items (split_ids, seed 42), primary feature set 'full', C chosen by dev out-of-fold log-loss,
thresholds per quality floor tuned on out-of-fold dev predictions. It is then applied ONCE to about
600 fresh items that were never scored, never used for design, and are disjoint from the original
1,220 (about 300 RouterBench, about 294 MMLU-Pro).

Decision rule. On the COMBINED fresh set, the frozen full-feature learned router (threshold rule)
is judged by its gap to the single-model frontier: its total cost (including the real Jev v2 cost
of the six-question call) above the cheapest random mixture of always-Haiku / always-Sonnet /
always-Opus that reaches the same accuracy, with the frontier computed from the fresh set's own
single-model results. Quality floors are 0.99, 0.97 and 0.95 of Opus's dev accuracy.
    'holds'          = gap < 0 at 2 of the 3 floors.
    'does not hold'  = otherwise.
Also reported, with bootstrap CIs: the frozen old difficulty-threshold router, the expected-loss
rule, same-mix random and same-mix prompt-length baselines, and the RouterBench / MMLU-Pro
breakdowns. The result is reported whichever way it goes; nothing is re-tuned after seeing it.
"""

import argparse
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_router.anthropic_runner import MODELS, load_cached_records, make_tracker, run
from jev_router.budget import BudgetExceeded, jev_cost_usd
from jev_router.build_samples import write_items
from jev_router.config import load_api_key
from jev_router.dataset import LabeledPrompt
from jev_router.experiment import summary
from jev_router.feasibility import collect, load_cached_features, load_cached_jev_tokens
from jev_router.learned_router import (
    Rule,
    apply_rule,
    fit_prob_models,
    frontier_cost,
    oof_probs,
    predict_probs,
    select_loss_rule,
    select_threshold_rule,
)
from jev_router.mmlu_pro import build_mmlu_pro_sample
from jev_router.policy import Tier
from jev_router.questions_v2 import FEATURE_NAMES, build_questions_v2, extract_features
from jev_router.router_v2_experiment import (
    C_GRID,
    JEV_V2_CACHE,
    _dev_logloss,
    collect_v2,
    load_cached_v2,
)
from jev_router.routing_eval import (
    HAIKU,
    OPUS,
    SONNET,
    ItemOutcome,
    assign_tiers,
    bootstrap_diff,
    build_outcomes,
    correct_vector,
    length_matched,
    length_tiers,
    random_correct_vector,
    random_matched,
    score_assignment,
    select_policy,
    single_model,
    split_ids,
)
from jev_router.run_experiment import ROOT, load_items, projected_cost
from jev_router.sampling import BenchItem, build_sample

FLOORS = (0.99, 0.97, 0.95)
SEED_ORIGINAL = 42
SEED_FRESH = 2027
ROUTERBENCH_PLAN = {"mmlu": 187, "arc-challenge": 38, "hellaswag": 38, "winogrande": 37}
MMLU_PRO_N = 294
PROJECTION_LIMIT_USD = 3.0
ANTHROPIC_CAP_USD = 7.0
JEV_V2_CAP_USD = 0.40
FRESH_PATH = ROOT / "data" / "fresh_sample.jsonl"


def fresh_routerbench_sample(
    frame: pd.DataFrame, models: Sequence[str], plan: Mapping[str, int], seed: int,
    exclude_ids: Collection[str], min_votes: int = 2,
) -> list[BenchItem]:
    """Same construction as the original sample, over the candidates that remain."""
    remaining = frame[~frame["sample_id"].isin(set(exclude_ids))]
    return build_sample(remaining, models, dict(plan), seed=seed, min_votes=min_votes)


def fresh_mmlu_pro_sample(
    frame: pd.DataFrame, n: int, seed: int, exclude_ids: Collection[str]
) -> list[BenchItem]:
    ids = "mmlu-pro." + frame["question_id"].astype(str)
    return build_mmlu_pro_sample(frame[~ids.isin(set(exclude_ids))], n, seed)


def assert_disjoint(fresh: Sequence[BenchItem], existing_ids: Collection[str]) -> None:
    clash = {i.id for i in fresh} & set(existing_ids)
    if clash:
        raise ValueError(f"fresh sample overlap with existing items: {sorted(clash)[:5]}")


def gap_pct(cost: float, accuracy: float, points: Sequence[tuple[float, float]]) -> float | None:
    """Cost above the cheapest single-model mixture at the same accuracy (negative = cheaper)."""
    frontier = frontier_cost(points, accuracy)
    return None if not np.isfinite(frontier) else 100 * (cost / frontier - 1)


def verdict(gaps: Mapping[float, float | None]) -> str:
    return "holds" if sum(g is not None and g < 0 for g in gaps.values()) >= 2 else "does not hold"


@dataclass(frozen=True, eq=False)
class FrozenRouter:
    models: tuple
    best_c: float
    mean_costs: np.ndarray
    rules: Mapping[tuple[str, float], Rule]
    columns: tuple[int, ...]

    def probs(self, X_full: np.ndarray) -> np.ndarray:
        return predict_probs(self.models, X_full[:, list(self.columns)])

    def tiers_for(self, X_full: np.ndarray, key: tuple[str, float]) -> np.ndarray:
        return apply_rule(self.rules[key], self.probs(X_full), self.mean_costs)


def fit_frozen_router(
    X_dev: np.ndarray, Y_dev: np.ndarray, C_dev: np.ndarray, opus_dev_accuracy: float,
    floors: Sequence[float], columns: Sequence[int],
) -> FrozenRouter:
    """Exactly the fitting steps of router_v2_experiment.score_v2, on dev arrays only."""
    cols = tuple(columns)
    Xd = X_dev[:, list(cols)]
    scored = {c: oof_probs(Xd, Y_dev, c) for c in C_GRID}
    best_c = min(C_GRID, key=lambda c: _dev_logloss(scored[c], Y_dev))
    P_dev = scored[best_c]
    rules: dict[tuple[str, float], Rule] = {}
    for floor in floors:
        floor_acc = floor * opus_dev_accuracy
        rules[("threshold", floor)] = select_threshold_rule(P_dev, Y_dev, C_dev, floor_acc)
        rules[("loss", floor)] = select_loss_rule(P_dev, Y_dev, C_dev, floor_acc)
    return FrozenRouter(fit_prob_models(Xd, Y_dev, best_c), best_c, C_dev.mean(axis=0), rules, cols)


def _arrays(outcomes: Sequence[ItemOutcome]) -> tuple[np.ndarray, np.ndarray]:
    correct = np.array([[float(o.correct[m]) for m in MODELS] for o in outcomes])
    cost = np.array([[o.cost[m] for m in MODELS] for o in outcomes])
    return correct, cost


def _ci(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    mean, low, high = bootstrap_diff(a, b, SEED_ORIGINAL)
    return {"mean": mean, "low": low, "high": high}


def score_router(
    outcomes: Sequence[ItemOutcome], tiers: Sequence[Tier], overhead: float
) -> dict[str, Any]:
    points = [(r.accuracy, r.total_cost) for r in (single_model(outcomes, m) for m in (HAIKU, SONNET, OPUS))]
    result = score_assignment(outcomes, tiers, overhead)
    mix = result.tier_counts
    vec = correct_vector(outcomes, tiers)
    return {
        "result": summary(result),
        "savings_vs_opus_pct": 100 * (1 - result.total_cost / single_model(outcomes, OPUS).total_cost),
        "frontier_gap_pct": gap_pct(result.total_cost, result.accuracy, points),
        "accuracy_vs_random_matched": _ci(vec, random_correct_vector(outcomes, mix)),
        "accuracy_vs_length_matched": _ci(vec, correct_vector(outcomes, length_tiers(outcomes, mix))),
        "random_matched": summary(random_matched(outcomes, mix)),
        "length_matched": summary(length_matched(outcomes, mix)),
    }


@dataclass(frozen=True, eq=False)
class Inputs:
    """Everything needed to score routers on one list of items."""
    items: list[BenchItem]
    outcomes: list[ItemOutcome]
    X: np.ndarray
    v2_cost: list[float]
    v1_features: list[Any]
    v1_cost: list[float]


def load_inputs(items: Sequence[BenchItem], anthropic_cache: Path, jev_cache: Path) -> Inputs:
    ready, records = load_cached_records(items, MODELS, anthropic_cache)
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    v2 = load_cached_v2(prompts, JEV_V2_CACHE, build_questions_v2())
    v1 = load_cached_features(prompts, jev_cache)
    v1_tokens = load_cached_jev_tokens(prompts, jev_cache)
    usable = [i for i in ready if i.id in v2 and i.id in v1]
    keep = {i.id for i in usable}
    records = [r for r in records if r.item_id in keep]
    return Inputs(
        items=usable,
        outcomes=build_outcomes(usable, records),
        X=np.array([[extract_features(v2[i.id]["answers"], len(i.prompt))[n] for n in FEATURE_NAMES] for i in usable]),
        v2_cost=[jev_cost_usd(v2[i.id]["usage"]["input_tokens"]) for i in usable],
        v1_features=[v1[i.id] for i in usable],
        v1_cost=[jev_cost_usd(v1_tokens[i.id]) for i in usable],
    )


def evaluate_frozen(original: Inputs, fresh: Inputs) -> dict[str, Any]:
    dev_ids, _ = split_ids([o.item_id for o in original.outcomes], SEED_ORIGINAL)
    dev = [k for k, o in enumerate(original.outcomes) if o.item_id in set(dev_ids)]
    dev_o = [original.outcomes[k] for k in dev]
    Yd, Cd = _arrays(dev_o)
    opus_dev = single_model(dev_o, OPUS).accuracy
    router = fit_frozen_router(original.X[dev], Yd, Cd, opus_dev, FLOORS, range(len(FEATURE_NAMES)))
    old_policies = {f: select_policy(dev_o, [original.v1_features[k] for k in dev], f * opus_dev)[0] for f in FLOORS}

    groups = {"combined": range(len(fresh.items))}
    for name, want in (("routerbench", False), ("mmlu-pro", True)):
        groups[name] = [k for k, i in enumerate(fresh.items) if (i.family == "mmlu-pro") == want]
    report: dict[str, Any] = {"best_c": router.best_c, "n_fresh": len(fresh.items), "groups": {}}
    for name, idx in groups.items():
        idx = list(idx)
        outcomes = [fresh.outcomes[k] for k in idx]
        v2_over = float(np.mean([fresh.v2_cost[k] for k in idx]))
        v1_over = float(np.mean([fresh.v1_cost[k] for k in idx]))
        block: dict[str, Any] = {
            "n": len(idx),
            "baselines": {m: summary(single_model(outcomes, m)) for m in (HAIKU, SONNET, OPUS)},
            "jev_v2_overhead_per_call": v2_over,
            "learned": [], "old_threshold": [],
        }
        for floor in FLOORS:
            for kind in ("threshold", "loss"):
                tiers = [Tier(int(t)) for t in router.tiers_for(fresh.X, (kind, floor))]
                block["learned"].append({"quality_floor": floor, "rule": kind,
                                         **score_router(outcomes, [tiers[k] for k in idx], v2_over)})
            old_tiers = assign_tiers(fresh.v1_features, old_policies[floor])
            block["old_threshold"].append({"quality_floor": floor,
                                           **score_router(outcomes, [old_tiers[k] for k in idx], v1_over)})
        report["groups"][name] = block
    combined = report["groups"]["combined"]["learned"]
    gaps = {r["quality_floor"]: r["frontier_gap_pct"] for r in combined if r["rule"] == "threshold"}
    report["preregistered_gaps_pct"] = {str(f): g for f, g in gaps.items()}
    report["verdict"] = verdict(gaps)

    # Sanity: the refit router must reproduce the seed-42 test-half numbers already on record.
    test = [k for k, o in enumerate(original.outcomes) if o.item_id not in set(dev_ids)]
    test_o = [original.outcomes[k] for k in test]
    v2_over = float(np.mean([original.v2_cost[k] for k in test]))
    recorded = json.loads((ROOT / "data" / "results" / "router_v2.json").read_text())["learned"]["full"]["rows"]
    checks = []
    for row in (r for r in recorded if r["rule"] == "threshold"):
        tiers = [Tier(int(t)) for t in router.tiers_for(original.X[test], ("threshold", row["quality_floor"]))]
        got = score_assignment(test_o, tiers, v2_over)
        checks.append(abs(got.accuracy - row["result"]["accuracy"]) < 1e-9 and abs(got.total_cost - row["result"]["total_cost"]) < 1e-6)
    report["reproduces_seed42_test_half"] = bool(all(checks))
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"fresh n={report['n_fresh']} | refit reproduces seed-42 test half: {report['reproduces_seed42_test_half']}"]
    for name, block in report["groups"].items():
        b = block["baselines"]
        lines.append(f"== {name} (n={block['n']}): Haiku {100*b[HAIKU]['accuracy']:.1f}% ${b[HAIKU]['total_cost']:.3f} | "
                     f"Sonnet {100*b[SONNET]['accuracy']:.1f}% ${b[SONNET]['total_cost']:.3f} | Opus {100*b[OPUS]['accuracy']:.1f}% ${b[OPUS]['total_cost']:.3f}")
        for label, rows in (("learned", block["learned"]), ("old", block["old_threshold"])):
            for r in rows:
                res, rand, ln = r["result"], r["accuracy_vs_random_matched"], r["accuracy_vs_length_matched"]
                gap = r["frontier_gap_pct"]
                lines.append(f"  {label:8s} {r.get('rule', 'thr'):9s} floor {r['quality_floor']:.2f}: acc {100*res['accuracy']:.1f}% "
                             f"cost ${res['total_cost']:.3f} saves {r['savings_vs_opus_pct']:.0f}% gap "
                             f"{'n/a' if gap is None else f'{gap:+.1f}%'} | vs random {100*rand['mean']:+.1f}pp [{100*rand['low']:+.1f},{100*rand['high']:+.1f}] "
                             f"| vs length {100*ln['mean']:+.1f}pp [{100*ln['low']:+.1f},{100*ln['high']:+.1f}]")
    lines.append(f"PRE-REGISTERED gaps (combined, threshold rule): {report['preregistered_gaps_pct']} -> VERDICT: {report['verdict']}")
    return "\n".join(lines)


def build_fresh_sample() -> list[BenchItem]:
    existing = load_items(ROOT / "data" / "routerbench_sample.jsonl") + load_items(ROOT / "data" / "mmlu_pro_sample.jsonl")
    used = {i.id for i in existing}
    frame = pd.read_pickle(ROOT / "data" / "raw" / "routerbench_0shot.pkl")
    models = [c for c in frame.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    fresh = fresh_routerbench_sample(frame, models, ROUTERBENCH_PLAN, SEED_FRESH, used)
    fresh += fresh_mmlu_pro_sample(pd.read_parquet(ROOT / "data" / "raw" / "mmlu_pro_test.parquet"), MMLU_PRO_N, SEED_FRESH, used)
    assert_disjoint(fresh, used)
    if len({i.id for i in fresh}) != len(fresh):
        raise ValueError("duplicate ids inside the fresh sample")
    return fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="cache only, no API calls")
    args = parser.parse_args(argv)
    anthropic_cache = ROOT / "data" / "cache" / "anthropic.jsonl"
    jev_cache = ROOT / "data" / "cache" / "jev.jsonl"

    if FRESH_PATH.exists():
        fresh_items = load_items(FRESH_PATH)
    else:
        fresh_items = build_fresh_sample()
        write_items(fresh_items, FRESH_PATH)
    print(f"fresh items: {len(fresh_items)} (routerbench {sum(i.family != 'mmlu-pro' for i in fresh_items)}, "
          f"mmlu-pro {sum(i.family == 'mmlu-pro' for i in fresh_items)}), disjoint from the original 1,220")

    if not args.evaluate_only:
        original_items = load_items(ROOT / "data" / "routerbench_sample.jsonl") + load_items(ROOT / "data" / "mmlu_pro_sample.jsonl")
        _, cached = load_cached_records(original_items, MODELS, anthropic_cache)
        families = {i.id: i.family for i in original_items}
        _, done = load_cached_records(fresh_items, MODELS, anthropic_cache)
        done_ids = {r.item_id for r in done}
        todo = [i for i in fresh_items if i.id not in done_ids]
        projected = projected_cost(todo, cached, families, MODELS)
        print(f"projected Anthropic spend for {len(todo)} uncached fresh items: ~${projected:.2f} (limit ${PROJECTION_LIMIT_USD:.2f})")
        if projected > PROJECTION_LIMIT_USD:
            print("STOP: projection exceeds the limit; not running.")
            return 3
        if args.project_only:
            return 0
        import anthropic

        from jev_router.jev_client import JevClient

        prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in fresh_items)
        jev = JevClient(load_api_key(ROOT / ".env"))
        try:
            collect(prompts, jev, jev_cache, workers=6)
            collect_v2(prompts, jev, JEV_V2_CACHE, build_questions_v2(), workers=6, cap_usd=JEV_V2_CAP_USD)
        finally:
            jev.close()
        client = anthropic.Anthropic(
            api_key=load_api_key(ROOT / ".env", name="ANTHROPIC_API_KEY"), max_retries=4,
            default_headers={"anthropic-workspace-id": load_api_key(ROOT / ".env", name="ANTHROPIC_WORKSPACE_ID")},
        )
        tracker = make_tracker(ANTHROPIC_CAP_USD, anthropic_cache)
        before = tracker.spent
        try:
            run(fresh_items, MODELS, client, anthropic_cache, tracker)
        except BudgetExceeded as exc:
            print(f"STOPPED AT BUDGET: {exc}")
            return 2
        print(f"anthropic spend this run: ${tracker.spent - before:.4f}; cumulative ledger ${tracker.spent:.4f}")

    original = load_inputs(
        load_items(ROOT / "data" / "routerbench_sample.jsonl") + load_items(ROOT / "data" / "mmlu_pro_sample.jsonl"),
        anthropic_cache, jev_cache)
    fresh = load_inputs(fresh_items, anthropic_cache, jev_cache)
    if len(fresh.items) != len(fresh_items):
        print(f"WARNING: only {len(fresh.items)} of {len(fresh_items)} fresh items are fully cached")
    report = evaluate_frozen(original, fresh)
    out = ROOT / "data" / "results" / "fresh_heldout.json"
    out.write_text(json.dumps(report, indent=2))
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
