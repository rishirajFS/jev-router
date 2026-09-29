"""Multi-seed robustness check: does Jev routing beat baselines across many dev/test splits?"""

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.dataset import LabeledPrompt
from jev_router.experiment import evaluate_routing
from jev_router.feasibility import load_cached_features, load_cached_jev_tokens
from jev_router.features import RoutingFeatures
from jev_router.routing_eval import ItemOutcome, build_outcomes
from jev_router.run_experiment import ROOT, load_all_items, subsample

_EPS = 1e-12
_SINGLES = ("always_haiku", "always_sonnet", "always_opus")


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _frontier_cost(points: Sequence[tuple[float, float]], accuracy: float) -> float | None:
    """Cheapest expected cost of reaching `accuracy` by randomly mixing single models.

    `points` are (accuracy, cost) pairs of the always-one-model strategies. Returns None
    when no mixture reaches the accuracy.
    """
    candidates = [c for a, c in points if a >= accuracy - _EPS]
    for a1, c1 in points:
        for a2, c2 in points:
            if a1 < accuracy < a2:
                weight = (accuracy - a1) / (a2 - a1)
                candidates.append(c1 + weight * (c2 - c1))
    return min(candidates) if candidates else None


def _seed_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One flat metrics dict per floor for a single seed's report."""
    points = [(report["baselines"][k]["accuracy"], report["baselines"][k]["total_cost"]) for k in _SINGLES]
    rows = []
    for row in report["jev"]:
        jev = row["jev"]
        frontier = _frontier_cost(points, jev["accuracy"])
        rows.append(
            {
                "jev_accuracy": jev["accuracy"],
                "jev_total_cost": jev["total_cost"],
                "savings_vs_opus_pct": row["savings_vs_opus_pct"],
                "acc_vs_random_matched": row["accuracy_vs_random_matched"]["mean"],
                "acc_vs_length_matched": row["accuracy_vs_length_matched"]["mean"],
                "wins_random": row["accuracy_vs_random_matched"]["low"] > 0,
                "wins_length": row["accuracy_vs_length_matched"]["low"] > 0,
                "frontier_gap_pct": None if frontier is None else 100 * (jev["total_cost"] / frontier - 1),
            }
        )
    return rows


def _aggregate_floor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = [r["frontier_gap_pct"] for r in rows if r["frontier_gap_pct"] is not None]
    out: dict[str, Any] = {
        key: _stats([r[key] for r in rows])
        for key in ("jev_accuracy", "jev_total_cost", "savings_vs_opus_pct",
                    "acc_vs_random_matched", "acc_vs_length_matched")
    }
    out["frac_ci_excludes_zero_vs_random"] = sum(r["wins_random"] for r in rows) / len(rows)
    out["frac_ci_excludes_zero_vs_length"] = sum(r["wins_length"] for r in rows) / len(rows)
    out["n_frontier_defined"] = len(gaps)
    out["frontier_gap_pct"] = _stats(gaps) if gaps else None
    out["frac_cheaper_than_frontier"] = sum(g < 0 for g in gaps) / len(gaps) if gaps else None
    return out


def _aggregate_reports(
    reports: Sequence[Mapping[str, Any]], seeds: Sequence[int], floors: Sequence[float]
) -> dict[str, Any]:
    per_seed = [_seed_rows(r) for r in reports]
    return {
        "seeds": list(seeds),
        "n_test": [r["n_test"] for r in reports],
        "baseline_accuracy": {
            k: _stats([r["baselines"][k]["accuracy"] for r in reports]) for k in (*_SINGLES, "oracle")
        },
        "per_floor": {
            str(floor): _aggregate_floor([rows[k] for rows in per_seed]) for k, floor in enumerate(floors)
        },
    }


def evaluate_many(
    outcomes: Sequence[ItemOutcome],
    features: Sequence[RoutingFeatures],
    seeds: Sequence[int],
    floors: Sequence[float],
    jev_cost_per_call: float,
) -> dict[str, Any]:
    """Runs `evaluate_routing` once per seed (a different dev/test split each) and aggregates."""
    if not seeds:
        raise ValueError("evaluate_many needs at least one seed")
    reports = [evaluate_routing(outcomes, features, s, floors, jev_cost_per_call) for s in seeds]
    return _aggregate_reports(reports, seeds, floors)


def _load_offline(seed: int = 42) -> tuple[list[Any], list[ItemOutcome], list[RoutingFeatures], list[float]]:
    """Cached items with complete results and Jev features, plus each item's real Jev cost."""
    items = []
    for name in ("routerbench", "mmlu-pro"):
        items += subsample(load_all_items([name], 420, seed), None, seed)
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in items)
    found, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    jev_cache = ROOT / "data" / "cache" / "jev.jsonl"
    feature_map = load_cached_features(tuple(p for p in prompts if p.id in {i.id for i in found}), jev_cache)
    tokens = load_cached_jev_tokens(tuple(p for p in prompts if p.id in feature_map), jev_cache)
    ready = [i for i in found if i.id in feature_map]
    keep = {i.id for i in ready}
    outcomes = build_outcomes(ready, [r for r in records if r.item_id in keep])
    return ready, outcomes, [feature_map[i.id] for i in ready], [jev_cost_usd(tokens[i.id]) for i in ready]


def _run_group(args: tuple[list[ItemOutcome], list[RoutingFeatures], int, Sequence[float], float]) -> dict[str, Any]:
    outcomes, features, seed, floors, per_call = args
    return evaluate_routing(outcomes, features, seed, floors, per_call)


def evaluate_many_parallel(
    outcomes: list[ItemOutcome],
    features: list[RoutingFeatures],
    seeds: Sequence[int],
    floors: Sequence[float],
    per_call: float,
) -> dict[str, Any]:
    """Same result as `evaluate_many`, with seeds spread over processes."""
    if not seeds:
        raise ValueError("evaluate_many needs at least one seed")
    with ProcessPoolExecutor() as pool:
        reports = list(pool.map(_run_group, [(outcomes, features, s, tuple(floors), per_call) for s in seeds]))
    return _aggregate_reports(reports, seeds, floors)


def _fmt(s: Mapping[str, float], scale: float = 1.0, digits: int = 1) -> str:
    return f"{scale * s['mean']:.{digits}f}+/-{scale * s['std']:.{digits}f}"


def format_table(name: str, report: Mapping[str, Any]) -> str:
    n = len(report["seeds"])
    head = ", ".join(f"{k.replace('always_', '')} {100 * v['mean']:.1f}%" for k, v in report["baseline_accuracy"].items())
    lines = [f"== {name}: {n} seeds, mean test n={statistics.fmean(report['n_test']):.0f} | {head} =="]
    lines.append(f"{'floor':>6} {'acc%':>11} {'cost$':>13} {'save%':>11} {'vs rand pp':>11} {'vs len pp':>11} "
                 f"{'win rand':>8} {'win len':>8} {'gap% vs frontier':>17} {'cheaper':>8}")
    for floor, row in report["per_floor"].items():
        gap = row["frontier_gap_pct"]
        lines.append(
            f"{float(floor):6.2f} {_fmt(row['jev_accuracy'], 100):>11} {_fmt(row['jev_total_cost'], 1, 3):>13} "
            f"{_fmt(row['savings_vs_opus_pct']):>11} {_fmt(row['acc_vs_random_matched'], 100):>11} "
            f"{_fmt(row['acc_vs_length_matched'], 100):>11} {row['frac_ci_excludes_zero_vs_random']:8.0%} "
            f"{row['frac_ci_excludes_zero_vs_length']:8.0%} {(_fmt(gap) if gap else 'n/a'):>17} "
            f"{(format(row['frac_cheaper_than_frontier'], '.0%') if gap else 'n/a'):>8}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--floors", default="0.99,0.97,0.95")
    args = parser.parse_args(argv)
    floors = tuple(float(f) for f in args.floors.split(","))
    seeds = tuple(range(args.seeds))

    ready, outcomes, features, jev_costs = _load_offline()
    print(f"offline: {len(ready)} items with complete cached results (no API calls)")
    groups = {
        "combined": list(range(len(ready))),
        "routerbench": [k for k, i in enumerate(ready) if i.family != "mmlu-pro"],
        "mmlu-pro": [k for k, i in enumerate(ready) if i.family == "mmlu-pro"],
    }
    results: dict[str, Any] = {}
    for name, idx in groups.items():
        per_call = sum(jev_costs[k] for k in idx) / len(idx)
        results[name] = evaluate_many_parallel([outcomes[k] for k in idx], [features[k] for k in idx], seeds, floors, per_call)
        results[name]["jev_cost_per_call"] = per_call
        print(format_table(name, results[name]))
    out = ROOT / "data" / "results" / "multi_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
