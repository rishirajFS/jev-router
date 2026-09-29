"""Jev vs no-Jev routing experiment: quality per dollar on held-out items."""

import json
import math
from collections.abc import Sequence
from typing import Any

from jev_router.features import RoutingFeatures
from jev_router.routing_eval import (
    HAIKU,
    OPUS,
    SONNET,
    ItemOutcome,
    StrategyResult,
    assign_tiers,
    bootstrap_diff,
    break_even_jev_cost,
    correct_vector,
    length_matched,
    length_tiers,
    oracle,
    random_correct_vector,
    random_matched,
    score_assignment,
    select_policy,
    single_model,
    split_ids,
)


def summary(result: StrategyResult) -> dict[str, Any]:
    qpd = result.correct_per_dollar
    return {
        "accuracy": result.accuracy,
        "correct": result.correct,
        "model_cost": result.model_cost,
        "overhead_cost": result.overhead_cost,
        "total_cost": result.total_cost,
        "correct_per_dollar": None if math.isinf(qpd) else qpd,
        "tier_counts": dict(result.tier_counts),
    }


def _ci(a: Sequence[float], b: Sequence[float], seed: int) -> dict[str, float]:
    mean, low, high = bootstrap_diff(a, b, seed)
    return {"mean": mean, "low": low, "high": high}


def evaluate_routing(
    outcomes: Sequence[ItemOutcome],
    features: Sequence[RoutingFeatures],
    seed: int,
    floors: Sequence[float] = (0.99, 0.97, 0.95),
    jev_cost_per_call: float = 0.0,
) -> dict[str, Any]:
    """Tune the Jev policy on a dev half, report everything on the untouched test half."""
    dev_ids, _ = split_ids([o.item_id for o in outcomes], seed)
    dev_set = set(dev_ids)
    dev = [i for i, o in enumerate(outcomes) if o.item_id in dev_set]
    test = [i for i, o in enumerate(outcomes) if o.item_id not in dev_set]
    dev_o, dev_f = [outcomes[i] for i in dev], [features[i] for i in dev]
    test_o, test_f = [outcomes[i] for i in test], [features[i] for i in test]

    opus = single_model(test_o, OPUS)
    opus_dev_accuracy = single_model(dev_o, OPUS).accuracy
    always_opus_vec = [float(o.correct[OPUS]) for o in test_o]
    report: dict[str, Any] = {
        "n_dev": len(dev_o),
        "n_test": len(test_o),
        "jev_cost_per_call": jev_cost_per_call,
        "baselines": {
            "always_haiku": summary(single_model(test_o, HAIKU)),
            "always_sonnet": summary(single_model(test_o, SONNET)),
            "always_opus": summary(opus),
            "oracle": summary(oracle(test_o)),
        },
        "jev": [],
    }
    for floor in floors:
        policy, _ = select_policy(dev_o, dev_f, floor * opus_dev_accuracy)
        tiers = assign_tiers(test_f, policy)
        jev = score_assignment(test_o, tiers, jev_cost_per_call)
        mix = jev.tier_counts
        random_result = random_matched(test_o, mix)
        length_result = length_matched(test_o, mix)
        jev_vec = correct_vector(test_o, tiers)
        length_vec = correct_vector(test_o, length_tiers(test_o, mix))
        report["jev"].append(
            {
                "quality_floor": floor,
                "policy": {"small_max": policy.small_max, "medium_max": policy.medium_max},
                "jev": summary(jev),
                "random_matched": summary(random_result),
                "length_matched": summary(length_result),
                "savings_vs_opus_pct": 100 * (1 - jev.total_cost / opus.total_cost),
                "accuracy_vs_opus": _ci(jev_vec, always_opus_vec, seed),
                "accuracy_vs_random_matched": _ci(jev_vec, random_correct_vector(test_o, mix), seed),
                "accuracy_vs_length_matched": _ci(jev_vec, length_vec, seed),
                "break_even_jev_cost_vs_opus": break_even_jev_cost(jev, opus),
                "break_even_jev_cost_vs_length": break_even_jev_cost(jev, length_result),
            }
        )
    return report


def format_report(name: str, report: dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:5.1f}%"

    lines = [f"== {name}: test n={report['n_test']} (policy tuned on dev n={report['n_dev']}) =="]
    lines.append(f"{'strategy':28s} {'acc':>7s} {'cost $':>9s} {'correct/$':>10s}")
    for key, value in report["baselines"].items():
        qpd = value["correct_per_dollar"]
        lines.append(f"{key:28s} {pct(value['accuracy']):>7s} {value['total_cost']:9.4f} {qpd or 0:10.0f}")
    for row in report["jev"]:
        f = row["quality_floor"]
        for label, key in (("Jev", "jev"), ("  random, same mix", "random_matched"), ("  length, same mix", "length_matched")):
            v = row[key]
            qpd = v["correct_per_dollar"]
            lines.append(f"{label + f' (floor {f:.0%})':28s} {pct(v['accuracy']):>7s} {v['total_cost']:9.4f} {qpd or 0:10.0f}")
        d = row["accuracy_vs_length_matched"]
        r = row["accuracy_vs_random_matched"]
        lines.append(
            f"    saves {row['savings_vs_opus_pct']:.0f}% vs Opus | acc vs length {100*d['mean']:+.1f}pp "
            f"[{100*d['low']:+.1f},{100*d['high']:+.1f}] | vs random {100*r['mean']:+.1f}pp "
            f"[{100*r['low']:+.1f},{100*r['high']:+.1f}] | break-even Jev price/call vs Opus "
            f"${row['break_even_jev_cost_vs_opus']:.5f}"
        )
    return "\n".join(lines)


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
