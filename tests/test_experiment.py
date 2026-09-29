import json

import pytest

from jev_router.anthropic_runner import MODELS, RunRecord
from jev_router.experiment import evaluate_routing
from jev_router.features import RoutingFeatures
from jev_router.routing_eval import ItemOutcome

HAIKU, SONNET, OPUS = MODELS


def make_data(n=120):
    outcomes, features = [], []
    for i in range(n):
        hard = i % 3 == 0
        outcomes.append(
            ItemOutcome(
                f"i{i}", "mmlu", 900 if hard else 300,
                {HAIKU: not hard, SONNET: not hard, OPUS: True},
                {HAIKU: 0.001, SONNET: 0.004, OPUS: 0.010},
            )
        )
        features.append(RoutingFeatures("math", 0.9, 2.6 if hard else 0.3, 0.9, 0.1))
    return outcomes, features


def test_evaluate_routing_finds_savings_at_matched_quality():
    outcomes, features = make_data()
    report = evaluate_routing(outcomes, features, seed=1, floors=(1.0,), jev_cost_per_call=0.0)
    row = report["jev"][0]
    assert row["jev"]["accuracy"] == pytest.approx(1.0)
    assert row["savings_vs_opus_pct"] > 40
    assert row["accuracy_vs_random_matched"]["mean"] > 0.1
    assert report["baselines"]["always_opus"]["accuracy"] == 1.0
    json.dumps(report)  # must be serializable for saving


def test_evaluate_routing_charges_jev_overhead():
    outcomes, features = make_data()
    free = evaluate_routing(outcomes, features, seed=1, floors=(1.0,), jev_cost_per_call=0.0)
    paid = evaluate_routing(outcomes, features, seed=1, floors=(1.0,), jev_cost_per_call=0.003)
    assert paid["jev"][0]["jev"]["total_cost"] > free["jev"][0]["jev"]["total_cost"]
    assert paid["jev"][0]["jev"]["overhead_cost"] > 0


def test_cached_loaders_return_only_fully_cached_items(tmp_path):
    from jev_router.anthropic_runner import load_cached_records
    from jev_router.sampling import BenchItem

    items = [BenchItem("a", "mmlu", "Q\nA) x\nB) y", "A", 0.5), BenchItem("b", "mmlu", "Q2\nA) x\nB) y", "B", 0.5)]
    cache = tmp_path / "c.jsonl"
    from jev_router.anthropic_runner import _key

    rows = [
        {"key": _key(items[0], m), "item_id": "a", "model": m, "correct": True, "input_tokens": 1,
         "output_tokens": 1, "cost_usd": 0.001, "stop_reason": "end_turn", "latency_seconds": 0.1}
        for m in MODELS
    ]
    rows.append({**rows[0], "key": _key(items[1], HAIKU), "item_id": "b", "model": HAIKU})
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    found_items, records = load_cached_records(items, MODELS, cache)
    assert [i.id for i in found_items] == ["a"]
    assert len(records) == 3
    assert all(isinstance(r, RunRecord) for r in records)
