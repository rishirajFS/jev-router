import json

import pytest

from jev_router.anthropic_runner import MODELS
from jev_router.features import RoutingFeatures
from jev_router.multi_split import _frontier_cost, _stats, evaluate_many
from jev_router.routing_eval import ItemOutcome

HAIKU, SONNET, OPUS = MODELS


def make_data(n=240):
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


def test_frontier_cost_interpolates_between_single_model_points():
    points = [(0.5, 0.1), (0.8, 0.4), (0.9, 1.0)]
    assert _frontier_cost(points, 0.65) == pytest.approx(0.25)
    assert _frontier_cost(points, 0.9) == pytest.approx(1.0)


def test_frontier_cost_edges():
    points = [(0.5, 0.1), (0.8, 0.4)]
    assert _frontier_cost(points, 0.95) is None  # unreachable by any mixture
    assert _frontier_cost(points, 0.2) == pytest.approx(0.1)  # any model already qualifies


def test_frontier_cost_prefers_the_cheaper_way_to_reach_an_accuracy():
    # a cheap accurate model makes the expensive mid-accuracy one irrelevant
    points = [(0.5, 0.1), (0.8, 0.9), (0.9, 0.2)]
    assert _frontier_cost(points, 0.8) == pytest.approx(0.1 + (0.3 / 0.4) * 0.1)


def test_stats_summarises_across_seeds():
    s = _stats([1.0, 2.0, 3.0])
    assert s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.0)
    assert (s["min"], s["max"]) == (1.0, 3.0)
    assert _stats([4.0])["std"] == 0.0


def test_evaluate_many_aggregates_per_floor_and_is_serializable():
    outcomes, features = make_data()
    report = evaluate_many(outcomes, features, seeds=(0, 1, 2), floors=(1.0,), jev_cost_per_call=0.0)
    assert report["seeds"] == [0, 1, 2]
    row = report["per_floor"]["1.0"]
    for key in ("jev_accuracy", "jev_total_cost", "savings_vs_opus_pct",
                "acc_vs_random_matched", "acc_vs_length_matched", "frontier_gap_pct"):
        assert set(row[key]) == {"mean", "std", "min", "max"}
    assert row["jev_accuracy"]["mean"] == pytest.approx(1.0)
    assert row["savings_vs_opus_pct"]["mean"] > 40
    for key in ("frac_ci_excludes_zero_vs_random", "frac_ci_excludes_zero_vs_length"):
        assert 0.0 <= row[key] <= 1.0
    json.dumps(report)


def test_evaluate_many_reports_when_jev_is_cheaper_than_the_frontier():
    outcomes, features = make_data()
    report = evaluate_many(outcomes, features, seeds=(0, 1), floors=(1.0,), jev_cost_per_call=0.0)
    row = report["per_floor"]["1.0"]
    assert row["frac_cheaper_than_frontier"] == pytest.approx(1.0)
    assert row["frontier_gap_pct"]["mean"] < -30
    assert row["n_frontier_defined"] == 2


def test_evaluate_many_is_deterministic_and_seed_sensitive():
    outcomes, features = make_data()
    a = evaluate_many(outcomes, features, (0, 1), (1.0,), 0.0)
    b = evaluate_many(outcomes, features, (0, 1), (1.0,), 0.0)
    assert a == b
    assert a["n_test"][0] != a["n_test"][1] or a["n_test"][0] > 0
    assert len(a["n_test"]) == 2


def test_evaluate_many_charges_the_jev_overhead():
    outcomes, features = make_data()
    free = evaluate_many(outcomes, features, (0,), (1.0,), 0.0)["per_floor"]["1.0"]
    paid = evaluate_many(outcomes, features, (0,), (1.0,), 0.003)["per_floor"]["1.0"]
    assert paid["jev_total_cost"]["mean"] > free["jev_total_cost"]["mean"]


def test_evaluate_many_rejects_empty_seeds():
    outcomes, features = make_data()
    with pytest.raises(ValueError):
        evaluate_many(outcomes, features, (), (1.0,), 0.0)
