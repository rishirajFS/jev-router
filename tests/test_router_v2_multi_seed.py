import pytest

from jev_router.router_v2_multi_seed import aggregate


def report(gap_full, gap_diff, gap_old, primary="full"):
    def row(floor, rule, gap):
        return {"quality_floor": floor, "rule": rule, "frontier_gap_pct": gap,
                "savings_vs_opus_pct": 30.0, "result": {"accuracy": 0.9},
                "accuracy_vs_random_matched": {"mean": 0.05}}
    return {
        "primary_feature_set": primary,
        "learned": {
            "full": {"rows": [row(0.97, "threshold", gap_full)]},
            "difficulty_only": {"rows": [row(0.97, "threshold", gap_diff)]},
        },
        "old_threshold": [{"quality_floor": 0.97, "frontier_gap_pct": gap_old,
                           "savings_vs_opus_pct": 20.0, "jev": {"accuracy": 0.9},
                           "accuracy_vs_random_matched": {"mean": 0.01}}],
    }


def test_aggregate_reports_mean_std_and_share_of_seeds_below_frontier():
    out = aggregate([report(-5, 10, 8), report(-1, 12, 9), report(3, 14, 7)])
    full = out["learned:full:threshold:0.97"]
    assert full["gap_mean"] == pytest.approx(-1.0)
    assert full["share_cheaper_than_frontier"] == pytest.approx(2 / 3)
    assert full["n_seeds"] == 3
    assert out["old_threshold:0.97"]["share_cheaper_than_frontier"] == 0.0
    assert out["primary:threshold:0.97"]["gap_mean"] == pytest.approx(-1.0)


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate([])
