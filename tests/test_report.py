import pytest

from jev_router.anthropic_runner import RunRecord
from jev_router.report import summarize_runs


def rec(item, model, correct, out=100, cost=0.01, stop="end_turn"):
    return RunRecord(item, model, correct, 200, out, cost, stop, 0.5)


def test_summarize_runs_reports_accuracy_cost_and_projection():
    records = [
        rec("a", "claude-haiku-4-5", True, cost=0.001),
        rec("b", "claude-haiku-4-5", False, cost=0.001),
        rec("a", "claude-opus-5-5", True, out=300, cost=0.02),
        rec("b", "claude-opus-5-5", True, out=500, cost=0.04, stop="max_tokens"),
    ]
    report = summarize_runs(records, full_size=100)
    haiku = report["models"]["claude-haiku-4-5"]
    opus = report["models"]["claude-opus-5-5"]
    assert haiku["accuracy"] == pytest.approx(0.5)
    assert opus["accuracy"] == pytest.approx(1.0)
    assert opus["mean_output_tokens"] == pytest.approx(400)
    assert opus["truncated"] == 1
    assert report["pilot_cost"] == pytest.approx(0.062)
    assert report["projected_full_cost"] == pytest.approx(0.062 / 2 * 100)


def test_summarize_runs_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_runs([], full_size=10)
