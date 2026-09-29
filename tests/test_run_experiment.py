import pytest

from jev_router.anthropic_runner import RunRecord
from jev_router.run_experiment import projected_cost, subsample
from jev_router.sampling import BenchItem


def item(i, family="mmlu"):
    return BenchItem(f"{family}.{i}", family, "Q\nA) x\nB) y", "A", None)


def test_subsample_is_deterministic_and_bounded():
    items = [item(i) for i in range(50)]
    a = subsample(items, 10, seed=3)
    assert [i.id for i in a] == [i.id for i in subsample(items, 10, seed=3)]
    assert len(a) == 10 and len({i.id for i in a}) == 10
    assert len(subsample(items, None, seed=3)) == 50
    assert len(subsample(items, 999, seed=3)) == 50


def rec(item_id, model, cost):
    return RunRecord(item_id, model, True, 100, 10, cost, "end_turn", 0.1)


def test_projected_cost_uses_cached_family_averages():
    todo = [item(1, "mmlu"), item(2, "mmlu")]
    cached = [rec("x", "claude-haiku-4-5", 0.001), rec("x", "claude-opus-5-5", 0.009)]
    families = {"x": "mmlu"}
    assert projected_cost(todo, cached, families, ("claude-haiku-4-5", "claude-opus-5-5")) == pytest.approx(
        2 * (0.001 + 0.009)
    )


def test_projected_cost_pads_unknown_families_with_a_safety_factor():
    todo = [item(1, "mmlu-pro")]
    cached = [rec("x", "claude-haiku-4-5", 0.001)]
    known = projected_cost(todo, cached, {"x": "mmlu-pro"}, ("claude-haiku-4-5",))
    unknown = projected_cost(todo, cached, {"x": "mmlu"}, ("claude-haiku-4-5",))
    assert unknown == pytest.approx(known * 2)
