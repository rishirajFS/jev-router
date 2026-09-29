import pytest

from jev_router.budget import PRICES, BudgetExceeded, BudgetTracker, cost_usd


def test_cost_usd_uses_per_million_prices():
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("gpt-imaginary", 1, 1)


def test_tracker_accumulates_and_reports_by_model():
    tracker = BudgetTracker(cap_usd=10.0)
    tracker.add("claude-opus-5-5", 1000, 500)
    tracker.add("claude-haiku-4-5", 1000, 100)
    assert tracker.spent == pytest.approx(
        cost_usd("claude-opus-5-5", 1000, 500) + cost_usd("claude-haiku-4-5", 1000, 100)
    )
    assert set(tracker.by_model) == {"claude-opus-5-5", "claude-haiku-4-5"}


def test_tracker_raises_once_cap_is_crossed():
    tracker = BudgetTracker(cap_usd=0.01)
    with pytest.raises(BudgetExceeded):
        tracker.add("claude-opus-5-5", 0, 10_000)


def test_tracker_can_check_headroom_before_a_call():
    tracker = BudgetTracker(cap_usd=1.0)
    assert tracker.can_afford("claude-opus-5-5", 1000, 2000)
    tracker.add("claude-opus-5-5", 0, 48_000)
    assert not tracker.can_afford("claude-opus-5-5", 1000, 4000)


def test_prices_cover_the_three_benchmark_models():
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"} <= set(PRICES)


def test_tracker_refuses_a_cap_above_the_hard_limit():
    from jev_router.budget import HARD_LIMIT_USD

    assert HARD_LIMIT_USD < 15.0  # leaves margin under the user's $15 of credit
    BudgetTracker(cap_usd=HARD_LIMIT_USD)
    with pytest.raises(ValueError):
        BudgetTracker(cap_usd=HARD_LIMIT_USD + 0.01)


def test_jev_cost_is_input_only_at_the_published_rate():
    from jev_router.budget import JEV_INPUT_USD_PER_M, jev_cost_usd

    assert JEV_INPUT_USD_PER_M == pytest.approx(0.042)
    assert jev_cost_usd(1_000_000) == pytest.approx(0.042)
    assert jev_cost_usd(850) == pytest.approx(850 * 0.042 / 1_000_000)
    assert jev_cost_usd(0) == 0.0
