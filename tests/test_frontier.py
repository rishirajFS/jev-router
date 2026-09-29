import pytest

from jev_router.frontier import (
    frontier_cost,
    frontier_gap_pct,
    frontier_summary,
    lower_hull,
)

HAIKU = (0.810, 0.1532)
SONNET = (0.925, 0.6465)
OPUS = (0.962, 1.0626)


def test_lower_hull_keeps_all_points_of_a_convex_frontier():
    assert lower_hull([OPUS, HAIKU, SONNET]) == [HAIKU, SONNET, OPUS]


def test_lower_hull_drops_a_dominated_point():
    dominated = (0.90, 0.90)  # Sonnet is more accurate and cheaper
    assert lower_hull([HAIKU, dominated, SONNET, OPUS]) == [HAIKU, SONNET, OPUS]


def test_lower_hull_drops_a_point_above_the_mixture_line():
    # Sonnet-like point that costs more than mixing Haiku and Opus at that accuracy.
    pricey_middle = (0.90, 0.90)
    line_cost = HAIKU[1] + (0.90 - HAIKU[0]) / (OPUS[0] - HAIKU[0]) * (OPUS[1] - HAIKU[1])
    assert pricey_middle[1] > line_cost
    assert lower_hull([HAIKU, pricey_middle, OPUS]) == [HAIKU, OPUS]


def test_lower_hull_resolves_equal_accuracy_ties_to_the_cheaper_point():
    assert lower_hull([(0.9, 0.5), (0.9, 0.3), (0.95, 0.9)]) == [(0.9, 0.3), (0.95, 0.9)]


def test_lower_hull_single_point_and_empty():
    assert lower_hull([HAIKU]) == [HAIKU]
    assert lower_hull([]) == []


def test_lower_hull_does_not_mutate_its_input():
    points = [OPUS, HAIKU, SONNET]
    lower_hull(points)
    assert points == [OPUS, HAIKU, SONNET]


def test_frontier_cost_hits_vertices_exactly():
    points = [HAIKU, SONNET, OPUS]
    assert frontier_cost(points, HAIKU[0]) == pytest.approx(HAIKU[1])
    assert frontier_cost(points, SONNET[0]) == pytest.approx(SONNET[1])
    assert frontier_cost(points, OPUS[0]) == pytest.approx(OPUS[1])


def test_frontier_cost_interpolates_between_neighbours():
    points = [HAIKU, SONNET, OPUS]
    expected = SONNET[1] + (0.935 - SONNET[0]) / (OPUS[0] - SONNET[0]) * (OPUS[1] - SONNET[1])
    assert frontier_cost(points, 0.935) == pytest.approx(expected)
    # fixture accuracies are rounded, so this is near (not equal to) the unrounded 0.755
    assert frontier_cost(points, 0.935) == pytest.approx(0.755, abs=0.005)


def test_frontier_cost_below_the_cheapest_accuracy_costs_the_cheapest_point():
    assert frontier_cost([HAIKU, SONNET, OPUS], 0.5) == pytest.approx(HAIKU[1])


def test_frontier_cost_above_the_maximum_accuracy_is_none():
    assert frontier_cost([HAIKU, SONNET, OPUS], 0.99) is None


def test_frontier_cost_single_point():
    assert frontier_cost([HAIKU], 0.7) == pytest.approx(HAIKU[1])
    assert frontier_cost([HAIKU], HAIKU[0]) == pytest.approx(HAIKU[1])
    assert frontier_cost([HAIKU], 0.9) is None


def test_frontier_cost_requires_at_least_one_point():
    with pytest.raises(ValueError):
        frontier_cost([], 0.5)


def test_frontier_gap_pct_is_positive_when_more_expensive():
    points = [HAIKU, SONNET, OPUS]
    frontier = frontier_cost(points, 0.935)
    assert frontier_gap_pct(frontier * 1.09, 0.935, points) == pytest.approx(9.0)
    assert frontier_gap_pct(frontier * 0.8, 0.935, points) == pytest.approx(-20.0)


def test_frontier_gap_pct_is_none_when_accuracy_is_unreachable():
    assert frontier_gap_pct(1.0, 0.99, [HAIKU, SONNET, OPUS]) is None


def test_frontier_summary_reads_the_report_baselines():
    baselines = {
        "always_haiku": {"accuracy": 0.81, "total_cost": 0.15},
        "always_sonnet": {"accuracy": 0.92, "total_cost": 0.64},
        "always_opus": {"accuracy": 0.96, "total_cost": 1.06},
        "oracle": {"accuracy": 0.98, "total_cost": 0.33},  # not a real strategy, ignored
    }
    assert frontier_summary(baselines) == [(0.81, 0.15), (0.92, 0.64), (0.96, 1.06)]
