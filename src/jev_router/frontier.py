"""Single-model frontier: the cheapest cost of reaching an accuracy by mixing always-one-model strategies.

A router is only worth its complexity if it beats this line. Randomly sending a fraction of
traffic to each single model gives any (accuracy, cost) on a straight segment between two
strategies, so the best such mixture is the lower convex hull of the single-model points.
"""

from collections.abc import Mapping, Sequence

Point = tuple[float, float]  # (accuracy, cost)

_SINGLE_MODEL_KEYS = ("always_haiku", "always_sonnet", "always_opus")
_EPS = 1e-9


def _non_dominated(points: Sequence[Point]) -> list[Point]:
    """Drops points that another point matches in accuracy for no more cost."""
    kept: list[Point] = []
    cheapest_at_or_above = float("inf")
    for accuracy, cost in sorted(points, key=lambda p: (-p[0], p[1])):
        if cost < cheapest_at_or_above:
            kept.append((accuracy, cost))
            cheapest_at_or_above = cost
    return sorted(kept)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def lower_hull(points: Sequence[Point]) -> list[Point]:
    """Efficient frontier sorted by accuracy: non-dominated points on the lower convex hull."""
    hull: list[Point] = []
    for point in _non_dominated(points):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull


def frontier_cost(points: Sequence[Point], accuracy: float) -> float | None:
    """Cheapest expected cost of reaching `accuracy` by randomly mixing the given strategies.

    At or below the least accurate frontier point this is that point's cost (the cheapest
    strategy overall). Above the most accurate point no mixture can reach it: returns None.
    """
    hull = lower_hull(points)
    if not hull:
        raise ValueError("frontier_cost needs at least one point")
    if accuracy > hull[-1][0] + _EPS:
        return None
    if accuracy <= hull[0][0]:
        return hull[0][1]
    for (a1, c1), (a2, c2) in zip(hull, hull[1:]):
        if a1 <= accuracy <= a2:
            weight = (accuracy - a1) / (a2 - a1)
            return c1 + weight * (c2 - c1)
    return hull[-1][1]


def frontier_gap_pct(cost: float, accuracy: float, points: Sequence[Point]) -> float | None:
    """100 * (cost / frontier cost - 1). Positive: more expensive than the best single-model mix.

    None when the accuracy is beyond every mixture, or the frontier cost is zero.
    """
    frontier = frontier_cost(points, accuracy)
    if frontier is None or frontier <= 0:
        return None
    return 100.0 * (cost / frontier - 1.0)


def frontier_summary(baselines: Mapping[str, Mapping[str, float]]) -> list[Point]:
    """(accuracy, cost) points for the always-one-model rows of an experiment report."""
    return [
        (baselines[key]["accuracy"], baselines[key]["total_cost"])
        for key in _SINGLE_MODEL_KEYS
    ]
