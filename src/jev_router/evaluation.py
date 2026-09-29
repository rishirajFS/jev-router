"""Metrics for judging whether routing decisions are apt."""

from collections import defaultdict
from collections.abc import Sequence


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("spearman needs two equal-length sequences of size >= 2")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def confusion_matrix(
    expected: Sequence[int], predicted: Sequence[int]
) -> dict[int, dict[int, int]]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    matrix: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for e, p in zip(expected, predicted):
        matrix[e][p] += 1
    return {e: dict(row) for e, row in matrix.items()}


def routing_report(
    expected: Sequence[int], predicted: Sequence[int]
) -> dict[str, float | int]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    n = len(expected)
    if n == 0:
        raise ValueError("routing_report needs at least one example")
    exact = sum(e == p for e, p in zip(expected, predicted))
    over = sum(p > e for e, p in zip(expected, predicted))
    under = sum(p < e for e, p in zip(expected, predicted))
    near = sum(abs(e - p) <= 1 for e, p in zip(expected, predicted))
    return {
        "n": n,
        "exact_accuracy": exact / n,
        "over_routed": over / n,
        "under_routed": under / n,
        "within_one": near / n,
    }
