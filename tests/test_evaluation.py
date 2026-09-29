import pytest

from jev_router.evaluation import (
    confusion_matrix,
    routing_report,
    spearman,
)


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_and_constant_input():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_confusion_matrix_counts():
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0])
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1
    assert matrix[2][0] == 1


def test_routing_report_splits_under_and_over_routing():
    report = routing_report(expected=[0, 1, 2, 2], predicted=[0, 2, 1, 2])
    assert report["exact_accuracy"] == pytest.approx(0.5)
    assert report["over_routed"] == pytest.approx(0.25)
    assert report["under_routed"] == pytest.approx(0.25)
    assert report["within_one"] == pytest.approx(1.0)


def test_routing_report_rejects_length_mismatch():
    with pytest.raises(ValueError):
        routing_report([0], [0, 1])


def test_routing_report_rejects_empty_input():
    with pytest.raises(ValueError):
        routing_report([], [])


def test_confusion_matrix_rejects_length_mismatch():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])
