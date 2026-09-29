import numpy as np
import pytest

from jev_router.learned_router import (
    Rule,
    apply_rule,
    fit_prob_models,
    frontier_cost,
    oof_probs,
    predict_probs,
    select_loss_rule,
    select_threshold_rule,
    tiers_by_expected_loss,
    tiers_by_thresholds,
)

COSTS = np.array([0.001, 0.004, 0.010])


def synth(n=300, seed=0):
    rng = np.random.RandomState(seed)
    hard = rng.rand(n) < 0.4
    X = np.column_stack([hard + 0.2 * rng.randn(n), rng.randn(n)])
    correct = np.column_stack([~hard, ~hard | (rng.rand(n) < 0.5), np.ones(n, dtype=bool)])
    return X, correct.astype(float), hard


def test_fit_prob_models_learns_the_informative_feature_from_dev_only():
    X, Y, hard = synth()
    models = fit_prob_models(X, Y, c=1.0)
    P = predict_probs(models, X)
    assert P.shape == (len(X), 3)
    assert P[~hard, 0].mean() > 0.8 and P[hard, 0].mean() < 0.3
    assert np.all((P >= 0) & (P <= 1))


def test_constant_labels_fall_back_to_a_constant_predictor():
    X, Y, _ = synth()
    P = predict_probs(fit_prob_models(X, Y, c=1.0), X)
    assert np.allclose(P[:, 2], 1.0)  # Opus always correct in the synthetic data


def test_oof_probs_are_deterministic_and_out_of_fold():
    X, Y, hard = synth()
    a = oof_probs(X, Y, c=1.0, folds=5, seed=0)
    b = oof_probs(X, Y, c=1.0, folds=5, seed=0)
    assert np.allclose(a, b) and a.shape == (len(X), 3)
    in_sample = predict_probs(fit_prob_models(X, Y, c=1.0), X)
    assert np.abs(a - in_sample).max() > 0  # not simply the in-sample fit


def test_tiers_by_thresholds_cascades_haiku_then_sonnet_then_opus():
    P = np.array([[0.9, 0.9, 0.99], [0.2, 0.8, 0.99], [0.2, 0.3, 0.99]])
    assert list(tiers_by_thresholds(P, t_h=0.5, t_s=0.5)) == [0, 1, 2]


def test_tiers_by_expected_loss_trades_error_against_cost():
    P = np.array([[0.9, 0.95, 0.99]])
    assert list(tiers_by_expected_loss(P, COSTS, lam=0.0)) == [0]
    assert list(tiers_by_expected_loss(P, COSTS, lam=1.0)) == [2]


def oracle_probs(hard):
    return np.column_stack([np.where(hard, 0.05, 0.95), np.where(hard, 0.5, 0.95), np.full(len(hard), 0.99)])


def test_select_threshold_rule_is_cheaper_than_all_opus_and_meets_the_floor():
    _, Y, hard = synth()
    cost = np.tile(COSTS, (len(Y), 1))
    P = oracle_probs(hard)
    rule = select_threshold_rule(P, Y, cost, floor_acc=0.99)
    tiers = apply_rule(rule, P, COSTS)
    accuracy = Y[np.arange(len(Y)), tiers].mean()
    assert rule.kind == "threshold"
    assert accuracy >= 0.99
    assert cost[np.arange(len(Y)), tiers].sum() < cost[:, 2].sum()


def test_select_loss_rule_meets_the_floor():
    _, Y, hard = synth()
    cost = np.tile(COSTS, (len(Y), 1))
    P = oracle_probs(hard)
    rule = select_loss_rule(P, Y, cost, floor_acc=0.9)
    tiers = apply_rule(rule, P, COSTS)
    assert rule.kind == "loss"
    assert Y[np.arange(len(Y)), tiers].mean() >= 0.9


def test_unreachable_floor_falls_back_to_all_opus():
    _, Y, hard = synth()
    cost = np.tile(COSTS, (len(Y), 1))
    P = oracle_probs(hard)
    for select in (select_threshold_rule, select_loss_rule):
        rule = select(P, Y, cost, floor_acc=1.5)
        assert rule.kind == "opus"
        assert set(apply_rule(rule, P, COSTS)) == {2}


def test_rule_is_frozen():
    with pytest.raises(Exception):
        Rule("opus", ()).kind = "x"  # type: ignore[misc]


def test_frontier_cost_interpolates_between_single_models():
    points = [(0.8, 0.1), (0.9, 0.4), (0.96, 1.0)]
    assert frontier_cost(points, 0.85) == pytest.approx(0.25)
    assert frontier_cost(points, 0.9) == pytest.approx(0.4)
    assert frontier_cost(points, 0.96) == pytest.approx(1.0)
    assert frontier_cost(points, 0.7) == pytest.approx(0.1)  # below the cheapest model
