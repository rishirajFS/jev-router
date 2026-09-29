"""Learned router: per-tier P(correct) from Jev features, fit on the dev half only."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_TIERS = 3  # 0 = Haiku, 1 = Sonnet, 2 = Opus
THRESHOLD_GRID = tuple(round(0.02 * k, 2) for k in range(51))
LAMBDA_GRID = tuple(float(x) for x in np.logspace(-4, 0.5, 70))


class _Constant:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        column = np.full(len(X), self.p)
        return np.column_stack([1.0 - column, column])


@dataclass(frozen=True)
class Rule:
    kind: str  # "threshold" | "loss" | "opus"
    params: tuple[float, ...]


def _fit_one(X: np.ndarray, y: np.ndarray, c: float):
    """A fitted regularized logistic model, or a constant when only one class is present."""
    labels = y.astype(int)
    if len(set(labels.tolist())) < 2:
        return _Constant(float(labels[0]))
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(X, labels)


def fit_prob_models(X: np.ndarray, Y: np.ndarray, c: float) -> tuple:
    return tuple(_fit_one(X, Y[:, k], c) for k in range(Y.shape[1]))


def predict_probs(models: Sequence, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def oof_probs(X: np.ndarray, Y: np.ndarray, c: float, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(correct), so dev-set thresholds are not chosen on in-sample fits."""
    out = np.zeros(Y.shape)
    for k in range(Y.shape[1]):
        y = Y[:, k].astype(int)
        if len(set(y.tolist())) < 2 or min(np.bincount(y)) < folds:
            out[:, k] = _fit_one(X, y, c).predict_proba(X)[:, 1]
            continue
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train, held in splitter.split(X, y):
            out[held, k] = _fit_one(X[train], y[train], c).predict_proba(X[held])[:, 1]
    return out


def tiers_by_thresholds(P: np.ndarray, t_h: float, t_s: float) -> np.ndarray:
    return np.where(P[:, 0] >= t_h, 0, np.where(P[:, 1] >= t_s, 1, 2))


def tiers_by_expected_loss(P: np.ndarray, costs: np.ndarray, lam: float) -> np.ndarray:
    """liteLLM-style rule: pick the tier minimizing lam * P(wrong) + dollar cost."""
    return np.argmin(lam * (1.0 - P) + costs[None, :], axis=1)


def apply_rule(rule: Rule, P: np.ndarray, mean_costs: np.ndarray) -> np.ndarray:
    if rule.kind == "threshold":
        return tiers_by_thresholds(P, *rule.params)
    if rule.kind == "loss":
        return tiers_by_expected_loss(P, mean_costs, rule.params[0])
    return np.full(len(P), 2)


def _score(tiers: np.ndarray, correct: np.ndarray, cost: np.ndarray) -> tuple[float, float]:
    rows = np.arange(len(tiers))
    return float(correct[rows, tiers].mean()), float(cost[rows, tiers].sum())


def _cheapest_feasible(
    candidates: Iterable[tuple[Rule, np.ndarray]],
    correct: np.ndarray,
    cost: np.ndarray,
    floor_acc: float,
) -> Rule:
    best: tuple[float, float, Rule] | None = None
    for rule, tiers in candidates:
        accuracy, total = _score(tiers, correct, cost)
        if accuracy + 1e-12 < floor_acc:
            continue
        if best is None or (total, -accuracy) < (best[0], -best[1]):
            best = (total, accuracy, rule)
    return best[2] if best else Rule("opus", ())


def select_threshold_rule(P, correct, cost, floor_acc: float) -> Rule:
    candidates = (
        (Rule("threshold", (t_h, t_s)), tiers_by_thresholds(P, t_h, t_s))
        for t_h in THRESHOLD_GRID
        for t_s in THRESHOLD_GRID
    )
    return _cheapest_feasible(candidates, correct, cost, floor_acc)


def select_loss_rule(P, correct, cost, floor_acc: float) -> Rule:
    mean_costs = cost.mean(axis=0)
    candidates = (
        (Rule("loss", (lam,)), tiers_by_expected_loss(P, mean_costs, lam))
        for lam in LAMBDA_GRID
    )
    return _cheapest_feasible(candidates, correct, cost, floor_acc)


def frontier_cost(points: Sequence[tuple[float, float]], accuracy: float) -> float:
    """Cheapest cost of randomly mixing single models to reach `accuracy` (linear hull)."""
    pts = sorted(points)
    if accuracy <= pts[0][0]:
        return min(c for a, c in pts if a == pts[0][0])
    best = float("inf")
    for a1, c1 in pts:
        for a2, c2 in pts:
            if a1 <= accuracy <= a2 and a2 > a1:
                best = min(best, c1 + (accuracy - a1) / (a2 - a1) * (c2 - c1))
            elif a1 == a2 == accuracy:
                best = min(best, c1, c2)
    return best
