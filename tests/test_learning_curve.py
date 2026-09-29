import json

import numpy as np
import pytest

from jev_router.ablation import Context, build_specs, evaluate_spec
from jev_router.anthropic_runner import MODELS
from jev_router.learning_curve import (
    aggregate_curve,
    curve_rows,
    load_embeddings_readonly,
    reproduction_check,
    restrict,
    training_subset,
)
from jev_router.questions_v2 import FEATURE_NAMES
from jev_router.routing_eval import ItemOutcome, split_ids


def make_data(n=260, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    outcomes = []
    for i in range(n):
        p = 1 / (1 + np.exp(-(1.5 - 1.5 * X[i, 0])))
        correct = {
            MODELS[0]: bool(rng.random() < 0.8 * p),
            MODELS[1]: bool(rng.random() < min(0.98, p + 0.1)),
            MODELS[2]: bool(rng.random() < 0.97),
        }
        cost = {MODELS[0]: 0.001, MODELS[1]: 0.004, MODELS[2]: 0.010}
        outcomes.append(ItemOutcome(f"i{i}", "mmlu", 300, correct, cost))
    texts = [f"word{i % 7} token{i % 5} extra{(i * 3) % 11} filler{i % 13}" for i in range(n)]
    return Context(X, texts, None), outcomes, np.full(n, 4e-5)


def dev_test_ids(outcomes, seed):
    return split_ids([o.item_id for o in outcomes], seed)


def test_training_subset_is_deterministic_nested_and_capped():
    dev, _ = dev_test_ids(make_data()[1], 3)
    a, b = training_subset(dev, 20, 3), training_subset(dev, 20, 3)
    assert a == b and len(a) == 20 and len(set(a)) == 20
    assert set(training_subset(dev, 10, 3)) <= set(training_subset(dev, 20, 3))  # nested across sizes
    assert set(training_subset(dev, 20, 3)) != set(training_subset(dev, 20, 4))  # seed matters
    assert set(training_subset(dev, 10**6, 3)) == set(dev)  # more than available -> everything
    assert set(a) <= set(dev)


def test_restrict_keeps_rows_aligned_and_in_original_order():
    ctx, outcomes, costs = make_data(40)
    keep = {"i5", "i2", "i30"}
    ctx2, out2, costs2 = restrict(ctx, outcomes, costs, keep)
    assert [o.item_id for o in out2] == ["i2", "i5", "i30"]
    assert np.array_equal(ctx2.jev_matrix, ctx.jev_matrix[[2, 5, 30]])
    assert list(ctx2.texts) == [ctx.texts[2], ctx.texts[5], ctx.texts[30]]
    assert len(costs2) == 3


def test_restricted_run_keeps_the_full_test_half_and_shrinks_only_the_dev_half():
    ctx, outcomes, costs = make_data()
    _, test = dev_test_ids(outcomes, 1)
    rows = curve_rows(ctx, outcomes, costs, seed=1, sizes=(40, 10**6), spec_names=("jev_plus_length",), floors=(0.97,))
    small = [r for r in rows if r["n_requested"] == 40]
    full = [r for r in rows if r["n_requested"] == 10**6]
    assert small and full
    assert all(r["n_test"] == len(test) for r in rows)  # the same untouched test half every time
    assert {r["n_train"] for r in small} == {40}
    assert {r["n_train"] for r in full} == {len(outcomes) - len(test)}


def test_full_size_reproduces_the_ablation_evaluation_exactly():
    ctx, outcomes, costs = make_data()
    direct = {
        (r["rule"], r["floor"]): r
        for r in evaluate_spec(build_specs(ctx)["jev_plus_length"], ctx, outcomes, costs, 2, (0.97,))
    }
    rows = curve_rows(ctx, outcomes, costs, seed=2, sizes=(10**6,), spec_names=("jev_plus_length",), floors=(0.97,))
    assert rows
    for row in rows:
        ref = direct[(row["rule"], row["floor"])]
        assert row["gap_pct"] == pytest.approx(ref["gap_pct"])
        assert row["accuracy"] == pytest.approx(ref["accuracy"])
        assert row["total_cost"] == pytest.approx(ref["total_cost"])


def test_tiny_training_sets_do_not_crash_any_feature_set():
    ctx, outcomes, costs = make_data()
    rows = curve_rows(ctx, outcomes, costs, seed=5, sizes=(12,), spec_names=("tfidf", "jev_no_length", "jev_plus_length"), floors=(0.95,))
    assert {r["spec"] for r in rows} == {"tfidf", "jev_no_length", "jev_plus_length"}
    assert all(np.isfinite(r["gap_pct"]) for r in rows)


def test_aggregate_curve_groups_by_training_size():
    def row(seed, n, gap):
        return {"spec": "s", "rule": "loss", "floor": 0.97, "seed": seed, "n_requested": n, "n_train": n,
                "gap_pct": gap, "accuracy": 0.9, "savings_vs_opus_pct": 30.0, "acc_vs_random_pp": 1.0,
                "above_single_max": False}

    runs = [[row(0, 100, -2.0), row(0, 400, -6.0)], [row(1, 100, 2.0), row(1, 400, -4.0)]]
    out = aggregate_curve(runs)
    assert set(out) == {100, 400}
    assert out[100]["s:loss:0.97"]["gap_mean"] == pytest.approx(0.0)
    assert out[100]["s:loss:0.97"]["share_cheaper_than_frontier"] == pytest.approx(0.5)
    assert out[400]["s:loss:0.97"]["gap_mean"] == pytest.approx(-5.0)


def test_reproduction_check_reports_the_largest_difference():
    cell = {"gap_mean": -5.0, "share_cheaper_than_frontier": 0.8, "accuracy_mean": 0.9}
    same = reproduction_check({"s:loss:0.97": cell}, {"s:loss:0.97": dict(cell)})
    assert same["cells_compared"] == 1 and same["max_abs_diff"] == pytest.approx(0.0)
    off = reproduction_check({"s:loss:0.97": cell}, {"s:loss:0.97": {**cell, "gap_mean": -4.5}})
    assert off["max_abs_diff"] == pytest.approx(0.5)
    none = reproduction_check({"a:loss:0.97": cell}, {"b:loss:0.97": cell})
    assert none["cells_compared"] == 0


def test_readonly_embeddings_select_rows_by_id_and_never_write(tmp_path):
    np.save(tmp_path / "embeddings.npy", np.arange(12.0).reshape(3, 4))
    (tmp_path / "embeddings_meta.json").write_text(json.dumps({"digest": "x", "ids": ["a", "b", "c"]}))
    before = sorted(p.name for p in tmp_path.iterdir())
    out = load_embeddings_readonly(["c", "a"], ["tc", "ta"], tmp_path, embed=lambda t: 1 / 0)
    assert out.tolist() == [[8.0, 9.0, 10.0, 11.0], [0.0, 1.0, 2.0, 3.0]]
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_readonly_embeddings_fall_back_in_memory_without_touching_disk(tmp_path):
    out = load_embeddings_readonly(["x"], ["tx"], tmp_path, embed=lambda t: np.ones((len(t), 4)))
    assert out.shape == (1, 4)
    assert list(tmp_path.iterdir()) == []
