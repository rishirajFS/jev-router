import numpy as np
import pytest

from jev_router.ablation import (
    Context,
    aggregate,
    build_specs,
    evaluate_spec,
    load_or_compute_embeddings,
    tfidf_svd_features,
)
from jev_router.questions_v2 import FEATURE_NAMES
from jev_router.routing_eval import HAIKU, OPUS, SONNET, ItemOutcome


def texts(n=60):
    return [f"question about topic {i % 7} with word{i} and shared words here" for i in range(n)]


def test_tfidf_svd_is_fit_on_the_dev_texts_only():
    dev, test = texts(40), texts(20)
    xd1, xt1 = tfidf_svd_features(dev, test, dims=5)
    xd2, _ = tfidf_svd_features(dev, ["completely different unseen tokens zzz qqq"] * 20, dims=5)
    assert xd1.shape == (40, 5) and xt1.shape == (20, 5)
    assert np.allclose(xd1, xd2)  # changing the test texts must not change the dev features


def test_tfidf_svd_caps_dimensions_by_data_size():
    xd, xt = tfidf_svd_features(texts(6), texts(3), dims=64)
    assert xd.shape[1] == xt.shape[1] <= 5


def make_context(n=40, with_embeddings=True):
    rng = np.random.default_rng(0)
    jev = rng.normal(size=(n, len(FEATURE_NAMES)))
    return Context(
        jev_matrix=jev,
        texts=texts(n),
        embeddings=rng.normal(size=(n, 12)) if with_embeddings else None,
    )


def test_build_specs_have_the_intended_columns_and_jev_flags():
    ctx = make_context()
    specs = build_specs(ctx)
    dev, test = np.arange(0, 30), np.arange(30, 40)
    shapes = {name: spec.build(ctx, dev, test)[0].shape[1] for name, spec in specs.items()}
    assert shapes["length_only"] == 1
    assert shapes["jev_no_length"] == len(FEATURE_NAMES) - 1
    assert shapes["jev_plus_length"] == len(FEATURE_NAMES)
    assert shapes["embedding"] == 12
    assert shapes["embedding_plus_length"] == 13
    assert shapes["full_plus_embedding"] == len(FEATURE_NAMES) + 12
    assert {n for n, s in specs.items() if s.uses_jev} == {"jev_no_length", "jev_plus_length", "full_plus_embedding"}
    assert "tfidf" in specs


def test_build_specs_skips_embedding_sets_when_unavailable():
    specs = build_specs(make_context(with_embeddings=False))
    assert not any("embedding" in name for name in specs)
    assert {"length_only", "tfidf", "jev_no_length", "jev_plus_length"} <= set(specs)


def test_length_only_uses_the_log_chars_column():
    ctx = make_context()
    specs = build_specs(ctx)
    xd, _ = specs["length_only"].build(ctx, np.arange(30), np.arange(30, 40))
    expected = ctx.jev_matrix[:30, FEATURE_NAMES.index("log_chars")].reshape(-1, 1)
    assert np.allclose(xd, expected)


def test_embeddings_are_computed_once_then_read_from_cache(tmp_path):
    calls = []

    def fake_embed(batch):
        calls.append(len(batch))
        return np.arange(len(batch) * 3, dtype=float).reshape(len(batch), 3)

    ids = ["a", "b", "c"]
    first = load_or_compute_embeddings(ids, ["x", "y", "z"], tmp_path, fake_embed)
    second = load_or_compute_embeddings(ids, ["x", "y", "z"], tmp_path, fake_embed)
    assert calls == [3]
    assert np.allclose(first, second)
    load_or_compute_embeddings(ids, ["x", "y", "CHANGED"], tmp_path, fake_embed)
    assert len(calls) == 2  # changed text invalidates the cache


def synthetic(n=240, seed=0):
    """Hard items (feature > 0) need Opus; easy items are solved by every model."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    outcomes = []
    for i, s in enumerate(signal):
        hard = s > 0.2
        outcomes.append(
            ItemOutcome(
                f"i{i}", "mmlu", 100,
                {HAIKU: not hard, SONNET: not hard, OPUS: True},
                {HAIKU: 0.001, SONNET: 0.004, OPUS: 0.010},
            )
        )
    jev = np.zeros((n, len(FEATURE_NAMES)))
    jev[:, FEATURE_NAMES.index("d_score")] = signal
    return outcomes, Context(jev_matrix=jev, texts=[f"t{i}" for i in range(n)], embeddings=None)


def test_evaluate_spec_finds_a_router_that_beats_the_frontier_and_charges_overhead_only_for_jev():
    outcomes, ctx = synthetic()
    specs = build_specs(ctx)
    jev_costs = np.full(len(outcomes), 0.0005)
    rows = evaluate_spec(specs["jev_no_length"], ctx, outcomes, jev_costs, seed=3, floors=(0.97,))
    assert {r["rule"] for r in rows} == {"threshold", "loss"}
    threshold = next(r for r in rows if r["rule"] == "threshold")
    assert threshold["gap_pct"] < 0 and threshold["overhead_cost"] > 0
    assert threshold["accuracy"] >= 0.9
    rows_len = evaluate_spec(specs["length_only"], ctx, outcomes, jev_costs, seed=3, floors=(0.97,))
    assert all(r["overhead_cost"] == 0 for r in rows_len)  # local features cost nothing
    # length is constant here, so it carries no signal and cannot beat the frontier
    assert next(r for r in rows_len if r["rule"] == "threshold")["gap_pct"] > threshold["gap_pct"]


def test_aggregate_summarizes_gap_share_and_accuracy():
    def row(spec, gap, acc=0.9):
        return {"spec": spec, "rule": "threshold", "floor": 0.97, "gap_pct": gap, "accuracy": acc,
                "savings_vs_opus_pct": 30.0, "acc_vs_random_pp": 2.0}

    out = aggregate([[row("a", -4), row("b", 5)], [row("a", 2), row("b", 7)], [row("a", -6), row("b", 9)]])
    a = out["a:threshold:0.97"]
    assert a["gap_mean"] == pytest.approx(-8 / 3)
    assert a["share_cheaper_than_frontier"] == pytest.approx(2 / 3)
    assert a["n_splits"] == 3
    assert out["b:threshold:0.97"]["share_cheaper_than_frontier"] == 0.0
    with pytest.raises(ValueError):
        aggregate([])


def test_capped_frontier_cost_interpolates_and_caps_above_the_best_single_model():
    from jev_router.ablation import capped_frontier_cost

    points = [(0.8, 1.0), (0.9, 3.0), (1.0, 8.0)]
    assert capped_frontier_cost(points, 0.85) == pytest.approx(2.0)   # halfway between the first two
    assert capped_frontier_cost(points, 0.9) == pytest.approx(3.0)    # exact vertex
    assert capped_frontier_cost(points, 1.2) == pytest.approx(8.0)    # above every point: cost of the best model
    assert capped_frontier_cost(points, 0.5) == pytest.approx(1.0)    # below every point: cheapest point


def test_paired_contrast_compares_two_specs_split_by_split():
    from jev_router.ablation import paired_contrast

    def row(spec, seed, gap):
        return {"spec": spec, "rule": "threshold", "floor": 0.97, "seed": seed, "gap_pct": gap}

    runs = [[row("a", s, -10.0 + s), row("b", s, -4.0)] for s in range(4)]  # a: -10,-9,-8,-7 ; b: -4
    out = paired_contrast(runs, "a", "b", "threshold", 0.97)
    assert out["mean_diff_pct"] == pytest.approx(-4.5)          # diffs -6,-5,-4,-3: a is 4.5 points cheaper on average
    assert out["share_a_cheaper"] == 1.0
    assert out["n_splits"] == 4
    assert out["t_stat"] < -5
    assert paired_contrast([[row("a", 0, 1.0), row("b", 0, 1.0)]], "a", "b", "threshold", 0.97)["share_a_cheaper"] == 0.0
    with pytest.raises(ValueError):
        paired_contrast([[row("a", 0, 1.0)]], "a", "missing", "threshold", 0.97)
