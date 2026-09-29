import numpy as np
import pytest

from jev_router.ablation import LENGTH, Context, build_specs
from jev_router.questions_v2 import FEATURE_NAMES
from jev_router.ablation_fresh import (
    combine_contexts,
    evaluate_frozen_spec,
    gap_or_none,
    margin_holds,
)
from jev_router.routing_eval import HAIKU, OPUS, SONNET, ItemOutcome

NON_JEV = ("length_only", "tfidf", "embedding", "embedding_plus_length")


def test_gap_or_none_flags_accuracy_above_the_best_single_model():
    points = [(0.8, 0.2), (0.9, 0.6), (0.95, 1.0)]
    gap, above = gap_or_none(cost=0.8, accuracy=0.93, points=points)
    assert above is False and gap is not None
    gap, above = gap_or_none(cost=1.0, accuracy=0.97, points=points)
    assert above is True and gap is None  # never scored as -100%


def gaps(jev, **others):
    base = {name: 5.0 for name in NON_JEV}
    base.update(others)
    return {"jev_plus_length": jev, **base}


def test_margin_holds_needs_a_lower_gap_than_every_non_jev_router_at_two_floors():
    by_floor = {
        0.99: gaps(-10.0),
        0.97: gaps(-8.0),
        0.95: gaps(6.0),  # jev is worse here
    }
    assert margin_holds(by_floor) is True
    by_floor[0.97] = gaps(-8.0, tfidf=-9.0)  # tfidf beats jev at 0.97 too
    assert margin_holds(by_floor) is False


def test_margin_holds_ignores_undefined_gaps_and_counts_a_flagged_jev_gap_as_no_win():
    by_floor = {0.99: gaps(-3.0, tfidf=None), 0.97: gaps(None), 0.95: gaps(-3.0)}
    assert margin_holds(by_floor) is True  # floors 0.99 and 0.95; the None router is ignored, None jev is not a win
    assert margin_holds({0.99: gaps(None), 0.97: gaps(None), 0.95: gaps(-3.0)}) is False


def make_context(n, offset, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURE_NAMES)))  # real column layout, so LENGTH indexes correctly
    x[:, LENGTH] += offset
    texts = [f"question {i} about topic {'alpha' if i % 2 else 'beta'}" for i in range(n)]
    return Context(x, texts, None), x


def synthetic_outcomes(x, prefix, fresh_flip=False):
    outcomes = []
    for i, row in enumerate(x):
        hard = row[0] > 0
        correct = {HAIKU: (not hard) != fresh_flip, SONNET: True, OPUS: True}
        outcomes.append(ItemOutcome(f"{prefix}{i}", "mmlu", 100, correct, {HAIKU: 0.001, SONNET: 0.004, OPUS: 0.010}))
    return outcomes


def test_combine_contexts_stacks_matrices_texts_and_index_ranges():
    orig, _ = make_context(6, 0.0, 1)
    fresh, _ = make_context(4, 0.0, 2)
    combined, fresh_idx = combine_contexts(orig, fresh)
    assert combined.jev_matrix.shape == (10, len(FEATURE_NAMES))
    assert len(combined.texts) == 10
    assert list(fresh_idx) == [6, 7, 8, 9]
    assert combined.embeddings is None


def test_frozen_evaluation_never_uses_fresh_outcomes_to_choose_tiers():
    orig_ctx, ox = make_context(80, 0.0, 3)
    fresh_ctx, fx = make_context(40, 0.0, 4)
    combined, fresh_idx = combine_contexts(orig_ctx, fresh_ctx)
    orig_out = synthetic_outcomes(ox, "o")
    dev_idx = np.arange(0, 40)
    spec = build_specs(combined)["length_only"]
    common = dict(spec=spec, ctx=combined, dev_idx=dev_idx, fresh_idx=fresh_idx, dev_outcomes=orig_out[:40],
                  fresh_jev_costs=np.zeros(40), floors=(0.97,), families=["mmlu"] * 40)
    a = evaluate_frozen_spec(fresh_outcomes=synthetic_outcomes(fx, "f"), **common)
    b = evaluate_frozen_spec(fresh_outcomes=synthetic_outcomes(fx, "f", fresh_flip=True), **common)
    tiers_a = {(r["rule"], r["floor"]): r["tier_counts"] for r in a if r["group"] == "combined"}
    tiers_b = {(r["rule"], r["floor"]): r["tier_counts"] for r in b if r["group"] == "combined"}
    assert tiers_a == tiers_b  # tier choices depend on the dev fit and fresh features only
    assert {r["group"] for r in a} == {"combined", "routerbench"}
    row = next(r for r in a if r["group"] == "combined" and r["rule"] == "threshold")
    assert {"accuracy", "total_cost", "savings_vs_opus_pct", "gap_pct", "above_single_max", "acc_vs_random"} <= set(row)
    assert row["overhead_cost"] == 0.0


def test_jev_specs_are_charged_the_mean_fresh_jev_cost():
    orig_ctx, ox = make_context(80, 0.0, 3)
    fresh_ctx, fx = make_context(40, 0.0, 4)
    combined, fresh_idx = combine_contexts(orig_ctx, fresh_ctx)
    spec = build_specs(combined)["jev_plus_length"]
    rows = evaluate_frozen_spec(
        spec=spec, ctx=combined, dev_idx=np.arange(40), fresh_idx=fresh_idx,
        dev_outcomes=synthetic_outcomes(ox, "o")[:40], fresh_outcomes=synthetic_outcomes(fx, "f"),
        fresh_jev_costs=np.full(40, 0.00004), floors=(0.97,), families=["mmlu"] * 40,
    )
    row = next(r for r in rows if r["group"] == "combined")
    assert row["overhead_cost"] == pytest.approx(40 * 0.00004)
