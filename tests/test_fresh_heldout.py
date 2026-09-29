import numpy as np
import pandas as pd
import pytest

from jev_router.fresh_heldout import (
    assert_disjoint,
    fit_frozen_router,
    fresh_mmlu_pro_sample,
    fresh_routerbench_sample,
    gap_pct,
    verdict,
)
from jev_router.sampling import BenchItem

MODELS3 = ["m1", "m2", "m3"]


def rb_frame(n=60):
    rows = []
    for i in range(n):
        passing = 1 + i % 3
        row = {"sample_id": f"mmlu-law.test.{i}", "eval_name": "mmlu-law",
               "prompt": "['Q?\\nA) x\\nB) y\\nC) z\\nD) w\\nPrint only a single choice']",
               "oracle_model_to_route_to": "m1"}
        for j, m in enumerate(MODELS3):
            right = j < passing
            row[m] = 1.0 if right else 0.0
            row[f"{m}|model_response"] = "['B']" if right else "['C']"
            row[f"{m}|total_cost"] = 0.001
        rows.append(row)
    return pd.DataFrame(rows)


def test_fresh_routerbench_sample_never_returns_excluded_ids():
    used = {f"mmlu-law.test.{i}" for i in range(0, 60, 2)}
    items = fresh_routerbench_sample(rb_frame(), MODELS3, {"mmlu": 12}, seed=5, exclude_ids=used, min_votes=1)
    assert len(items) == 12
    assert not {i.id for i in items} & used
    again = fresh_routerbench_sample(rb_frame(), MODELS3, {"mmlu": 12}, seed=5, exclude_ids=used, min_votes=1)
    assert [i.id for i in items] == [i.id for i in again]


def mp_frame():
    rows = []
    for cat in ("math", "law"):
        for i in range(30):
            rows.append({"question_id": f"{cat}{i}", "question": f"Q {cat} {i}?",
                         "options": ["a", "b", "c", "d"], "answer": "B", "category": cat})
    return pd.DataFrame(rows)


def test_fresh_mmlu_pro_sample_skips_used_questions():
    used = {f"mmlu-pro.math{i}" for i in range(20)} | {f"mmlu-pro.law{i}" for i in range(20)}
    items = fresh_mmlu_pro_sample(mp_frame(), n=10, seed=3, exclude_ids=used)
    assert len(items) == 10
    assert not {i.id for i in items} & used
    assert {i.id.split(".")[1].rstrip("0123456789") for i in items} == {"math", "law"}


def test_assert_disjoint_raises_on_any_overlap():
    fresh = [BenchItem("a", "mmlu", "Q", "A", 0.5), BenchItem("b", "mmlu", "Q", "A", 0.5)]
    assert_disjoint(fresh, {"x", "y"})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(fresh, {"b", "z"})


def test_fit_frozen_router_uses_only_the_arrays_it_is_given_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.random(n)
    X = np.column_stack([signal, rng.random(n)])
    Y = np.column_stack([signal < 0.5, signal < 0.8, np.ones(n)]).astype(float)
    C = np.column_stack([np.full(n, 0.001), np.full(n, 0.004), np.full(n, 0.010)])
    a = fit_frozen_router(X, Y, C, opus_dev_accuracy=1.0, floors=(0.99, 0.95), columns=(0, 1))
    b = fit_frozen_router(X, Y, C, opus_dev_accuracy=1.0, floors=(0.99, 0.95), columns=(0, 1))
    assert a.rules == b.rules and a.best_c == b.best_c
    assert set(a.rules) == {("threshold", 0.99), ("threshold", 0.95), ("loss", 0.99), ("loss", 0.95)}
    tiers = a.tiers_for(X, ("threshold", 0.95))
    assert len(tiers) == n
    assert tiers[signal < 0.2].mean() < tiers[signal > 0.9].mean()  # easy prompts go to cheaper tiers


def test_gap_pct_is_positive_when_more_expensive_than_the_single_model_mix():
    points = [(0.8, 0.1), (0.9, 0.5), (0.95, 1.0)]
    assert gap_pct(0.6, 0.9, points) == pytest.approx(20.0)
    assert gap_pct(0.4, 0.9, points) == pytest.approx(-20.0)
    assert gap_pct(1.0, 0.99, points) is None  # accuracy above every single model


def test_verdict_follows_the_preregistered_two_of_three_rule():
    assert verdict({0.99: -3.0, 0.97: -1.0, 0.95: 4.0}) == "holds"
    assert verdict({0.99: -3.0, 0.97: 1.0, 0.95: 4.0}) == "does not hold"
    assert verdict({0.99: None, 0.97: -1.0, 0.95: -2.0}) == "holds"
