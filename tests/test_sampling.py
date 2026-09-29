import pandas as pd
import pytest

from jev_router.sampling import build_sample, family_of

MODELS = ["m1", "m2", "m3"]


def make_frame(n_per_family=30):
    rows = []
    for family, prefix in (("mmlu", "mmlu-law"), ("arc-challenge", "arc-challenge")):
        for i in range(n_per_family):
            passing = i % 4  # 0..3 old models correct
            row = {
                "sample_id": f"{prefix}.test.{i}",
                "eval_name": prefix,
                "prompt": "['Q?\\nA) x\\nB) y\\nC) z\\nD) w\\nPrint only a single choice']",
                "oracle_model_to_route_to": "m1",
            }
            for j, m in enumerate(MODELS):
                right = j < passing
                row[m] = 1.0 if right else 0.0
                row[f"{m}|model_response"] = "['B']" if right else "['C']"
                row[f"{m}|total_cost"] = 0.001
            rows.append(row)
    return pd.DataFrame(rows)


def test_family_of_maps_eval_names():
    assert family_of("mmlu-anatomy") == "mmlu"
    assert family_of("grade-school-math") == "grade-school-math"
    assert family_of("hellaswag") == "hellaswag"
    assert family_of("chinese_idioms") == "other"


def test_build_sample_is_deterministic_and_respects_plan():
    df = make_frame()
    plan = {"mmlu": 8, "arc-challenge": 4}
    first = build_sample(df, MODELS, plan, seed=7, min_votes=1)
    second = build_sample(df, MODELS, plan, seed=7, min_votes=1)
    assert [i.id for i in first] == [i.id for i in second]
    assert sum(i.family == "mmlu" for i in first) == 8
    assert sum(i.family == "arc-challenge" for i in first) == 4


def test_build_sample_recovers_ground_truth_and_drops_unanswerable():
    df = make_frame()
    items = build_sample(df, MODELS, {"mmlu": 30}, seed=1, min_votes=1)
    assert all(item.truth == "B" for item in items)
    assert all(item.old_pass_rate > 0 for item in items)  # zero-pass rows have no truth


def test_build_sample_spans_difficulty_strata():
    df = make_frame(60)
    items = build_sample(df, MODELS, {"mmlu": 30}, seed=3, min_votes=1)
    rates = {round(i.old_pass_rate, 2) for i in items}
    assert len(rates) >= 2


def test_build_sample_rejects_unknown_family():
    with pytest.raises(ValueError):
        build_sample(make_frame(), MODELS, {"nope": 1}, seed=1)
