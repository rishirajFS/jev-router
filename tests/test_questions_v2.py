import math

import numpy as np
import pytest

from jev_router.questions_v2 import (
    FEATURE_NAMES,
    FEATURE_SETS,
    build_questions_v2,
    extract_features,
    select_columns,
)


def raw_answers():
    return {
        "task_type": {"type": "choice", "choice": "math", "confidence": 0.9,
                      "probabilities": {"math": 0.9, "code": 0.1}},
        "difficulty": {"type": "score", "score": 1.5, "confidence": 0.6,
                       "probabilities": {"0": 0.1, "1": 0.3, "2": 0.5, "3": 0.1}},
        "multistep": {"type": "noul", "noul": 0.7},
        "small_ok": {"type": "noul", "noul": 0.2},
        "mid_ok": {"type": "noul", "noul": 0.6},
        "expert_needed": {"type": "noul", "noul": 0.4},
    }


def test_build_questions_v2_keeps_originals_and_adds_three_noul_questions():
    questions = build_questions_v2()
    assert {"task_type", "difficulty", "multistep"} <= set(questions)
    for name in ("small_ok", "mid_ok", "expert_needed"):
        assert questions[name]["type"] == "noul"
        assert len(questions[name]["instructions"]) > 20
    assert questions["difficulty"]["type"] == "score"


def test_build_questions_v2_returns_fresh_copies():
    first = build_questions_v2()
    first["small_ok"]["instructions"] = "mutated"
    assert build_questions_v2()["small_ok"]["instructions"] != "mutated"


def test_extract_features_reads_full_distributions_and_length():
    f = extract_features(raw_answers(), prompt_chars=99)
    assert set(f) == set(FEATURE_NAMES)
    assert f["d_p2"] == pytest.approx(0.5)
    assert f["d_score"] == pytest.approx(1.5)
    assert f["small_ok"] == pytest.approx(0.2)
    assert f["mid_ok"] == pytest.approx(0.6)
    assert f["expert_needed"] == pytest.approx(0.4)
    assert f["t_math"] == pytest.approx(0.9)
    assert f["t_chat"] == 0.0  # absent probabilities default to zero
    assert f["log_chars"] == pytest.approx(math.log1p(99))


def test_extract_features_raises_on_missing_question():
    answers = raw_answers()
    del answers["mid_ok"]
    with pytest.raises(ValueError, match="mid_ok"):
        extract_features(answers, prompt_chars=10)


def test_select_columns_picks_named_feature_subsets():
    matrix = np.arange(len(FEATURE_NAMES) * 2, dtype=float).reshape(2, -1)
    subset = select_columns(matrix, FEATURE_SETS["difficulty_only"])
    assert subset.shape == (2, len(FEATURE_SETS["difficulty_only"]))
    first = FEATURE_NAMES.index(FEATURE_SETS["difficulty_only"][0])
    assert subset[0, 0] == matrix[0, first]
    assert set(FEATURE_SETS["difficulty_only"]) < set(FEATURE_SETS["full"])
