import pytest

from jev_router.features import RoutingFeatures, parse_features
from jev_router.questions import DIFFICULTY_LEVELS, TASK_TYPES, build_questions


def test_build_questions_has_expected_typed_questions():
    questions = build_questions()
    assert questions["task_type"]["type"] == "choice"
    assert set(questions["task_type"]["criteria"]) == set(TASK_TYPES)
    assert questions["difficulty"]["type"] == "score"
    assert len(questions["difficulty"]["criteria"]) == len(DIFFICULTY_LEVELS)
    assert questions["multistep"]["type"] == "noul"


def test_build_questions_returns_fresh_copies():
    first = build_questions()
    first["task_type"]["criteria"]["math"] = "mutated"
    assert build_questions()["task_type"]["criteria"]["math"] != "mutated"


def test_parse_features_maps_answers():
    answers = {
        "task_type": {"choice": "math", "confidence": 0.9},
        "difficulty": {"score": 2.1, "confidence": 0.8},
        "multistep": {"noul": 0.7},
    }
    assert parse_features(answers) == RoutingFeatures(
        task_type="math",
        task_type_confidence=0.9,
        difficulty=2.1,
        difficulty_confidence=0.8,
        multistep=0.7,
    )


def test_parse_features_raises_on_missing_answer():
    with pytest.raises(ValueError, match="difficulty"):
        parse_features({"task_type": {"choice": "math", "confidence": 1.0}})


def test_parse_features_wraps_malformed_sub_fields():
    answers = {
        "task_type": {"choice": "math"},
        "difficulty": {"score": 1.0, "confidence": 0.5},
        "multistep": {"noul": 0.1},
    }
    with pytest.raises(ValueError, match="task_type"):
        parse_features(answers)
