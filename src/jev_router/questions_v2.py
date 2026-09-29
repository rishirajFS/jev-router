"""Richer typed Jev questions and the feature vector built from their raw answers."""

import copy
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from jev_router.questions import DIFFICULTY_LEVELS, TASK_TYPES, build_questions

_TIER_NOULS: dict[str, dict[str, Any]] = {
    "small_ok": {
        "type": "noul",
        "instructions": (
            "Would a small, fast, inexpensive language model answer this request "
            "correctly on the first try?"
        ),
        "criteria": {
            "true": "A small model would very likely get it right",
            "false": "A small model would likely get it wrong or miss key details",
        },
    },
    "mid_ok": {
        "type": "noul",
        "instructions": (
            "Would a mid-sized, general-purpose language model answer this request "
            "correctly on the first try?"
        ),
        "criteria": {
            "true": "A mid-sized model would very likely get it right",
            "false": "A mid-sized model would likely get it wrong or miss key details",
        },
    },
    "expert_needed": {
        "type": "noul",
        "instructions": (
            "Does answering this correctly need expert-level domain knowledge or "
            "careful multi-step calculation that only the strongest models do reliably?"
        ),
    },
}

DIFFICULTY_KEYS = tuple(str(i) for i in range(len(DIFFICULTY_LEVELS)))
TASK_KEYS = tuple(TASK_TYPES)

FEATURE_NAMES: tuple[str, ...] = (
    *(f"d_p{k}" for k in DIFFICULTY_KEYS),
    "d_score",
    "d_conf",
    "multistep",
    "small_ok",
    "mid_ok",
    "expert_needed",
    "log_chars",
    *(f"t_{name}" for name in TASK_KEYS),
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "difficulty_only": (
        *(f"d_p{k}" for k in DIFFICULTY_KEYS),
        "d_score",
        "d_conf",
        "multistep",
    ),
    "full": FEATURE_NAMES,
}


def build_questions_v2() -> dict[str, dict[str, Any]]:
    questions = build_questions()
    questions.update(copy.deepcopy(_TIER_NOULS))
    return questions


def _answer(answers: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in answers:
        raise ValueError(f"Jev response is missing the '{name}' answer")
    return answers[name]


def extract_features(answers: dict[str, Any], prompt_chars: int) -> dict[str, float]:
    difficulty = _answer(answers, "difficulty")
    task = _answer(answers, "task_type")
    probs = difficulty.get("probabilities", {})
    task_probs = task.get("probabilities", {})
    features: dict[str, float] = {f"d_p{k}": float(probs.get(k, 0.0)) for k in DIFFICULTY_KEYS}
    features["d_score"] = float(difficulty["score"])
    features["d_conf"] = float(difficulty["confidence"])
    for name in ("multistep", "small_ok", "mid_ok", "expert_needed"):
        features[name] = float(_answer(answers, name)["noul"])
    features["log_chars"] = math.log1p(prompt_chars)
    for name in TASK_KEYS:
        features[f"t_{name}"] = float(task_probs.get(name, 0.0))
    return features


def select_columns(matrix: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return matrix[:, [FEATURE_NAMES.index(n) for n in names]]
