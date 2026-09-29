"""Typed routing features parsed from a Jev response."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingFeatures:
    task_type: str
    task_type_confidence: float
    difficulty: float
    difficulty_confidence: float
    multistep: float


def _require(answers: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in answers:
        raise ValueError(f"Jev response is missing the '{name}' answer")
    return answers[name]


def _field(answer: dict[str, Any], name: str, field: str) -> Any:
    try:
        return answer[field]
    except (KeyError, TypeError):
        raise ValueError(f"Jev '{name}' answer is missing '{field}'") from None


def parse_features(answers: dict[str, Any]) -> RoutingFeatures:
    task = _require(answers, "task_type")
    difficulty = _require(answers, "difficulty")
    multistep = _require(answers, "multistep")
    return RoutingFeatures(
        task_type=_field(task, "task_type", "choice"),
        task_type_confidence=_field(task, "task_type", "confidence"),
        difficulty=_field(difficulty, "difficulty", "score"),
        difficulty_confidence=_field(difficulty, "difficulty", "confidence"),
        multistep=_field(multistep, "multistep", "noul"),
    )
