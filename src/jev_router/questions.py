"""Typed Jev questions that describe a prompt for routing purposes."""

import copy
from typing import Any

TASK_TYPES: dict[str, str] = {
    "math": "Solving or proving something mathematical, or a quantitative word problem",
    "code": "Writing, debugging, explaining, or reviewing source code",
    "extraction": "Pulling specific fields or facts out of provided text, or reformatting it",
    "classification": "Assigning a label or category to provided text",
    "summarization": "Condensing or rewriting provided text",
    "factual_qa": "Answering a factual question from general knowledge",
    "writing": "Open-ended creative or professional writing",
    "reasoning": "Multi-step analysis, planning, or a nuanced tradeoff judgement",
    "chat": "Casual conversation or a trivial request",
}

DIFFICULTY_LEVELS: list[str] = [
    "Trivial: greeting, lookup, or one obvious step; any small model answers it",
    "Easy: a single well-defined step with a clear expected answer",
    "Hard: several dependent steps, careful reasoning, or non-trivial code or math",
    "Expert: deep domain expertise, long chains of reasoning, or subtle correctness matters",
]

_QUESTIONS: dict[str, dict[str, Any]] = {
    "task_type": {
        "type": "choice",
        "instructions": "What kind of task is the user asking an AI assistant to do?",
        "criteria": TASK_TYPES,
    },
    "difficulty": {
        "type": "score",
        "instructions": (
            "How difficult is this request for a language model to answer "
            "correctly? Judge the reasoning and expertise required, not the length."
        ),
        "criteria": DIFFICULTY_LEVELS,
    },
    "multistep": {
        "type": "noul",
        "instructions": (
            "Does answering this correctly require chaining several dependent "
            "reasoning steps, where an early mistake would break the final answer?"
        ),
    },
}


def build_questions() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(_QUESTIONS)
