"""Loading of the hand-labeled probe prompts."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledPrompt:
    id: str
    text: str
    task_type: str
    expected_tier: int


def load_prompts(path: Path | str) -> tuple[LabeledPrompt, ...]:
    prompts = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            prompts.append(
                LabeledPrompt(
                    id=row["id"],
                    text=row["text"],
                    task_type=row["task_type"],
                    expected_tier=row["expected_tier"],
                )
            )
    return tuple(prompts)
