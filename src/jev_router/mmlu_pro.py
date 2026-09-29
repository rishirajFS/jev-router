"""MMLU-Pro (TIGER-Lab, MIT license): harder 10-option questions with exact answers."""

import random

import pandas as pd

from jev_router.sampling import BenchItem

LETTERS = "ABCDEFGHIJ"


def format_prompt(question: str, options: list[str]) -> str:
    letters = LETTERS[: len(options)]
    lines = "\n".join(f"{letter}) {option}" for letter, option in zip(letters, options))
    choices = " or ".join(f'"{letter}"' for letter in letters)
    return (
        f"{question}\n{lines}\n"
        f"Print only a single choice from {choices} without explanation. Answer:"
    )


def build_mmlu_pro_sample(df: pd.DataFrame, n: int, seed: int) -> list[BenchItem]:
    """Equal-sized draws per subject category so no single subject dominates."""
    rng = random.Random(seed)
    categories = sorted(df["category"].unique())
    base, extra = divmod(n, len(categories))
    items: list[BenchItem] = []
    for k, category in enumerate(categories):
        rows = df[df["category"] == category]
        quota = min(base + (1 if k < extra else 0), len(rows))
        for index in rng.sample(list(rows.index), quota):
            row = rows.loc[index]
            items.append(
                BenchItem(
                    id=f"mmlu-pro.{row['question_id']}",
                    family="mmlu-pro",
                    prompt=format_prompt(row["question"], [str(o) for o in row["options"]]),
                    truth=str(row["answer"]),
                    old_pass_rate=None,
                )
            )
    return items
