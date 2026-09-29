"""Builds a difficulty-stratified, ground-truthed sample from RouterBench."""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from jev_router.answers import (
    consensus,
    extract_letter,
    extract_number,
    parse_listified,
)

MCQ_FAMILIES = ("mmlu", "arc-challenge", "hellaswag", "winogrande")
NUMERIC_FAMILIES = ("grade-school-math",)
SUPPORTED = MCQ_FAMILIES + NUMERIC_FAMILIES


@dataclass(frozen=True)
class BenchItem:
    id: str
    family: str
    prompt: str
    truth: str
    old_pass_rate: float | None  # None for datasets without prior model results


def family_of(eval_name: str) -> str:
    if eval_name.startswith("mmlu-"):
        return "mmlu"
    if eval_name in SUPPORTED or eval_name == "mbpp":
        return eval_name
    return "other"


def _truth(row: pd.Series, models: Sequence[str], family: str, prompt: str, min_votes: int):
    correct = [m for m in models if row[m] == 1.0]
    if family in NUMERIC_FAMILIES:
        answers = [extract_number(parse_listified(row[f"{m}|model_response"])) for m in correct]
    else:
        valid = "".join(re.findall(r"^([A-Z])\)", prompt, re.M)) or "ABCD"
        answers = [
            extract_letter(parse_listified(row[f"{m}|model_response"]), valid)
            for m in correct
        ]
    return consensus(answers, min_votes=min_votes)


def _stratified_pick(items: list[BenchItem], n: int, rng: random.Random) -> list[BenchItem]:
    """Takes n items spread evenly over old-model pass-rate terciles."""
    ordered = sorted(items, key=lambda i: (i.old_pass_rate, i.id))
    third = max(1, len(ordered) // 3)
    strata = [ordered[:third], ordered[third : 2 * third], ordered[2 * third :]]
    picked: list[BenchItem] = []
    quotas = [n // 3 + (1 if k < n % 3 else 0) for k in range(3)]
    for stratum, quota in zip(strata, quotas):
        picked.extend(rng.sample(stratum, min(quota, len(stratum))))
    if len(picked) < n:
        chosen = {i.id for i in picked}
        rest = [i for i in ordered if i.id not in chosen]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    return picked


def build_sample(
    df: pd.DataFrame,
    models: Sequence[str],
    plan: dict[str, int],
    seed: int,
    min_votes: int = 2,
) -> list[BenchItem]:
    unknown = set(plan) - set(SUPPORTED)
    if unknown:
        raise ValueError(f"unsupported families: {sorted(unknown)}")
    rng = random.Random(seed)
    frame = df.assign(_family=df["eval_name"].map(family_of))
    sample: list[BenchItem] = []
    for family in SUPPORTED:
        if family not in plan:
            continue
        candidates: list[BenchItem] = []
        for _, row in frame[frame["_family"] == family].iterrows():
            prompt = parse_listified(row["prompt"])
            truth = _truth(row, models, family, prompt, min_votes)
            if truth is None:
                continue
            candidates.append(
                BenchItem(
                    id=row["sample_id"],
                    family=family,
                    prompt=prompt,
                    truth=truth,
                    old_pass_rate=float(sum(row[m] for m in models) / len(models)),
                )
            )
        sample.extend(_stratified_pick(candidates, plan[family], rng))
    return sample
