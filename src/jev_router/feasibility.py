"""Feasibility experiment: do Jev's routing features track how hard prompts really are?"""

import hashlib
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from jev_router.config import load_api_key
from jev_router.dataset import LabeledPrompt, load_prompts
from jev_router.evaluation import confusion_matrix, routing_report, spearman
from jev_router.features import RoutingFeatures, parse_features
from jev_router.jev_client import JevClient, JevResult
from jev_router.policy import Policy, route
from jev_router.questions import build_questions

ROOT = Path(__file__).resolve().parents[2]


class Analyzer(Protocol):
    def analyze(self, state: str, questions: dict[str, Any]) -> JevResult: ...


@dataclass(frozen=True)
class Record:
    prompt_id: str
    features: RoutingFeatures
    latency_seconds: float
    tokens: int


def _cache_key(prompt: LabeledPrompt, questions: dict[str, Any], model: str) -> str:
    blob = json.dumps([model, prompt.text, questions], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Reads the JSONL cache, skipping lines truncated by an interrupted write."""
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            cache[row["key"]] = row
        except (ValueError, KeyError, TypeError):
            continue
    return cache


def _to_record(prompt_id: str, row: dict[str, Any]) -> Record:
    return Record(
        prompt_id=prompt_id,
        features=RoutingFeatures(**row["features"]),
        latency_seconds=row["latency_seconds"],
        tokens=row["tokens"],
    )


def load_cached_features(
    prompts: Sequence[LabeledPrompt], cache_path: Path, model: str = "jev-latest"
) -> dict[str, RoutingFeatures]:
    """Cached Jev features by prompt id; prompts not yet cached are omitted."""
    questions = build_questions()
    cache = _load_cache(cache_path)
    return {
        p.id: RoutingFeatures(**cache[key]["features"])
        for p in prompts
        if (key := _cache_key(p, questions, model)) in cache
    }


def load_cached_jev_tokens(
    prompts: Sequence[LabeledPrompt], cache_path: Path, model: str = "jev-latest"
) -> dict[str, int]:
    """Billable (input) Jev tokens per cached prompt. Older rows only stored input+output
    combined, which slightly overstates the bill since output is free."""
    questions = build_questions()
    cache = _load_cache(cache_path)
    tokens: dict[str, int] = {}
    for p in prompts:
        row = cache.get(_cache_key(p, questions, model))
        if row is not None:
            tokens[p.id] = row.get("usage", {}).get("input_tokens", row["tokens"])
    return tokens


class CollectError(RuntimeError):
    """Some prompts failed; every successful result is already saved to the cache."""

    def __init__(self, failed_ids: Sequence[str]) -> None:
        self.failed_ids = tuple(failed_ids)
        super().__init__(f"{len(self.failed_ids)} prompt(s) failed: {self.failed_ids}")


def collect(
    prompts: Sequence[LabeledPrompt],
    client: Analyzer,
    cache_path: Path,
    workers: int = 4,
) -> list[Record]:
    questions = build_questions()
    model = getattr(client, "model", "unknown")
    keys = {p.id: _cache_key(p, questions, model) for p in prompts}
    cache = _load_cache(cache_path)

    unique: dict[str, LabeledPrompt] = {}
    for prompt in prompts:
        if keys[prompt.id] not in cache:
            unique.setdefault(keys[prompt.id], prompt)

    def fetch(item: tuple[str, LabeledPrompt]) -> tuple[str, dict[str, Any] | None]:
        key, prompt = item
        try:
            result = client.analyze(prompt.text, questions)
            return key, {
                "key": key,
                "features": asdict(parse_features(result.answers)),
                "latency_seconds": result.latency_seconds,
                "tokens": sum(result.usage.values()),
                "answers": result.answers,
                "usage": result.usage,
                "model_returned": result.model,
            }
        except Exception:
            return key, None

    failed_keys: set[str] = set()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as out:
        for key, row in pool.map(fetch, unique.items()):
            if row is None:
                failed_keys.add(key)
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[key] = row
    if failed_keys:
        raise CollectError([p.id for p in prompts if keys[p.id] in failed_keys])
    return [_to_record(p.id, cache[keys[p.id]]) for p in prompts]


def summarize(
    prompts: Sequence[LabeledPrompt], records: Sequence[Record], policy: Policy
) -> dict[str, Any]:
    if len(prompts) != len(records):
        raise ValueError("prompts and records must have the same length")
    if not prompts:
        raise ValueError("summarize needs at least one prompt")
    expected = [p.expected_tier for p in prompts]
    decisions = [route(r.features, policy) for r in records]
    predicted = [int(d.tier) for d in decisions]
    difficulties = [r.features.difficulty for r in records]
    task_hits = [
        r.features.task_type == p.task_type for p, r in zip(prompts, records, strict=True)
    ]
    n = len(records)
    return {
        "routing": routing_report(expected, predicted),
        "confusion_matrix": confusion_matrix(expected, predicted),
        "difficulty_spearman": spearman(difficulties, expected),
        "task_type_accuracy": sum(task_hits) / n,
        "mean_difficulty_by_expected_tier": {
            t: sum(d for d, e in zip(difficulties, expected) if e == t)
            / max(1, expected.count(t))
            for t in sorted(set(expected))
        },
        "escalations": {
            "low_confidence": sum(d.escalated_for_low_confidence for d in decisions),
            "multistep": sum(d.escalated_for_multistep for d in decisions),
        },
        "mean_latency_seconds": sum(r.latency_seconds for r in records) / n,
        "total_tokens": sum(r.tokens for r in records),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    prompts = load_prompts(ROOT / "data" / "probe_prompts.jsonl")
    client = JevClient(load_api_key(ROOT / ".env"))
    try:
        records = collect(prompts, client, ROOT / "data" / "cache" / "jev.jsonl")
    finally:
        client.close()
    report = summarize(prompts, records, Policy())
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feasibility.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
