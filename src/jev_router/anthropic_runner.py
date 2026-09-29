"""Runs benchmark items through the Anthropic tiers with caching and a hard budget cap."""

import hashlib
import json
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from jev_router.answers import extract_letter
from jev_router.budget import BudgetExceeded, BudgetTracker, cost_usd
from jev_router.sampling import BenchItem

MODELS = ("claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5")
MAX_TOKENS = 4096
# Haiku 4.5 rejects `effort`; Sonnet/Opus 5.5 default to adaptive thinking, so pin it low.
EFFORT_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}


class Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class Client(Protocol):
    messages: Messages


@dataclass(frozen=True)
class RunRecord:
    item_id: str
    model: str
    correct: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str
    latency_seconds: float


class RunError(RuntimeError):
    def __init__(self, failures: Sequence[tuple[str, str]]) -> None:
        self.failures = tuple(failures)
        first = self.failures[0][2] if self.failures else ""
        super().__init__(
            f"{len(self.failures)} call(s) failed; successes are cached. First: {first}"
        )


def request_kwargs(model: str, prompt: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model in EFFORT_MODELS:
        kwargs["output_config"] = {"effort": "low"}
    return kwargs


def grade(item: BenchItem, text: str) -> bool:
    valid = "".join(re.findall(r"^([A-Z])\)", item.prompt, re.M)) or "ABCD"
    return extract_letter(text, valid) == item.truth


def _key(item: BenchItem, model: str) -> str:
    params = request_kwargs(model, item.prompt)
    blob = json.dumps(params, sort_keys=True)
    return hashlib.sha256(f"{item.id}|{blob}".encode()).hexdigest()[:32]


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
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


def _to_record(row: dict[str, Any]) -> RunRecord:
    return RunRecord(**{k: row[k] for k in RunRecord.__dataclass_fields__})


def _record_for(item: BenchItem, row: dict[str, Any]) -> RunRecord:
    """Rebuilds the graded record, re-grading from the saved response text when present."""
    record = _to_record(row)
    if "text" in row:
        return replace(record, correct=row["stop_reason"] != "refusal" and grade(item, row["text"]))
    return record


def ledger_spent(cache_path: Path) -> float:
    """Total dollars ever billed: every line in the cache, including repeated keys."""
    total = 0.0
    if not cache_path.exists():
        return total
    for line in cache_path.read_text().splitlines():
        try:
            total += float(json.loads(line).get("cost_usd", 0.0))
        except (ValueError, TypeError, AttributeError):
            continue
    return total


def make_tracker(cap_usd: float, cache_path: Path) -> BudgetTracker:
    """A tracker whose cap is cumulative across runs, seeded from what the cache says was spent."""
    return BudgetTracker(cap_usd=cap_usd, spent=ledger_spent(cache_path))


def run(
    items: Sequence[BenchItem],
    models: Sequence[str],
    client: Client,
    cache_path: Path,
    tracker: BudgetTracker,
    workers: int = 8,
) -> list[RunRecord]:
    """Runs every missing (item, model) call. Each call's full response is cached on disk.

    Spending is capped hard: before a call starts, its worst-case cost (max output tokens)
    is reserved, and a call only starts if spent + in-flight reservations + its own worst
    case fit under the cap.
    """
    cache = _load_cache(cache_path)
    tasks = [
        (item, model)
        for item in items
        for model in models
        if "text" not in cache.get(_key(item, model), {})
    ]
    lock = threading.Lock()
    over_budget = threading.Event()
    in_flight = [0.0]

    def admit(worst: float) -> bool:
        while True:
            with lock:
                if over_budget.is_set() or tracker.spent + worst > tracker.cap_usd:
                    over_budget.set()
                    return False
                if tracker.spent + in_flight[0] + worst <= tracker.cap_usd:
                    in_flight[0] += worst
                    return True
            time.sleep(0.02)

    def call(item: BenchItem, model: str) -> dict[str, Any] | None:
        worst = cost_usd(model, len(item.prompt) // 2 + 50, MAX_TOKENS)
        if not admit(worst):
            return None
        params = request_kwargs(model, item.prompt)
        started = time.perf_counter()
        try:
            response = client.messages.create(**params)
        except BaseException:
            with lock:
                in_flight[0] -= worst
            raise
        latency = time.perf_counter() - started
        usage = response.usage
        with lock:
            in_flight[0] -= worst
            try:
                tracker.add(model, usage.input_tokens, usage.output_tokens)
            except BudgetExceeded:
                over_budget.set()
        blocks = response.content
        text = "".join(b.text for b in blocks if b.type == "text")
        refused = response.stop_reason == "refusal"
        return {
            "key": _key(item, model),
            "item_id": item.id,
            "model": model,
            "correct": bool(not refused and grade(item, text)),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(model, usage.input_tokens, usage.output_tokens),
            "stop_reason": response.stop_reason,
            "latency_seconds": latency,
            "text": text,
            "thinking": "".join(getattr(b, "thinking", "") for b in blocks if b.type == "thinking"),
            "content_types": [b.type for b in blocks],
            "model_returned": getattr(response, "model", None),
            "request": {k: v for k, v in params.items() if k != "messages"},
        }

    failures: list[tuple[str, str]] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as out:
        futures = {pool.submit(call, item, model): (item.id, model) for item, model in tasks}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:  # per-call failure must not lose paid results
                failures.append((*futures[future], f"{type(exc).__name__}: {str(exc)[:200]}"))
                continue
            if row is None:
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[row["key"]] = row
    if over_budget.is_set():
        raise BudgetExceeded(
            f"stopped at ${tracker.spent:.2f} of the ${tracker.cap_usd:.2f} cap; "
            "completed calls are cached"
        )
    if failures:
        raise RunError(failures)
    return [_record_for(i, cache[_key(i, m)]) for i in items for m in models]


def load_cached_records(
    items: Sequence[BenchItem], models: Sequence[str], cache_path: Path
) -> tuple[list[BenchItem], list[RunRecord]]:
    """Items with a cached result for every model, and those results (no API calls)."""
    cache = _load_cache(cache_path)
    found = [i for i in items if all(_key(i, m) in cache for m in models)]
    return found, [_record_for(i, cache[_key(i, m)]) for i in found for m in models]
