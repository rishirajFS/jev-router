"""Summaries of Anthropic benchmark runs."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from jev_router.anthropic_runner import RunRecord


def summarize_runs(records: Sequence[RunRecord], full_size: int) -> dict[str, Any]:
    """Per-model accuracy, tokens, and cost, plus a projection to `full_size` items."""
    if not records:
        raise ValueError("summarize_runs needs at least one record")
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.model].append(record)

    models: dict[str, dict[str, Any]] = {}
    for model, rows in grouped.items():
        n = len(rows)
        models[model] = {
            "n": n,
            "accuracy": sum(r.correct for r in rows) / n,
            "mean_input_tokens": sum(r.input_tokens for r in rows) / n,
            "mean_output_tokens": sum(r.output_tokens for r in rows) / n,
            "mean_latency_seconds": sum(r.latency_seconds for r in rows) / n,
            "cost": sum(r.cost_usd for r in rows),
            "truncated": sum(r.stop_reason == "max_tokens" for r in rows),
            "refusals": sum(r.stop_reason == "refusal" for r in rows),
        }
    pilot_cost = sum(r.cost_usd for r in records)
    items = len({r.item_id for r in records})
    return {
        "models": models,
        "pilot_cost": pilot_cost,
        "projected_full_cost": pilot_cost / items * full_size,
    }
