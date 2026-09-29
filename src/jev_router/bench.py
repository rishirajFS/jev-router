"""Runs the RouterBench sample through the Anthropic tiers (pilot or full)."""

import argparse
import json
import random
import sys
from pathlib import Path

import anthropic

from jev_router.anthropic_runner import MODELS, run
from jev_router.anthropic_runner import make_tracker
from jev_router.budget import BudgetExceeded
from jev_router.config import load_api_key
from jev_router.report import summarize_runs
from jev_router.sampling import BenchItem

ROOT = Path(__file__).resolve().parents[2]


def load_items(path: Path) -> list[BenchItem]:
    return [BenchItem(**json.loads(line)) for line in path.read_text().splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="items to run (pilot size)")
    parser.add_argument("--cap", type=float, default=12.0, help="CUMULATIVE hard spend cap in USD")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    all_items = load_items(ROOT / "data" / "routerbench_sample.jsonl")
    items = random.Random(args.seed).sample(all_items, min(args.n, len(all_items)))

    workspace_id = load_api_key(ROOT / ".env", name="ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        api_key=load_api_key(ROOT / ".env", name="ANTHROPIC_API_KEY"),
        max_retries=4,
        default_headers={"anthropic-workspace-id": workspace_id},
    )
    cache = ROOT / "data" / "cache" / "anthropic.jsonl"
    tracker = make_tracker(args.cap, cache)
    try:
        records = run(items, MODELS, client, cache, tracker)
    except BudgetExceeded as exc:
        print(f"STOPPED: {exc}")
        return 2
    report = summarize_runs(records, full_size=len(all_items))
    report["spent_this_run"] = tracker.spent
    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"anthropic_n{len(items)}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
