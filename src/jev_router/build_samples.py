"""Rebuilds the exact benchmark samples from the public datasets (deterministic, seed 42).

Only item IDs are published with the repository; the question text stays in the source
datasets (RouterBench, MMLU-Pro) and is regenerated locally by this script.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from jev_router.mmlu_pro import build_mmlu_pro_sample
from jev_router.sampling import BenchItem, build_sample

ROOT = Path(__file__).resolve().parents[2]
ROUTERBENCH_PLAN = {"mmlu": 500, "arc-challenge": 100, "hellaswag": 100, "winogrande": 100}
MMLU_PRO_N = 420
SEED = 42


def ids_of(items: Sequence[BenchItem]) -> list[str]:
    return sorted(item.id for item in items)


def write_ids(items: Sequence[BenchItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids_of(items)))


def write_items(items: Sequence[BenchItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item.__dict__) + "\n" for item in items))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-dir", type=Path, default=ROOT / "samples")
    args = parser.parse_args(argv)

    frame = pd.read_pickle(ROOT / "data" / "raw" / "routerbench_0shot.pkl")
    models = [c for c in frame.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to")]
    routerbench = build_sample(frame, models, ROUTERBENCH_PLAN, seed=SEED)
    mmlu_pro = build_mmlu_pro_sample(pd.read_parquet(ROOT / "data" / "raw" / "mmlu_pro_test.parquet"), MMLU_PRO_N, SEED)

    write_items(routerbench, ROOT / "data" / "routerbench_sample.jsonl")
    write_items(mmlu_pro, ROOT / "data" / "mmlu_pro_sample.jsonl")
    write_ids(routerbench, args.ids_dir / "routerbench_ids.txt")
    write_ids(mmlu_pro, args.ids_dir / "mmlu_pro_ids.txt")
    print(f"routerbench={len(routerbench)} mmlu-pro={len(mmlu_pro)} ids written to {args.ids_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
