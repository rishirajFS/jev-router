"""Learning curve: does Jev's advantage over TF-IDF / embedding routers shrink with more training data?

Each router is trained on the first n items of a seeded permutation of a split's DEV half (nested
subsets), with everything inside the pipeline (TF-IDF/SVD fit, regularization by out-of-fold
log-loss, thresholds from out-of-fold predictions, the quality floor measured on the subset) using
only that subset. It is then scored on the full, untouched TEST half of the same split.

The scoring is `ablation.evaluate_spec`, unchanged: split membership is a per-item hash, so
restricting the item list to (training subset + full test half) keeps every item on its side of the
split and lets the existing code run as is. At the full dev size this reproduces the ablation.
"""

import argparse
import json
import random
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from jev_router.ablation import (
    Context,
    _embed_with_fastembed,
    aggregate,
    build_specs,
    evaluate_spec,
)
from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.dataset import LabeledPrompt
from jev_router.questions_v2 import FEATURE_NAMES, build_questions_v2, extract_features
from jev_router.router_v2_experiment import JEV_V2_CACHE, load_cached_v2
from jev_router.routing_eval import ItemOutcome, build_outcomes, split_ids
from jev_router.run_experiment import ROOT, load_all_items

FULL = 10**6  # "use the whole dev half"
SIZES = (100, 200, 400, FULL)
SPECS = ("tfidf", "embedding", "jev_no_length", "jev_plus_length")
FLOORS = (0.97, 0.95)


def training_subset(dev_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """First n ids of a seeded permutation, so subsets are nested across sizes within a split."""
    order = sorted(dev_ids)
    random.Random(seed).shuffle(order)
    return order[:n]


def restrict(
    ctx: Context, outcomes: Sequence[ItemOutcome], jev_costs: np.ndarray, keep_ids: Collection[str]
) -> tuple[Context, list[ItemOutcome], np.ndarray]:
    """The same aligned data limited to `keep_ids`, in the original item order."""
    idx = [i for i, o in enumerate(outcomes) if o.item_id in keep_ids]
    embeddings = None if ctx.embeddings is None else ctx.embeddings[idx]
    restricted = Context(ctx.jev_matrix[idx], [ctx.texts[i] for i in idx], embeddings)
    return restricted, [outcomes[i] for i in idx], jev_costs[idx]


def curve_rows(
    ctx: Context,
    outcomes: Sequence[ItemOutcome],
    jev_costs: np.ndarray,
    seed: int,
    sizes: Sequence[int],
    spec_names: Sequence[str],
    floors: Sequence[float],
) -> list[dict[str, Any]]:
    dev_ids, test_ids = split_ids([o.item_id for o in outcomes], seed)
    rows: list[dict[str, Any]] = []
    for n in sizes:
        subset = training_subset(dev_ids, n, seed)
        ctx_r, out_r, costs_r = restrict(ctx, outcomes, jev_costs, set(subset) | set(test_ids))
        r_dev, r_test = split_ids([o.item_id for o in out_r], seed)
        if set(r_dev) != set(subset) or set(r_test) != set(test_ids):
            raise RuntimeError("restriction moved items across the dev/test split")
        specs = build_specs(ctx_r)
        for name in spec_names:
            if name in specs:
                for row in evaluate_spec(specs[name], ctx_r, out_r, costs_r, seed, floors):
                    rows.append({**row, "n_requested": n, "n_train": len(subset)})
    return rows


def aggregate_curve(runs: Sequence[Sequence[dict[str, Any]]]) -> dict[int, dict[str, dict[str, Any]]]:
    """Per training size: the ablation's per-(spec, rule, floor) statistics over splits."""
    sizes = sorted({r["n_requested"] for rows in runs for r in rows})
    return {
        n: aggregate([[r for r in rows if r["n_requested"] == n] for rows in runs])
        for n in sizes
    }


def reproduction_check(
    curve_cells: Mapping[str, Mapping[str, float]], ablation_cells: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """Largest absolute difference between the full-size curve row and the recorded ablation."""
    shared = sorted(set(curve_cells) & set(ablation_cells))
    diffs = {"gap": [], "share": [], "accuracy": []}
    for key in shared:
        a, b = curve_cells[key], ablation_cells[key]
        diffs["gap"].append(abs(a["gap_mean"] - b["gap_mean"]))
        diffs["share"].append(abs(a["share_cheaper_than_frontier"] - b["share_cheaper_than_frontier"]))
        diffs["accuracy"].append(abs(a["accuracy_mean"] - b["accuracy_mean"]))
    maxima = {k: (max(v) if v else 0.0) for k, v in diffs.items()}
    return {
        "cells_compared": len(shared),
        "max_gap_diff": maxima["gap"],
        "max_share_diff": maxima["share"],
        "max_accuracy_diff": maxima["accuracy"],
        "max_abs_diff": max(maxima.values()),
    }


def load_embeddings_readonly(
    ids: Sequence[str],
    texts: Sequence[str],
    cache_dir: Path,
    embed: Callable[[Sequence[str]], np.ndarray],
) -> np.ndarray:
    """Cached embeddings selected by id, never written. Falls back to embedding in memory.

    Another process may be rewriting the shared cache, so this only reads it and treats anything
    unreadable, partial or missing an id as a miss.
    """
    try:
        matrix = np.load(cache_dir / "embeddings.npy")
        cached_ids = json.loads((cache_dir / "embeddings_meta.json").read_text())["ids"]
        position = {item_id: k for k, item_id in enumerate(cached_ids)}
        if matrix.shape[0] == len(cached_ids) and all(i in position for i in ids):
            return matrix[[position[i] for i in ids]]
    except Exception:  # unreadable, partial or absent cache: recompute in memory instead
        pass
    return np.asarray(embed(list(texts)), dtype=float)


def load_context_readonly() -> tuple[Context, list[ItemOutcome], np.ndarray]:
    items = load_all_items(("routerbench", "mmlu-pro"), 420, 42)
    ready, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    rows = load_cached_v2(prompts, JEV_V2_CACHE, build_questions_v2())
    X = np.array([[extract_features(rows[p.id]["answers"], len(p.text))[n] for n in FEATURE_NAMES] for p in prompts])
    jev_costs = np.array([jev_cost_usd(rows[p.id]["usage"]["input_tokens"]) for p in prompts])
    texts = [p.text for p in prompts]
    embeddings = load_embeddings_readonly([p.id for p in prompts], texts, ROOT / "data" / "cache", _embed_with_fastembed)
    return Context(X, texts, embeddings), build_outcomes(ready, records), jev_costs


def _run_seed(args: tuple) -> list[dict[str, Any]]:
    ctx, outcomes, jev_costs, seed, sizes, spec_names, floors = args
    return curve_rows(ctx, outcomes, jev_costs, seed, sizes, spec_names, floors)


def _label(n: int) -> str:
    return "full" if n >= FULL else str(n)


def _print_tables(curve: Mapping[int, Mapping[str, Mapping[str, Any]]], floors: Sequence[float]) -> None:
    sizes = sorted(curve)
    for rule in ("loss", "threshold"):
        for floor in floors:
            print(f"\n-- {rule} rule, floor {floor}: mean gap to frontier % (share of splits cheaper)")
            print(f"{'feature set':18s}" + "".join(f"{_label(n):>16s}" for n in sizes))
            for spec in SPECS:
                cells = []
                for n in sizes:
                    v = curve[n].get(f"{spec}:{rule}:{floor}")
                    cells.append("n/a" if v is None else f"{v['gap_mean']:+6.1f} ({100 * v['share_cheaper_than_frontier']:3.0f}%)")
                print(f"{spec:18s}" + "".join(f"{c:>16s}" for c in cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--floors", default="0.97,0.95")
    parser.add_argument("--sizes", default="100,200,400")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    floors = tuple(float(f) for f in args.floors.split(","))
    sizes = tuple(int(s) for s in args.sizes.split(",")) + (FULL,)

    ctx, outcomes, jev_costs = load_context_readonly()
    specs = SPECS if ctx.embeddings is not None else tuple(s for s in SPECS if s != "embedding")
    jobs = [(ctx, outcomes, jev_costs, seed, sizes, specs, floors) for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(_run_seed, jobs))
    curve = aggregate_curve(runs)

    recorded = json.loads((ROOT / "data" / "results" / "ablation.json").read_text())["multi_seed"]
    check = reproduction_check(curve[FULL], recorded)
    n_train = {_label(n): float(np.mean([r["n_train"] for rows in runs for r in rows if r["n_requested"] == n])) for n in sizes}
    out = ROOT / "data" / "results" / "learning_curve.json"
    out.write_text(json.dumps({
        "sizes": [_label(n) for n in sizes], "mean_n_train": n_train, "seeds": args.seeds, "floors": list(floors),
        "n_items": len(outcomes), "reproduction_vs_ablation": check,
        "curve": {_label(n): curve[n] for n in sizes}, "per_seed": runs,
    }, indent=2))

    print(f"items {len(outcomes)}, splits {args.seeds}, mean training sizes {n_train}")
    print(f"reproduction of ablation.json at full dev size: {check}")
    _print_tables(curve, floors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
