"""Ablation: does the learned router still beat the single-model frontier without Jev's signals?

Every feature set goes through the identical pipeline as the winning router (per-tier
P(correct) models fit on the dev half only, C chosen by dev out-of-fold log-loss, thresholds
chosen on out-of-fold dev predictions, one scoring pass on the test half). Local features
(length, TF-IDF, embeddings) pay no per-call overhead; Jev-based sets pay the real Jev v2 cost.
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.dataset import LabeledPrompt
from jev_router.learned_router import (
    apply_rule,
    fit_prob_models,
    frontier_cost,
    oof_probs,
    predict_probs,
    select_loss_rule,
    select_threshold_rule,
)
from jev_router.policy import Tier
from jev_router.questions_v2 import FEATURE_NAMES, build_questions_v2, extract_features
from jev_router.router_v2_experiment import (
    C_GRID,
    JEV_V2_CACHE,
    _arrays,
    _dev_logloss,
    load_cached_v2,
)
from jev_router.routing_eval import (
    OPUS,
    ItemOutcome,
    build_outcomes,
    correct_vector,
    random_correct_vector,
    score_assignment,
    single_model,
    split_ids,
)
from jev_router.run_experiment import ROOT, load_all_items

C_GRID_WIDE = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)  # high-dimensional baselines get more tuning room
LENGTH = FEATURE_NAMES.index("log_chars")
JEV_COLUMNS = [i for i in range(len(FEATURE_NAMES)) if i != LENGTH]
SELECTORS = {"threshold": select_threshold_rule, "loss": select_loss_rule}
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Context:
    jev_matrix: np.ndarray  # rows follow FEATURE_NAMES (Jev v2 answers + log prompt length)
    texts: Sequence[str]
    embeddings: np.ndarray | None


@dataclass(frozen=True)
class Spec:
    name: str
    uses_jev: bool
    c_grid: tuple[float, ...]
    build: Callable[[Context, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def tfidf_svd_features(
    dev_texts: Sequence[str], test_texts: Sequence[str], dims: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF then SVD, with the vocabulary and projection fit on the dev texts only."""
    vectorizer = TfidfVectorizer(min_df=2, sublinear_tf=True)
    try:
        dev_matrix = vectorizer.fit_transform(dev_texts)
    except ValueError:  # every term is rarer than min_df on tiny inputs
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        dev_matrix = vectorizer.fit_transform(dev_texts)
    components = max(1, min(dims, dev_matrix.shape[0] - 1, dev_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=0).fit(dev_matrix)
    return svd.transform(dev_matrix), svd.transform(vectorizer.transform(test_texts))


def _split(matrix: np.ndarray, dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[dev], matrix[test]


def build_specs(ctx: Context) -> dict[str, Spec]:
    length = ctx.jev_matrix[:, [LENGTH]]
    specs = {
        "length_only": Spec("length_only", False, C_GRID, lambda c, d, t: _split(length, d, t)),
        "tfidf": Spec(
            "tfidf", False, C_GRID_WIDE,
            lambda c, d, t: tfidf_svd_features([c.texts[i] for i in d], [c.texts[i] for i in t]),
        ),
        "jev_no_length": Spec(
            "jev_no_length", True, C_GRID, lambda c, d, t: _split(c.jev_matrix[:, JEV_COLUMNS], d, t)
        ),
        "jev_plus_length": Spec("jev_plus_length", True, C_GRID, lambda c, d, t: _split(c.jev_matrix, d, t)),
    }
    if ctx.embeddings is not None:
        embedding = ctx.embeddings
        specs["embedding"] = Spec("embedding", False, C_GRID_WIDE, lambda c, d, t: _split(embedding, d, t))
        with_length = np.hstack([embedding, length])
        specs["embedding_plus_length"] = Spec(
            "embedding_plus_length", False, C_GRID_WIDE, lambda c, d, t: _split(with_length, d, t)
        )
        everything = np.hstack([ctx.jev_matrix, embedding])
        specs["full_plus_embedding"] = Spec(
            "full_plus_embedding", True, C_GRID_WIDE, lambda c, d, t: _split(everything, d, t)
        )
    return specs


def load_or_compute_embeddings(
    ids: Sequence[str], texts: Sequence[str], cache_dir: Path, embed: Callable[[Sequence[str]], np.ndarray]
) -> np.ndarray:
    """Embeds each prompt once; the cache is keyed on the ids and texts so edits invalidate it."""
    digest = hashlib.sha256(json.dumps([list(ids), list(texts)]).encode()).hexdigest()
    vectors, meta = cache_dir / "embeddings.npy", cache_dir / "embeddings_meta.json"
    if vectors.exists() and meta.exists() and json.loads(meta.read_text()).get("digest") == digest:
        return np.load(vectors)
    matrix = np.asarray(embed(list(texts)), dtype=float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors, matrix)
    meta.write_text(json.dumps({"digest": digest, "ids": list(ids), "model": EMBEDDING_MODEL}))
    return matrix


def capped_frontier_cost(points: Sequence[tuple[float, float]], accuracy: float) -> float:
    """Frontier cost, capped at the most accurate single model when a router exceeds every point.

    Without the cap a router that is more accurate than Opus has no mixture to compare to and the
    raw gap collapses to -100%, an artifact rather than a saving.
    """
    cost = frontier_cost(points, accuracy)
    if np.isfinite(cost):
        return cost
    best_accuracy = max(a for a, _ in points)
    return min(c for a, c in points if a == best_accuracy)


def evaluate_spec(
    spec: Spec,
    ctx: Context,
    outcomes: Sequence[ItemOutcome],
    jev_costs: np.ndarray,
    seed: int,
    floors: Sequence[float],
    rules: Sequence[str] = ("threshold", "loss"),
) -> list[dict[str, Any]]:
    dev_ids, _ = split_ids([o.item_id for o in outcomes], seed)
    dev_set = set(dev_ids)
    dev = np.array([i for i, o in enumerate(outcomes) if o.item_id in dev_set])
    test = np.array([i for i, o in enumerate(outcomes) if o.item_id not in dev_set])
    dev_o, test_o = [outcomes[i] for i in dev], [outcomes[i] for i in test]
    Xd, Xt = spec.build(ctx, dev, test)
    Yd, Cd = _arrays(dev_o)
    mean_costs = Cd.mean(axis=0)
    opus_dev = single_model(dev_o, OPUS).accuracy
    singles = [single_model(test_o, m) for m in MODELS]
    points = [(r.accuracy, r.total_cost) for r in singles]
    opus_cost = single_model(test_o, OPUS).total_cost
    overhead = float(np.mean(jev_costs[test])) if spec.uses_jev else 0.0

    scored = {c: oof_probs(Xd, Yd, c) for c in spec.c_grid}
    best_c = min(spec.c_grid, key=lambda c: _dev_logloss(scored[c], Yd))
    P_dev = scored[best_c]
    P_test = predict_probs(fit_prob_models(Xd, Yd, best_c), Xt)

    rows = []
    for floor in floors:
        for kind in rules:
            rule = SELECTORS[kind](P_dev, Yd, Cd, floor * opus_dev)
            tiers = [Tier(int(t)) for t in apply_rule(rule, P_test, mean_costs)]
            result = score_assignment(test_o, tiers, overhead)
            mix = result.tier_counts
            vs_random = np.mean(correct_vector(test_o, tiers)) - np.mean(random_correct_vector(test_o, mix))
            rows.append({
                "spec": spec.name, "rule": rule.kind if rule.kind != "opus" else kind, "floor": floor,
                "seed": seed, "n_test": len(test_o), "best_c": best_c,
                "accuracy": result.accuracy, "total_cost": result.total_cost,
                "overhead_cost": result.overhead_cost,
                "gap_pct": 100 * (result.total_cost / capped_frontier_cost(points, result.accuracy) - 1),
                "above_single_max": bool(result.accuracy > max(a for a, _ in points)),
                "savings_vs_opus_pct": 100 * (1 - result.total_cost / opus_cost),
                "acc_vs_random_pp": 100 * float(vs_random),
            })
    return rows


def aggregate(runs: Sequence[Sequence[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per (spec, rule, floor) statistics over splits. Gap below zero = cheaper than the best mix."""
    if not runs:
        raise ValueError("aggregate needs at least one run")
    cells: dict[str, list[dict[str, Any]]] = {}
    for rows in runs:
        for row in rows:
            cells.setdefault(f"{row['spec']}:{row['rule']}:{row['floor']}", []).append(row)
    out = {}
    for key, rows in cells.items():
        gaps = [r["gap_pct"] for r in rows]
        out[key] = {
            "n_splits": len(rows),
            "gap_mean": statistics.fmean(gaps),
            "gap_std": statistics.pstdev(gaps),
            "share_cheaper_than_frontier": sum(g < 0 for g in gaps) / len(gaps),
            "accuracy_mean": statistics.fmean(r["accuracy"] for r in rows),
            "savings_vs_opus_mean": statistics.fmean(r["savings_vs_opus_pct"] for r in rows),
            "acc_vs_random_mean_pp": statistics.fmean(r["acc_vs_random_pp"] for r in rows),
            "splits_above_best_single_model": sum(bool(r.get("above_single_max", False)) for r in rows),
        }
    return out


def paired_contrast(
    runs: Sequence[Sequence[dict[str, Any]]], spec_a: str, spec_b: str, rule: str, floor: float
) -> dict[str, Any]:
    """Split-by-split gap difference (a minus b); negative means spec a is cheaper than spec b.

    Splits are random halves of the same 1,220 items, so they are not independent draws and the
    t statistic is descriptive, not a formal test.
    """
    def gaps(spec: str) -> dict[int, float]:
        return {r["seed"]: r["gap_pct"] for rows in runs for r in rows
                if r["spec"] == spec and r["rule"] == rule and r["floor"] == floor}

    a, b = gaps(spec_a), gaps(spec_b)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise ValueError(f"no shared splits for {spec_a} vs {spec_b} ({rule}, {floor})")
    diffs = [a[s] - b[s] for s in seeds]
    spread = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {
        "spec_a": spec_a, "spec_b": spec_b, "rule": rule, "floor": floor, "n_splits": len(diffs),
        "mean_diff_pct": statistics.fmean(diffs),
        "share_a_cheaper": sum(d < 0 for d in diffs) / len(diffs),
        "t_stat": (statistics.fmean(diffs) / (spread / len(diffs) ** 0.5)) if spread > 0 else 0.0,
    }


CONTRASTS = (
    ("jev_plus_length", "embedding"), ("jev_plus_length", "tfidf"), ("jev_plus_length", "length_only"),
    ("jev_plus_length", "jev_no_length"), ("full_plus_embedding", "embedding"), ("jev_no_length", "embedding"),
)


def _run_seed(args: tuple) -> list[dict[str, Any]]:
    ctx, outcomes, jev_costs, seed, floors = args
    rows: list[dict[str, Any]] = []
    for spec in build_specs(ctx).values():
        rows.extend(evaluate_spec(spec, ctx, outcomes, jev_costs, seed, floors))
    return rows


def _embed_with_fastembed(texts: Sequence[str]) -> np.ndarray:
    from fastembed import TextEmbedding

    return np.array(list(TextEmbedding(EMBEDDING_MODEL).embed(list(texts))))


def load_context() -> tuple[Context, list[ItemOutcome], np.ndarray]:
    items = load_all_items(("routerbench", "mmlu-pro"), 420, 42)
    ready, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    rows = load_cached_v2(prompts, JEV_V2_CACHE, build_questions_v2())
    X = np.array([[extract_features(rows[p.id]["answers"], len(p.text))[n] for n in FEATURE_NAMES] for p in prompts])
    jev_costs = np.array([jev_cost_usd(rows[p.id]["usage"]["input_tokens"]) for p in prompts])
    texts = [p.text for p in prompts]
    try:
        embeddings = load_or_compute_embeddings([p.id for p in prompts], texts, ROOT / "data" / "cache", _embed_with_fastembed)
    except Exception as exc:  # embedding model unavailable: fall back to length/tfidf/Jev sets only
        print(f"embeddings unavailable ({type(exc).__name__}); skipping embedding sets")
        embeddings = None
    return Context(X, texts, embeddings), build_outcomes(ready, records), jev_costs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--floors", default="0.99,0.97,0.95")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    floors = tuple(float(f) for f in args.floors.split(","))
    ctx, outcomes, jev_costs = load_context()

    seed42 = _run_seed((ctx, outcomes, jev_costs, 42, floors))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(_run_seed, [(ctx, outcomes, jev_costs, s, floors) for s in range(args.seeds)]))
    summary = aggregate(runs)
    names = {r["spec"] for rows in runs for r in rows}
    paired = [paired_contrast(runs, a, b, rule, floor) for a, b in CONTRASTS if a in names and b in names
              for rule in ("threshold", "loss") for floor in floors]
    out = ROOT / "data" / "results" / "ablation.json"
    out.write_text(json.dumps({"seed42": seed42, "multi_seed": summary, "paired": paired, "per_seed": runs,
                               "n_items": len(outcomes),
                               "embedding_model": EMBEDDING_MODEL if ctx.embeddings is not None else None}, indent=2))

    print("== seed 42, threshold rule: gap to frontier (accuracy, savings vs Opus) ==")
    for row in seed42:
        if row["rule"] == "threshold":
            print(f"{row['spec']:22s} floor {row['floor']:.2f} gap {row['gap_pct']:+6.1f}%  acc {100*row['accuracy']:5.1f}%  saves {row['savings_vs_opus_pct']:5.1f}%")
    print(f"\n== {args.seeds} splits: mean gap (share of splits cheaper than frontier) ==")
    specs = sorted({k.split(":")[0] for k in summary})
    for rule in ("threshold", "loss"):
        for floor in floors:
            print(f"-- {rule}, floor {floor}")
            for spec in specs:
                v = summary[f"{spec}:{rule}:{floor}"]
                print(f"   {spec:22s} {v['gap_mean']:+7.1f}% (+/-{v['gap_std']:4.1f}) cheaper in {100*v['share_cheaper_than_frontier']:3.0f}%  acc {100*v['accuracy_mean']:5.1f}%  saves {v['savings_vs_opus_mean']:5.1f}%  above-best-single {v['splits_above_best_single_model']}")
    print("\n== paired, split by split: mean gap difference a - b (negative = a cheaper), share of splits a cheaper ==")
    for c in paired:
        print(f"   {c['spec_a']:20s} vs {c['spec_b']:16s} {c['rule']:9s} floor {c['floor']:.2f}  "
              f"{c['mean_diff_pct']:+6.1f}pp  a cheaper in {100*c['share_a_cheaper']:3.0f}%  t={c['t_stat']:+5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
