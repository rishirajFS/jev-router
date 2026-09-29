"""Frozen ablation routers on the FRESH untouched set: does Jev's margin hold on unseen data?

PRE-REGISTRATION (written before any number for this check was computed)
------------------------------------------------------------------------
Every router is fit EXACTLY as in ablation.py's single split, seed 42: on the dev half of the
ORIGINAL 1,220 items (TF-IDF/SVD fit on dev texts only, per-tier P(correct) models with C chosen
by dev out-of-fold log-loss, thresholds chosen on out-of-fold dev predictions, quality floors 0.99,
0.97 and 0.95 of Opus's dev accuracy). It is then FROZEN and applied ONCE to the 594 items in
data/fresh_sample.jsonl, which were never scored or used for design. Feature sets: length_only,
tfidf, embedding, embedding_plus_length, jev_no_length, jev_plus_length, full_plus_embedding.
Non-Jev routers pay no routing cost; Jev routers pay the real six-question Jev cost per fresh item.

Headline rule: "Jev's margin holds on fresh data" means that, under the EXPECTED-LOSS rule on the
combined fresh set, jev_plus_length has a strictly LOWER cost gap to the fresh set's own
single-model frontier than the best of the non-Jev routers (length_only, tfidf, embedding,
embedding_plus_length) at 2 or more of the 3 floors. The threshold rule is also reported. A router
more accurate than every single model has no frontier cost; its gap is reported as undefined and
flagged, never scored as -100%, and it does not count as a win. Sanity check that must pass:
jev_plus_length with the threshold rule reproduces data/results/fresh_heldout.json.
The result is reported whichever way it goes; nothing is re-tuned after seeing it.
"""

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jev_router.ablation import (
    EMBEDDING_MODEL,
    SELECTORS,
    Context,
    Spec,
    _embed_with_fastembed,
    build_specs,
    load_context,
)
from jev_router.anthropic_runner import MODELS
from jev_router.frontier import frontier_cost
from jev_router.fresh_heldout import FLOORS, FRESH_PATH, SEED_ORIGINAL, load_inputs
from jev_router.learned_router import apply_rule, fit_prob_models, oof_probs, predict_probs
from jev_router.policy import Tier
from jev_router.router_v2_experiment import _arrays, _dev_logloss
from jev_router.routing_eval import (
    OPUS,
    ItemOutcome,
    bootstrap_diff,
    correct_vector,
    random_correct_vector,
    score_assignment,
    single_model,
    split_ids,
)
from jev_router.run_experiment import ROOT, load_items

JEV = "jev_plus_length"
NON_JEV = ("length_only", "tfidf", "embedding", "embedding_plus_length")
BOOTSTRAP_REPS = 1000


def gap_or_none(cost: float, accuracy: float, points: Sequence[tuple[float, float]]) -> tuple[float | None, bool]:
    """Cost above the single-model frontier in percent, or (None, True) when it is undefined."""
    frontier = frontier_cost(points, accuracy)
    if frontier is None:
        return None, True
    return 100 * (cost / frontier - 1), False


def margin_holds(
    gaps_by_floor: Mapping[float, Mapping[str, float | None]],
    jev: str = JEV,
    non_jev: Sequence[str] = NON_JEV,
) -> bool:
    """True when Jev's gap is strictly below every defined non-Jev gap at 2 or more floors."""
    wins = 0
    for gaps in gaps_by_floor.values():
        others = [gaps[name] for name in non_jev if gaps.get(name) is not None]
        if gaps.get(jev) is not None and others and gaps[jev] < min(others):
            wins += 1
    return wins >= 2


def combine_contexts(original: Context, fresh: Context) -> tuple[Context, np.ndarray]:
    """Original then fresh rows in one Context, so every feature builder sees both."""
    both_embedded = original.embeddings is not None and fresh.embeddings is not None
    combined = Context(
        jev_matrix=np.vstack([original.jev_matrix, fresh.jev_matrix]),
        texts=[*original.texts, *fresh.texts],
        embeddings=np.vstack([original.embeddings, fresh.embeddings]) if both_embedded else None,
    )
    n_orig = len(original.texts)
    return combined, np.arange(n_orig, n_orig + len(fresh.texts))


def _groups(families: Sequence[str]) -> dict[str, list[int]]:
    groups = {"combined": list(range(len(families)))}
    for name, want in (("routerbench", False), ("mmlu-pro", True)):
        members = [k for k, family in enumerate(families) if (family == "mmlu-pro") == want]
        if members:
            groups[name] = members
    return groups


def evaluate_frozen_spec(
    spec: Spec,
    ctx: Context,
    dev_idx: np.ndarray,
    fresh_idx: np.ndarray,
    dev_outcomes: Sequence[ItemOutcome],
    fresh_outcomes: Sequence[ItemOutcome],
    fresh_jev_costs: np.ndarray,
    floors: Sequence[float],
    families: Sequence[str],
    rules: Sequence[str] = ("threshold", "loss"),
) -> list[dict[str, Any]]:
    """Fit on the original dev half only, then score the frozen router on the fresh items once."""
    Xd, Xf = spec.build(ctx, dev_idx, fresh_idx)
    Yd, Cd = _arrays(list(dev_outcomes))
    mean_costs = Cd.mean(axis=0)
    opus_dev = single_model(list(dev_outcomes), OPUS).accuracy
    scored = {c: oof_probs(Xd, Yd, c) for c in spec.c_grid}
    best_c = min(spec.c_grid, key=lambda c: _dev_logloss(scored[c], Yd))
    P_dev = scored[best_c]
    P_fresh = predict_probs(fit_prob_models(Xd, Yd, best_c), Xf)

    rows: list[dict[str, Any]] = []
    for floor in floors:
        for kind in rules:
            rule = SELECTORS[kind](P_dev, Yd, Cd, floor * opus_dev)
            tiers = [Tier(int(t)) for t in apply_rule(rule, P_fresh, mean_costs)]
            for group, idx in _groups(families).items():
                sub_outcomes = [fresh_outcomes[k] for k in idx]
                sub_tiers = [tiers[k] for k in idx]
                points = [(r.accuracy, r.total_cost) for r in (single_model(sub_outcomes, m) for m in MODELS)]
                overhead = float(np.mean(fresh_jev_costs[idx])) if spec.uses_jev else 0.0
                result = score_assignment(sub_outcomes, sub_tiers, overhead)
                gap, above = gap_or_none(result.total_cost, result.accuracy, points)
                mean, low, high = bootstrap_diff(
                    correct_vector(sub_outcomes, sub_tiers),
                    random_correct_vector(sub_outcomes, result.tier_counts),
                    SEED_ORIGINAL, BOOTSTRAP_REPS,
                )
                rows.append({
                    "spec": spec.name, "rule": kind, "rule_kind": rule.kind, "floor": floor, "group": group,
                    "n": len(idx), "best_c": best_c,
                    "accuracy": result.accuracy, "total_cost": result.total_cost,
                    "overhead_cost": result.overhead_cost,
                    "savings_vs_opus_pct": 100 * (1 - result.total_cost / single_model(sub_outcomes, OPUS).total_cost),
                    "gap_pct": gap, "above_single_max": above,
                    "tier_counts": dict(result.tier_counts),
                    "acc_vs_random": {"mean": mean, "low": low, "high": high},
                })
    return rows


def load_or_compute_fresh_embeddings(
    ids: Sequence[str], texts: Sequence[str], cache_dir: Path, embed: Callable[[Sequence[str]], np.ndarray]
) -> np.ndarray:
    digest = hashlib.sha256(json.dumps([list(ids), list(texts)]).encode()).hexdigest()
    vectors, meta = cache_dir / "embeddings_fresh.npy", cache_dir / "embeddings_fresh_meta.json"
    if vectors.exists() and meta.exists() and json.loads(meta.read_text()).get("digest") == digest:
        return np.load(vectors)
    matrix = np.asarray(embed(list(texts)), dtype=float)
    np.save(vectors, matrix)
    meta.write_text(json.dumps({"digest": digest, "ids": list(ids), "model": EMBEDDING_MODEL}))
    return matrix


def verdicts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered verdict (expected-loss rule, combined set) plus the threshold rule for context."""
    out: dict[str, Any] = {}
    for rule in ("loss", "threshold"):
        by_floor: dict[float, dict[str, float | None]] = {}
        for r in rows:
            if r["rule"] == rule and r["group"] == "combined":
                by_floor.setdefault(r["floor"], {})[r["spec"]] = r["gap_pct"]
        out[rule] = {"margin_holds": margin_holds(by_floor), "gaps_by_floor": {str(f): g for f, g in sorted(by_floor.items())}}
    out["preregistered"] = "holds" if out["loss"]["margin_holds"] else "does not hold"
    return out


def reproduction_check(rows: Sequence[Mapping[str, Any]], recorded_path: Path) -> dict[str, Any]:
    recorded = json.loads(recorded_path.read_text())["groups"]["combined"]["learned"]
    checks = []
    for row in (r for r in recorded if r["rule"] == "threshold"):
        mine = next(r for r in rows if r["spec"] == JEV and r["rule"] == "threshold"
                    and r["group"] == "combined" and r["floor"] == row["quality_floor"])
        checks.append({
            "floor": row["quality_floor"],
            "recorded_gap_pct": row["frontier_gap_pct"], "gap_pct": mine["gap_pct"],
            "same_accuracy": abs(mine["accuracy"] - row["result"]["accuracy"]) < 1e-9,
            "same_cost": abs(mine["total_cost"] - row["result"]["total_cost"]) < 1e-6,
        })
    return {"reproduces": all(c["same_accuracy"] and c["same_cost"] for c in checks), "checks": checks}


def main() -> int:
    ctx, outcomes, _ = load_context()
    items = load_items(FRESH_PATH)
    fresh = load_inputs(items, ROOT / "data" / "cache" / "anthropic.jsonl", ROOT / "data" / "cache" / "jev.jsonl")
    fresh_texts = [i.prompt for i in fresh.items]
    fresh_embeddings = load_or_compute_fresh_embeddings(
        [i.id for i in fresh.items], fresh_texts, ROOT / "data" / "cache", _embed_with_fastembed
    ) if ctx.embeddings is not None else None
    combined, fresh_idx = combine_contexts(ctx, Context(fresh.X, fresh_texts, fresh_embeddings))

    dev_ids, _ = split_ids([o.item_id for o in outcomes], SEED_ORIGINAL)
    dev_set = set(dev_ids)
    dev_idx = np.array([i for i, o in enumerate(outcomes) if o.item_id in dev_set])
    dev_outcomes = [outcomes[i] for i in dev_idx]
    families = [i.family for i in fresh.items]
    jev_costs = np.array(fresh.v2_cost)

    rows: list[dict[str, Any]] = []
    for spec in build_specs(combined).values():
        rows.extend(evaluate_frozen_spec(spec, combined, dev_idx, fresh_idx, dev_outcomes, fresh.outcomes,
                                         jev_costs, FLOORS, families))
    groups = _groups(families)
    baselines = {g: {m: {"accuracy": single_model([fresh.outcomes[k] for k in idx], m).accuracy,
                         "total_cost": single_model([fresh.outcomes[k] for k in idx], m).total_cost}
                     for m in MODELS} for g, idx in groups.items()}
    report = {
        "n_fresh": len(fresh.items), "embedding_model": EMBEDDING_MODEL if ctx.embeddings is not None else None,
        "baselines": baselines, "rows": rows, "verdicts": verdicts(rows),
        "reproduction": reproduction_check(rows, ROOT / "data" / "results" / "fresh_heldout.json"),
    }
    (ROOT / "data" / "results" / "ablation_fresh.json").write_text(json.dumps(report, indent=2))
    print(f"fresh n={report['n_fresh']} | reproduces fresh_heldout.json: {report['reproduction']['reproduces']}")
    for rule in ("loss", "threshold"):
        print(f"\n== {rule} rule, combined fresh set: gap to fresh frontier % (accuracy) ==")
        specs = list(dict.fromkeys(r["spec"] for r in rows))
        for spec in specs:
            cells = []
            for floor in FLOORS:
                r = next(x for x in rows if x["spec"] == spec and x["rule"] == rule and x["group"] == "combined" and x["floor"] == floor)
                gap = "  undef" if r["gap_pct"] is None else f"{r['gap_pct']:+6.1f}%"
                cells.append(f"{gap} ({100 * r['accuracy']:4.1f}%)")
            print(f"  {spec:22s} " + "  ".join(cells))
    print(f"\npre-registered verdict (expected-loss rule): {report['verdicts']['preregistered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
