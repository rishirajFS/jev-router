"""Diagnostics for the learned router: feature ablations and per-dataset frontier gaps."""

import sys

import numpy as np

from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.dataset import LabeledPrompt
from jev_router.learned_router import (
    apply_rule, fit_prob_models, frontier_cost, oof_probs, predict_probs, select_threshold_rule,
)
from jev_router.policy import Tier
from jev_router.questions_v2 import FEATURE_NAMES, FEATURE_SETS, build_questions_v2, extract_features, select_columns
from jev_router.router_v2_experiment import JEV_V2_CACHE, ROOT, C_GRID, _arrays, _dev_logloss, load_cached_v2
from jev_router.routing_eval import HAIKU, OPUS, SONNET, build_outcomes, score_assignment, single_model, split_ids
from jev_router.run_experiment import load_all_items

ABLATIONS = {
    "difficulty_only": FEATURE_SETS["difficulty_only"],
    "tier_nouls (no length, no task)": FEATURE_SETS["difficulty_only"] + ("small_ok", "mid_ok", "expert_needed"),
    "full minus log_chars": tuple(n for n in FEATURE_NAMES if n != "log_chars"),
    "full": FEATURE_NAMES,
}


def main() -> int:
    items = load_all_items(("routerbench", "mmlu-pro"), 420, 42)
    ready, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    rows = load_cached_v2(prompts, JEV_V2_CACHE, build_questions_v2())
    outcomes = build_outcomes(ready, records)
    X = np.array([[extract_features(rows[p.id]["answers"], len(p.text))[n] for n in FEATURE_NAMES] for p in prompts])
    overhead = float(np.mean([jev_cost_usd(rows[p.id]["usage"]["input_tokens"]) for p in prompts]))
    dev_ids, _ = split_ids([o.item_id for o in outcomes], 42)
    dev_set = set(dev_ids)
    dev = [i for i, o in enumerate(outcomes) if o.item_id in dev_set]
    test = [i for i, o in enumerate(outcomes) if o.item_id not in dev_set]
    dev_o = [outcomes[i] for i in dev]
    Yd, Cd = _arrays(dev_o)
    opus_dev = single_model(dev_o, OPUS).accuracy
    groups = {"routerbench": [k for k in test if ready[k].family != "mmlu-pro"],
              "mmlu-pro": [k for k in test if ready[k].family == "mmlu-pro"]}
    frontiers = {}
    for g, idx in groups.items():
        sub = [outcomes[k] for k in idx]
        frontiers[g] = [(r.accuracy, r.total_cost) for r in (single_model(sub, m) for m in (HAIKU, SONNET, OPUS))]
    pos = {k: n for n, k in enumerate(test)}
    for name, cols in ABLATIONS.items():
        Xm = select_columns(X, cols)
        scored = {c: oof_probs(Xm[dev], Yd, c) for c in C_GRID}
        c = min(C_GRID, key=lambda v: _dev_logloss(scored[v], Yd))
        P_test = predict_probs(fit_prob_models(Xm[dev], Yd, c), Xm[test])
        print(f"== {name} (C={c}, dev logloss {_dev_logloss(scored[c], Yd):.4f})")
        for floor in (0.99, 0.97, 0.95):
            rule = select_threshold_rule(scored[c], Yd, Cd, floor * opus_dev)
            tiers = [Tier(int(t)) for t in apply_rule(rule, P_test, Cd.mean(axis=0))]
            parts = []
            for g, idx in groups.items():
                sub = [outcomes[k] for k in idx]
                sub_tiers = [tiers[pos[k]] for k in idx]
                r = score_assignment(sub, sub_tiers, overhead)
                gap = 100 * (r.total_cost / frontier_cost(frontiers[g], r.accuracy) - 1)
                parts.append(f"{g}: acc {100*r.accuracy:5.1f}% cost ${r.total_cost:.3f} gap {gap:+5.1f}%")
            print(f"  floor {floor:.0%} | " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
