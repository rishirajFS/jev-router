"""Learned Jev router: richer questions, dev-only fitting, held-out test scoring."""

import argparse
import hashlib
import json
import random
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from jev_router.anthropic_runner import MODELS, load_cached_records
from jev_router.budget import jev_cost_usd
from jev_router.config import load_api_key
from jev_router.dataset import LabeledPrompt
from jev_router.experiment import evaluate_routing, summary
from jev_router.feasibility import load_cached_features, load_cached_jev_tokens
from jev_router.jev_client import JevClient
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
from jev_router.questions_v2 import (
    FEATURE_NAMES,
    FEATURE_SETS,
    build_questions_v2,
    extract_features,
    select_columns,
)
from jev_router.routing_eval import (
    HAIKU,
    OPUS,
    SONNET,
    ItemOutcome,
    bootstrap_diff,
    build_outcomes,
    correct_vector,
    length_matched,
    length_tiers,
    random_correct_vector,
    random_matched,
    score_assignment,
    single_model,
    split_ids,
)
from jev_router.run_experiment import load_all_items

ROOT = Path(__file__).resolve().parents[2]
JEV_V2_CACHE = ROOT / "data" / "cache" / "jev_v2.jsonl"
RESERVE_TOKENS = 2500  # conservative allowance for the question definitions in each call
C_GRID = (0.03, 0.1, 0.3, 1.0)


class JevSpendExceeded(RuntimeError):
    pass


def _key(model: str, text: str, questions: dict[str, Any]) -> str:
    blob = json.dumps([model, text, questions], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
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


def jev_ledger(cache_path: Path) -> float:
    """Dollars billed for every call ever written to this cache (input tokens only)."""
    total = 0.0
    if not cache_path.exists():
        return total
    for line in cache_path.read_text().splitlines():
        try:
            total += jev_cost_usd(int(json.loads(line)["usage"]["input_tokens"]))
        except (ValueError, KeyError, TypeError):
            continue
    return total


def load_cached_v2(
    prompts: Sequence[LabeledPrompt], cache_path: Path, questions: dict[str, Any], model: str = "jev-latest"
) -> dict[str, dict[str, Any]]:
    cache = _read_cache(cache_path)
    return {p.id: cache[k] for p in prompts if (k := _key(model, p.text, questions)) in cache}


def collect_v2(
    prompts: Sequence[LabeledPrompt],
    client: Any,
    cache_path: Path,
    questions: dict[str, Any],
    workers: int,
    cap_usd: float,
) -> dict[str, dict[str, Any]]:
    """Runs uncached prompts through Jev, saving raw answers and usage, under a spend cap."""
    model = getattr(client, "model", "unknown")
    keys = {p.id: _key(model, p.text, questions) for p in prompts}
    cache = _read_cache(cache_path)
    todo: dict[str, LabeledPrompt] = {}
    for p in prompts:
        if keys[p.id] not in cache:
            todo.setdefault(keys[p.id], p)
    lock = threading.Lock()
    spent = [jev_ledger(cache_path)]
    stopped = threading.Event()

    def fetch(item: tuple[str, LabeledPrompt]) -> dict[str, Any] | None:
        key, prompt = item
        reserve = jev_cost_usd(len(prompt.text) // 2 + RESERVE_TOKENS)
        with lock:
            if stopped.is_set() or spent[0] + reserve > cap_usd:
                stopped.set()
                return None
        result = client.analyze(prompt.text, questions)
        with lock:
            spent[0] += jev_cost_usd(result.usage.get("input_tokens", 0))
        return {
            "key": key,
            "prompt_id": prompt.id,
            "answers": result.answers,
            "usage": result.usage,
            "model_returned": result.model,
            "latency_seconds": result.latency_seconds,
        }

    failures = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as out:
        for item, row in zip(todo.items(), pool.map(_safe(fetch), todo.items())):
            if row is None:
                failures += 0 if stopped.is_set() else 1
                continue
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[row["key"]] = row
    if stopped.is_set():
        raise JevSpendExceeded(f"Jev spend would exceed ${cap_usd:.2f}; paid results are cached")
    if failures:
        raise RuntimeError(f"{failures} Jev call(s) failed; successes are cached")
    return {p.id: cache[keys[p.id]] for p in prompts}


def _safe(fn):
    def wrapped(item):
        try:
            return fn(item)
        except Exception:
            return None

    return wrapped


def _arrays(outcomes: Sequence[ItemOutcome]) -> tuple[np.ndarray, np.ndarray]:
    correct = np.array([[float(o.correct[m]) for m in MODELS] for o in outcomes])
    cost = np.array([[o.cost[m] for m in MODELS] for o in outcomes])
    return correct, cost


def _ci(a: Sequence[float], b: Sequence[float], seed: int) -> dict[str, float]:
    mean, low, high = bootstrap_diff(a, b, seed)
    return {"mean": mean, "low": low, "high": high}


def _dev_logloss(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean([log_loss(Y[:, k], np.clip(P[:, k], 1e-4, 1 - 1e-4), labels=[0, 1]) for k in (0, 1)]))


def score_v2(
    outcomes: Sequence[ItemOutcome],
    X_full: np.ndarray,
    v2_cost: Sequence[float],
    v1_features: Sequence[Any],
    v1_cost: float,
    seed: int,
    floors: Sequence[float],
) -> dict[str, Any]:
    dev_ids, _ = split_ids([o.item_id for o in outcomes], seed)
    dev_set = set(dev_ids)
    dev = [i for i, o in enumerate(outcomes) if o.item_id in dev_set]
    test = [i for i, o in enumerate(outcomes) if o.item_id not in dev_set]
    dev_o, test_o = [outcomes[i] for i in dev], [outcomes[i] for i in test]
    Yd, Cd = _arrays(dev_o)
    mean_costs = Cd.mean(axis=0)
    opus_dev = single_model(dev_o, OPUS).accuracy
    points = [(r.accuracy, r.total_cost) for r in (single_model(test_o, m) for m in (HAIKU, SONNET, OPUS))]
    test_overhead = float(np.mean([v2_cost[i] for i in test]))

    def gap(accuracy: float, total: float) -> float:
        return 100 * (total / frontier_cost(points, accuracy) - 1)

    report: dict[str, Any] = {
        "n_dev": len(dev), "n_test": len(test),
        "jev_v2_overhead_per_call": test_overhead,
        "baselines": {
            "always_haiku": summary(single_model(test_o, HAIKU)),
            "always_sonnet": summary(single_model(test_o, SONNET)),
            "always_opus": summary(single_model(test_o, OPUS)),
        },
    }
    old = evaluate_routing(list(outcomes), list(v1_features), seed, floors, v1_cost)
    report["old_threshold"] = [
        {"quality_floor": r["quality_floor"], "jev": r["jev"], "savings_vs_opus_pct": r["savings_vs_opus_pct"],
         "accuracy_vs_random_matched": r["accuracy_vs_random_matched"],
         "accuracy_vs_length_matched": r["accuracy_vs_length_matched"],
         "frontier_gap_pct": gap(r["jev"]["accuracy"], r["jev"]["total_cost"])}
        for r in old["jev"]
    ]
    report["learned"] = {}
    for set_name, names in FEATURE_SETS.items():
        X = select_columns(X_full, names)
        Xd, Xt = X[dev], X[test]
        scored = {c: oof_probs(Xd, Yd, c) for c in C_GRID}
        best_c = min(C_GRID, key=lambda c: _dev_logloss(scored[c], Yd))
        P_dev = scored[best_c]
        P_test = predict_probs(fit_prob_models(Xd, Yd, best_c), Xt)
        rows = []
        for floor in floors:
            floor_acc = floor * opus_dev
            for select in (select_threshold_rule, select_loss_rule):
                rule = select(P_dev, Yd, Cd, floor_acc)
                tiers = [Tier(int(t)) for t in apply_rule(rule, P_test, mean_costs)]
                result = score_assignment(test_o, tiers, test_overhead)
                mix = result.tier_counts
                rows.append({
                    "quality_floor": floor, "rule": rule.kind, "params": list(rule.params),
                    "dev_oof_accuracy": float(Yd[np.arange(len(dev)), apply_rule(rule, P_dev, mean_costs)].mean()),
                    "result": summary(result),
                    "savings_vs_opus_pct": 100 * (1 - result.total_cost / single_model(test_o, OPUS).total_cost),
                    "accuracy_vs_random_matched": _ci(correct_vector(test_o, tiers), random_correct_vector(test_o, mix), seed),
                    "accuracy_vs_length_matched": _ci(correct_vector(test_o, tiers), correct_vector(test_o, length_tiers(test_o, mix)), seed),
                    "random_matched": summary(random_matched(test_o, mix)),
                    "length_matched": summary(length_matched(test_o, mix)),
                    "frontier_gap_pct": gap(result.accuracy, result.total_cost),
                })
        report["learned"][set_name] = {"c": best_c, "dev_oof_logloss": _dev_logloss(P_dev, Yd), "rows": rows}
    report["primary_feature_set"] = min(report["learned"], key=lambda s: report["learned"][s]["dev_oof_logloss"])
    return report


def format_v2(report: dict[str, Any], jev_spend: float) -> str:
    lines = [f"test n={report['n_test']} (fit on dev n={report['n_dev']}); Jev v2 spend ${jev_spend:.3f}; "
             f"v2 overhead/call ${report['jev_v2_overhead_per_call']:.6f}"]
    for k, v in report["baselines"].items():
        lines.append(f"{k:15s} acc {100*v['accuracy']:5.1f}%  cost ${v['total_cost']:.4f}")
    lines.append("-- old difficulty-threshold policy --")
    for r in report["old_threshold"]:
        j = r["jev"]
        lines.append(f"  floor {r['quality_floor']:.0%}: acc {100*j['accuracy']:5.1f}% cost ${j['total_cost']:.4f} "
                     f"saves {r['savings_vs_opus_pct']:4.0f}% | frontier gap {r['frontier_gap_pct']:+5.1f}%")
    for name, block in report["learned"].items():
        star = " (dev-selected primary)" if name == report["primary_feature_set"] else ""
        lines.append(f"-- learned router, features={name}{star}, C={block['c']}, dev logloss {block['dev_oof_logloss']:.4f} --")
        for r in block["rows"]:
            x = r["result"]; a = r["accuracy_vs_random_matched"]; b = r["accuracy_vs_length_matched"]
            lines.append(
                f"  floor {r['quality_floor']:.0%} {r['rule']:9s}: acc {100*x['accuracy']:5.1f}% cost ${x['total_cost']:.4f} "
                f"saves {r['savings_vs_opus_pct']:4.0f}% | vs rand {100*a['mean']:+.1f}pp [{100*a['low']:+.1f},{100*a['high']:+.1f}] "
                f"vs len {100*b['mean']:+.1f}pp [{100*b['low']:+.1f},{100*b['high']:+.1f}] | frontier gap {r['frontier_gap_pct']:+5.1f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("explore", "full", "score"), default="score")
    parser.add_argument("--explore-n", type=int, default=100)
    parser.add_argument("--cap", type=float, default=0.40, help="cumulative Jev v2 spend cap in USD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--floors", default="0.99,0.97,0.95")
    args = parser.parse_args(argv)

    items = load_all_items(("routerbench", "mmlu-pro"), 420, args.seed)
    ready, records = load_cached_records(items, MODELS, ROOT / "data" / "cache" / "anthropic.jsonl")
    prompts = tuple(LabeledPrompt(i.id, i.prompt, i.family, 0) for i in ready)
    questions = build_questions_v2()

    if args.phase in ("explore", "full"):
        chosen = prompts
        if args.phase == "explore":
            dev_ids, _ = split_ids([p.id for p in prompts], args.seed)
            dev_prompts = [p for p in prompts if p.id in set(dev_ids)]
            chosen = tuple(_take(dev_prompts, args.explore_n, args.seed))
        client = JevClient(load_api_key(ROOT / ".env"))
        try:
            collect_v2(chosen, client, JEV_V2_CACHE, questions, workers=6, cap_usd=args.cap)
        except JevSpendExceeded as exc:
            print(f"STOPPED: {exc}")
            return 2
        finally:
            client.close()
        rows = load_cached_v2(chosen, JEV_V2_CACHE, questions)
        tokens = [r["usage"]["input_tokens"] for r in rows.values()]
        print(f"calls cached: {len(rows)} | mean input tokens/call {np.mean(tokens):.0f} | "
              f"ledger ${jev_ledger(JEV_V2_CACHE):.4f} | projected full pass ${np.mean(tokens) * len(prompts) * 0.042 / 1e6:.3f}")
        if args.phase == "explore":
            return 0

    rows = load_cached_v2(prompts, JEV_V2_CACHE, questions)
    missing = [p.id for p in prompts if p.id not in rows]
    if missing:
        print(f"{len(missing)} items lack v2 Jev answers; run --phase full first")
        return 1
    outcomes = build_outcomes(ready, records)
    X = np.array([[extract_features(rows[p.id]["answers"], len(p.text))[n] for n in FEATURE_NAMES] for p in prompts])
    v2_cost = [jev_cost_usd(rows[p.id]["usage"]["input_tokens"]) for p in prompts]
    v1 = load_cached_features(prompts, ROOT / "data" / "cache" / "jev.jsonl")
    v1_tokens = load_cached_jev_tokens(prompts, ROOT / "data" / "cache" / "jev.jsonl")
    v1_cost = float(np.mean([jev_cost_usd(v1_tokens[p.id]) for p in prompts]))
    floors = tuple(float(f) for f in args.floors.split(","))
    report = score_v2(outcomes, X, v2_cost, [v1[p.id] for p in prompts], v1_cost, args.seed, floors)
    spend = jev_ledger(JEV_V2_CACHE)
    report["jev_v2_total_spend"] = spend
    out = ROOT / "data" / "results" / "router_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(format_v2(report, spend))
    return 0


def _take(prompts: Sequence[LabeledPrompt], n: int, seed: int) -> list[LabeledPrompt]:
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


if __name__ == "__main__":
    sys.exit(main())
