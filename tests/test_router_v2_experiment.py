import json

import pytest

from jev_router.dataset import LabeledPrompt
from jev_router.jev_client import JevResult
from jev_router.router_v2_experiment import JevSpendExceeded, collect_v2, jev_ledger

PROMPTS = (LabeledPrompt("a", "first", "chat", 0), LabeledPrompt("b", "second", "chat", 0))
QUESTIONS = {"q": {"type": "noul", "instructions": "?"}}


class FakeJev:
    def __init__(self, tokens=1000):
        self.calls = 0
        self.tokens = tokens

    def analyze(self, state, questions):
        self.calls += 1
        return JevResult("jev-test", {"q": {"type": "noul", "noul": 0.5}},
                         {"input_tokens": self.tokens, "output_tokens": 5}, 0.1)


def test_collect_v2_caches_raw_answers_and_usage(tmp_path):
    cache = tmp_path / "v2.jsonl"
    rows = collect_v2(PROMPTS, FakeJev(), cache, QUESTIONS, workers=1, cap_usd=0.4)
    assert set(rows) == {"a", "b"}
    saved = [json.loads(line) for line in cache.read_text().splitlines()]
    assert saved[0]["answers"]["q"]["noul"] == 0.5
    assert saved[0]["usage"] == {"input_tokens": 1000, "output_tokens": 5}


def test_collect_v2_reuses_the_cache(tmp_path):
    cache = tmp_path / "v2.jsonl"
    client = FakeJev()
    collect_v2(PROMPTS, client, cache, QUESTIONS, workers=1, cap_usd=0.4)
    collect_v2(PROMPTS, client, cache, QUESTIONS, workers=1, cap_usd=0.4)
    assert client.calls == 2


def test_ledger_charges_input_tokens_only(tmp_path):
    cache = tmp_path / "v2.jsonl"
    collect_v2(PROMPTS, FakeJev(tokens=1_000_000), cache, QUESTIONS, workers=1, cap_usd=1.0)
    assert jev_ledger(cache) == pytest.approx(2 * 0.042)
    assert jev_ledger(tmp_path / "missing.jsonl") == 0.0


def test_collect_v2_stops_at_the_spend_cap_and_keeps_paid_results(tmp_path):
    cache = tmp_path / "v2.jsonl"
    client = FakeJev(tokens=1_000_000)  # $0.042 per call
    with pytest.raises(JevSpendExceeded):
        many = tuple(LabeledPrompt(f"p{i}", f"text {i}", "chat", 0) for i in range(6))
        collect_v2(many, client, cache, QUESTIONS, workers=1, cap_usd=0.05)
    assert client.calls <= 3
    assert cache.exists() and cache.read_text().strip()


def test_score_v2_fits_on_dev_and_reports_test_with_overhead():
    import numpy as np

    from jev_router.anthropic_runner import MODELS
    from jev_router.features import RoutingFeatures
    from jev_router.questions_v2 import FEATURE_NAMES, FEATURE_SETS
    from jev_router.router_v2_experiment import score_v2
    from jev_router.routing_eval import ItemOutcome

    rng = np.random.RandomState(0)
    n = 300
    hard = rng.rand(n) < 0.4
    outcomes, v1 = [], []
    for i in range(n):
        outcomes.append(ItemOutcome(f"i{i}", "mmlu", 200, {MODELS[0]: not hard[i], MODELS[1]: bool((not hard[i]) or rng.rand() < 0.5), MODELS[2]: True},
                                    {MODELS[0]: 0.001, MODELS[1]: 0.004, MODELS[2]: 0.010}))
        v1.append(RoutingFeatures("math", 0.9, 2.6 if hard[i] else 0.3, 0.9, 0.1))
    X = rng.randn(n, len(FEATURE_NAMES))
    X[:, FEATURE_NAMES.index("small_ok")] = np.where(hard, 0.1, 0.9) + 0.05 * rng.randn(n)
    report = score_v2(outcomes, X, [0.00005] * n, v1, 0.00004, seed=42, floors=(0.99, 0.95))

    assert report["n_dev"] + report["n_test"] == n
    assert report["primary_feature_set"] in FEATURE_SETS
    rows = report["learned"]["full"]["rows"]
    assert len(rows) == 2 * 2  # two floors x two rules
    first = rows[0]
    assert first["result"]["overhead_cost"] == pytest.approx(0.00005 * report["n_test"])
    assert first["result"]["accuracy"] >= 0.9  # the informative column is usable
    assert {"mean", "low", "high"} <= set(first["accuracy_vs_random_matched"])
    assert "frontier_gap_pct" in first
    assert len(report["old_threshold"]) == 2
    json.dumps(report)
