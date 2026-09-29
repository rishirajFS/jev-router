from jev_router.dataset import LabeledPrompt
from jev_router.feasibility import collect, summarize
from jev_router.jev_client import JevResult
from jev_router.policy import Policy

PROMPTS = (
    LabeledPrompt("a", "easy thing", "chat", 0),
    LabeledPrompt("b", "hard thing", "math", 2),
)


class FakeClient:
    def __init__(self):
        self.calls = 0

    def analyze(self, state, questions):
        self.calls += 1
        hard = "hard" in state
        return JevResult(
            model="jev-test",
            answers={
                "task_type": {"choice": "math" if hard else "chat", "confidence": 0.9},
                "difficulty": {"score": 2.6 if hard else 0.2, "confidence": 0.9},
                "multistep": {"noul": 0.9 if hard else 0.05},
            },
            usage={"input_tokens": 100, "output_tokens": 10},
            latency_seconds=0.2,
        )


def test_collect_caches_results_between_runs(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = FakeClient()
    first = collect(PROMPTS, client, cache, workers=2)
    second = collect(PROMPTS, client, cache, workers=2)
    assert client.calls == 2
    assert [r.prompt_id for r in first] == [r.prompt_id for r in second] == ["a", "b"]


def test_collect_reruns_when_prompt_text_changes(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = FakeClient()
    collect(PROMPTS, client, cache, workers=1)
    changed = (LabeledPrompt("a", "different easy thing", "chat", 0), PROMPTS[1])
    collect(changed, client, cache, workers=1)
    assert client.calls == 3


def test_summarize_reports_perfect_routing_for_clean_signal(tmp_path):
    records = collect(PROMPTS, FakeClient(), tmp_path / "c.jsonl", workers=1)
    report = summarize(PROMPTS, records, Policy())
    assert report["routing"]["exact_accuracy"] == 1.0
    assert report["routing"]["under_routed"] == 0.0
    assert report["difficulty_spearman"] == 1.0
    assert report["task_type_accuracy"] == 1.0
    assert report["mean_latency_seconds"] == 0.2
    assert report["total_tokens"] == 220


def test_failure_keeps_successful_results_in_cache(tmp_path):
    import pytest

    from jev_router.feasibility import CollectError

    class Flaky(FakeClient):
        def analyze(self, state, questions):
            if "hard" in state:
                raise RuntimeError("boom")
            return super().analyze(state, questions)

    cache = tmp_path / "cache.jsonl"
    with pytest.raises(CollectError) as exc:
        collect(PROMPTS, Flaky(), cache, workers=1)
    assert exc.value.failed_ids == ("b",)

    client = FakeClient()
    records = collect(PROMPTS, client, cache, workers=1)
    assert client.calls == 1
    assert len(records) == 2


def test_collect_skips_corrupt_cache_lines_and_dedupes(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text('{"key": "partial\n')
    dupes = (PROMPTS[0], LabeledPrompt("a2", PROMPTS[0].text, "chat", 0))
    client = FakeClient()
    records = collect(dupes, client, cache, workers=1)
    assert client.calls == 1
    assert [r.prompt_id for r in records] == ["a", "a2"]


def test_summarize_rejects_mismatched_and_empty_inputs(tmp_path):
    import pytest

    records = collect(PROMPTS, FakeClient(), tmp_path / "c.jsonl", workers=1)
    with pytest.raises(ValueError):
        summarize(PROMPTS[:1], records, Policy())
    with pytest.raises(ValueError):
        summarize((), [], Policy())


def test_jev_cache_keeps_the_raw_answers_and_usage(tmp_path):
    import json

    cache = tmp_path / "cache.jsonl"
    collect(PROMPTS, FakeClient(), cache, workers=1)
    row = json.loads(cache.read_text().splitlines()[0])
    assert row["answers"]["difficulty"]["score"] in (0.2, 2.6)
    assert row["usage"] == {"input_tokens": 100, "output_tokens": 10}
    assert row["model_returned"] == "jev-test"


def test_load_cached_jev_tokens_returns_only_cached_prompts(tmp_path):
    from jev_router.feasibility import load_cached_jev_tokens

    cache = tmp_path / "cache.jsonl"
    collect(PROMPTS[:1], FakeClient(), cache, workers=1)
    tokens = load_cached_jev_tokens(PROMPTS, cache, model="unknown")
    assert tokens == {"a": 100}  # input tokens only; Jev output is free
