from types import SimpleNamespace

import pytest

from jev_router.anthropic_runner import MODELS, grade, request_kwargs, run
from jev_router.budget import BudgetExceeded, BudgetTracker
from jev_router.sampling import BenchItem

ITEMS = (
    BenchItem("q1", "mmlu", "Q1?\nA) x\nB) y\nC) z\nD) w\nPrint only a single choice", "B", 0.5),
    BenchItem("q2", "mmlu", "Q2?\nA) x\nB) y\nC) z\nD) w\nPrint only a single choice", "C", 0.2),
)


class FakeMessages:
    def __init__(self, answers, stop_reason="end_turn"):
        self.answers = answers
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        question = kwargs["messages"][0]["content"]
        letter = self.answers["q1" if question.startswith("Q1") else "q2"]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=letter)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


class FakeClient:
    def __init__(self, answers, **kw):
        self.messages = FakeMessages(answers, **kw)


def test_request_kwargs_sets_effort_only_where_supported():
    opus = request_kwargs("claude-opus-5-5", "hi")
    assert opus["output_config"] == {"effort": "low"}
    assert "temperature" not in opus and "thinking" not in opus
    haiku = request_kwargs("claude-haiku-4-5", "hi")
    assert "output_config" not in haiku
    assert all(k["max_tokens"] > 0 for k in (opus, haiku))


def test_grade_compares_extracted_letter_to_truth():
    assert grade(ITEMS[0], "B") is True
    assert grade(ITEMS[0], "C") is False
    assert grade(ITEMS[0], "no idea") is False


def test_run_grades_every_item_for_every_model_and_tracks_cost(tmp_path):
    client = FakeClient({"q1": "B", "q2": "A"})
    tracker = BudgetTracker(cap_usd=5.0)
    records = run(ITEMS, MODELS, client, tmp_path / "c.jsonl", tracker, workers=2)
    assert len(records) == len(ITEMS) * len(MODELS)
    by_item = {(r.item_id, r.model): r.correct for r in records}
    assert by_item[("q1", "claude-opus-5-5")] is True
    assert by_item[("q2", "claude-opus-5-5")] is False
    assert tracker.spent > 0


def test_run_uses_cache_and_does_not_repay(tmp_path):
    cache = tmp_path / "c.jsonl"
    client = FakeClient({"q1": "B", "q2": "C"})
    tracker = BudgetTracker(cap_usd=5.0)
    run(ITEMS, MODELS, client, cache, tracker, workers=1)
    calls_after_first = len(client.messages.calls)
    second = BudgetTracker(cap_usd=5.0)
    run(ITEMS, MODELS, client, cache, second, workers=1)
    assert len(client.messages.calls) == calls_after_first
    assert second.spent == 0


def test_run_stops_before_exceeding_the_budget(tmp_path):
    client = FakeClient({"q1": "B", "q2": "C"})
    tracker = BudgetTracker(cap_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        run(ITEMS, MODELS, client, tmp_path / "c.jsonl", tracker, workers=1)
    assert len(client.messages.calls) == 0


def test_refusals_are_recorded_as_incorrect(tmp_path):
    client = FakeClient({"q1": "B", "q2": "C"}, stop_reason="refusal")
    records = run(ITEMS, ("claude-haiku-4-5",), client, tmp_path / "c.jsonl", BudgetTracker(5.0), workers=1)
    assert all(r.correct is False and r.stop_reason == "refusal" for r in records)


def test_cache_rows_keep_the_full_response_and_request(tmp_path):
    import json

    cache = tmp_path / "c.jsonl"
    run(ITEMS, ("claude-opus-5-5",), FakeClient({"q1": "B", "q2": "C"}), cache, BudgetTracker(5.0), workers=1)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert {r["text"] for r in rows} == {"B", "C"}
    assert all(r["content_types"] == ["text"] for r in rows)
    assert all(r["request"]["output_config"] == {"effort": "low"} for r in rows)
    assert all("messages" not in r["request"] for r in rows)


def test_ledger_sums_every_billed_line_and_tracker_starts_from_it(tmp_path):
    import json

    from jev_router.anthropic_runner import ledger_spent, make_tracker

    cache = tmp_path / "c.jsonl"
    cache.write_text(
        json.dumps({"key": "a", "cost_usd": 0.5}) + "\n"
        + json.dumps({"key": "a", "cost_usd": 0.25}) + "\n"  # a re-run of the same key was billed too
        + "not json\n"
    )
    assert ledger_spent(cache) == pytest.approx(0.75)
    tracker = make_tracker(cap_usd=1.0, cache_path=cache)
    assert tracker.spent == pytest.approx(0.75)
    assert ledger_spent(tmp_path / "missing.jsonl") == 0.0


def test_cumulative_cap_blocks_new_calls_when_ledger_is_nearly_full(tmp_path):
    import json

    from jev_router.anthropic_runner import make_tracker

    cache = tmp_path / "c.jsonl"
    cache.write_text(json.dumps({"key": "old", "cost_usd": 0.99}) + "\n")
    client = FakeClient({"q1": "B", "q2": "C"})
    with pytest.raises(BudgetExceeded):
        run(ITEMS, MODELS, client, cache, make_tracker(1.0, cache), workers=1)
    assert client.messages.calls == []


def test_rows_without_saved_text_are_rerun_so_the_cache_ends_complete(tmp_path):
    import json

    from jev_router.anthropic_runner import _key

    cache = tmp_path / "c.jsonl"
    stale = {"key": _key(ITEMS[0], "claude-haiku-4-5"), "item_id": "q1", "model": "claude-haiku-4-5",
             "correct": True, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.001,
             "stop_reason": "end_turn", "latency_seconds": 0.1}
    cache.write_text(json.dumps(stale) + "\n")
    client = FakeClient({"q1": "B", "q2": "C"})
    run(ITEMS[:1], ("claude-haiku-4-5",), client, cache, BudgetTracker(5.0), workers=1)
    assert len(client.messages.calls) == 1
    assert "text" in json.loads(cache.read_text().splitlines()[-1])


def test_worst_case_reservation_bounds_concurrent_calls(tmp_path):
    import threading
    import time

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class Slow(FakeMessages):
        def create(self, **kwargs):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return super().create(**kwargs)

    client = FakeClient({"q1": "B", "q2": "C"})
    client.messages = Slow({"q1": "B", "q2": "C"})
    many = [BenchItem(f"q{i}", "mmlu", "Q1?\nA) x\nB) y\nPrint", "B", 0.5) for i in range(12)]
    # each haiku call reserves ~$0.0205 worst case, so a $0.05 cap admits at most 2 at once
    records = run(many, ("claude-haiku-4-5",), client, tmp_path / "c.jsonl", BudgetTracker(0.05), workers=6)
    assert len(records) == 12  # calls queue for a reservation instead of aborting the run
    assert state["peak"] <= 2


def test_load_cached_records_regrades_from_saved_text(tmp_path):
    from jev_router.anthropic_runner import load_cached_records

    cache = tmp_path / "c.jsonl"
    run(ITEMS, ("claude-haiku-4-5",), FakeClient({"q1": "B", "q2": "A"}), cache, BudgetTracker(5.0), workers=1)
    items, records = load_cached_records(ITEMS, ("claude-haiku-4-5",), cache)
    assert {r.item_id: r.correct for r in records} == {"q1": True, "q2": False}
    corrected = (ITEMS[0], BenchItem("q2", "mmlu", ITEMS[1].prompt, "A", 0.2))  # truth fixed later
    _, regraded = load_cached_records(corrected, ("claude-haiku-4-5",), cache)
    assert {r.item_id: r.correct for r in regraded}["q2"] is True
