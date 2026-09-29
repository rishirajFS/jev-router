import json

import pandas as pd

from jev_router.build_samples import ids_of, write_ids, write_items
from jev_router.sampling import BenchItem


def items():
    return [BenchItem("b", "mmlu", "Q\nA) x", "A", 0.5), BenchItem("a", "mmlu-pro", "Q2\nA) y", "A", None)]


def test_ids_are_sorted_and_free_of_question_text(tmp_path):
    path = tmp_path / "ids.txt"
    write_ids(items(), path)
    assert path.read_text() == "a\nb\n"
    assert ids_of(items()) == ["a", "b"]


def test_write_items_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "s.jsonl"
    write_items(items(), path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["b", "a"]
    assert BenchItem(**rows[1]) == items()[1]
