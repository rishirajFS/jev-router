from pathlib import Path

from jev_router.dataset import load_prompts
from jev_router.questions import TASK_TYPES

DATA = Path(__file__).resolve().parent.parent / "data" / "probe_prompts.jsonl"


def test_dataset_loads_with_valid_labels():
    prompts = load_prompts(DATA)
    assert len(prompts) >= 60
    assert all(p.expected_tier in (0, 1, 2) for p in prompts)
    assert all(p.task_type in TASK_TYPES for p in prompts)
    assert len({p.id for p in prompts}) == len(prompts)


def test_dataset_is_balanced_across_tiers():
    prompts = load_prompts(DATA)
    for tier in (0, 1, 2):
        assert sum(p.expected_tier == tier for p in prompts) >= 15
