import pandas as pd

from jev_router.mmlu_pro import build_mmlu_pro_sample, format_prompt


def make_frame():
    rows = []
    for cat in ("math", "law", "biology"):
        for i in range(20):
            rows.append(
                {
                    "question_id": f"{cat}{i}",
                    "question": f"Question {cat} {i}?",
                    "options": ["opt a", "opt b", "opt c", "opt d", "opt e"],
                    "answer": "C",
                    "category": cat,
                }
            )
    return pd.DataFrame(rows)


def test_format_prompt_lists_lettered_options_and_instruction():
    text = format_prompt("What is 2+2?", ["3", "4", "5"])
    assert text.startswith("What is 2+2?\nA) 3\nB) 4\nC) 5\n")
    assert '"A" or "B" or "C"' in text
    assert text.rstrip().endswith("Answer:")


def test_build_sample_balances_categories_and_sets_truth():
    items = build_mmlu_pro_sample(make_frame(), n=9, seed=1)
    assert len(items) == 9
    assert {i.family for i in items} == {"mmlu-pro"}
    assert all(i.truth == "C" and i.old_pass_rate is None for i in items)
    cats = [i.id.split(".")[1].rstrip("0123456789") for i in items]
    assert {c: cats.count(c) for c in set(cats)} == {"math": 3, "law": 3, "biology": 3}


def test_build_sample_is_deterministic():
    a = build_mmlu_pro_sample(make_frame(), n=9, seed=5)
    b = build_mmlu_pro_sample(make_frame(), n=9, seed=5)
    assert [i.id for i in a] == [i.id for i in b]
