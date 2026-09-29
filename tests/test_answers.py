import pytest

from jev_router.answers import consensus, extract_letter, extract_number, parse_listified


def test_parse_listified_unwraps_single_string_lists():
    assert parse_listified("['hello world']") == "hello world"
    assert parse_listified('["it\'s fine"]') == "it's fine"
    assert parse_listified("plain text") == "plain text"


def test_extract_letter_handles_common_formats():
    assert extract_letter("B", "ABCD") == "B"
    assert extract_letter("A)", "ABCD") == "A"
    assert extract_letter("  (c) ", "ABCD") == "C"
    assert extract_letter("Answer: D", "ABCD") == "D"
    assert extract_letter("The answer is B) because...", "ABCD") == "B"


def test_extract_letter_rejects_out_of_range_or_missing():
    assert extract_letter("E", "ABCD") is None
    assert extract_letter("I do not know", "ABCD") is None
    assert extract_letter("", "AB") is None


def test_extract_number_takes_last_number_and_normalizes():
    assert extract_number("so 12 + 30 = 42 apples") == "42"
    assert extract_number("The answer is $1,250.00") == "1250"
    assert extract_number("Answer: 3.50") == "3.5"
    assert extract_number("-7 degrees") == "-7"
    assert extract_number("no digits here") is None


def test_consensus_requires_agreement_and_minimum_votes():
    assert consensus(["B", "B", "B"], min_votes=2) == "B"
    assert consensus(["B"], min_votes=2) is None
    assert consensus(["B", "C"], min_votes=2) is None
    assert consensus(["B", "B", "C"], min_votes=2) is None
    assert consensus([None, "B", "B"], min_votes=2) == "B"
    assert consensus([], min_votes=1) is None


def test_consensus_rejects_bad_min_votes():
    with pytest.raises(ValueError):
        consensus(["A"], min_votes=0)


def test_extract_letter_reads_the_final_answer_of_a_verbose_response():
    v = "ABCDEFGHIJ"
    assert extract_letter("10 - 2k = 0.5(5+k)\n\nso k = 3.\n\n**D**", v) == "D"
    assert extract_letter("Reasoning here.\nAnswer: **B**", v) == "B"
    assert extract_letter("So the answer is (C).", v) == "C"
    assert extract_letter("The closest choice is F, not G.", v) == "F"
    assert extract_letter("Some reasoning\nwith two lines\nH", v) == "H"


def test_extract_letter_takes_the_last_explicit_answer():
    assert extract_letter("Answer: A ... on reflection the answer is C", "ABCD") == "C"


def test_extract_letter_does_not_mistake_prose_for_an_answer():
    assert extract_letter("The answer is a strategy that works.", "ABCD") is None
    assert extract_letter("I think the pronoun I is used here.", "ABCD") is None
    assert extract_letter("A man walked in and the reasoning is unclear.", "ABCD") is None
