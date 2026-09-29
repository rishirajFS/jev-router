"""Answer parsing and consensus for recovering RouterBench ground truth."""

import ast
import re
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_listified(raw: str) -> str:
    """RouterBench stores prompts and responses as the repr of a one-item list."""
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return raw
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            return value[0]
    return raw


_BARE = re.compile(r"\(?([A-Za-z])\)?[.):]?")
_EXPLICIT = re.compile(r"(?i:answer|choice|option)\s*(?:is|:)?\s*[:\-]?\s*\**\(?([A-Z])\)?\**(?![A-Za-z])")
_BOLD = re.compile(r"\*\*\(?([A-Z])[.):]?(?=[\s*])")
_LEADING = re.compile(r"\(?([A-Za-z])\)")


def _last_valid(pattern: re.Pattern[str], text: str, valid: str) -> str | None:
    found = [m for m in pattern.findall(text) if m in valid]
    return found[-1] if found else None


def extract_letter(text: str, valid: str) -> str | None:
    """Final multiple-choice letter of a response, whether bare or buried in an explanation."""
    stripped = text.strip()
    if not stripped:
        return None
    bare = _BARE.fullmatch(stripped)
    if bare:
        letter = bare.group(1).upper()
        return letter if letter in valid else None
    # Explicit statements and bold letters require an uppercase letter, so prose such as
    # "the answer is a strategy" or the pronoun "I" is never read as a choice.
    for pattern in (_EXPLICIT, _BOLD):
        letter = _last_valid(pattern, stripped, valid)
        if letter:
            return letter
    last_line = re.sub(r"[*_`]", "", stripped.splitlines()[-1]).strip()
    tail = re.fullmatch(r"\(?([A-Z])\)?[.):]?", last_line)
    if tail and tail.group(1) in valid:
        return tail.group(1)
    leading = _LEADING.match(stripped)
    if leading and leading.group(1).upper() in valid:
        return leading.group(1).upper()
    return None


def extract_number(text: str) -> str | None:
    found = _NUMBER.findall(text)
    if not found:
        return None
    cleaned = found[-1].replace(",", "").rstrip(".")
    try:
        return format(Decimal(cleaned).normalize(), "f")
    except InvalidOperation:
        return None


def consensus(answers: Iterable[str | None], min_votes: int = 2) -> str | None:
    if min_votes < 1:
        raise ValueError("min_votes must be at least 1")
    votes = Counter(a for a in answers if a is not None)
    if sum(votes.values()) < min_votes or len(votes) != 1:
        return None
    return next(iter(votes))
