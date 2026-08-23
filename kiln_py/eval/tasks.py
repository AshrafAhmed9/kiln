"""Task scoring. Kept deliberately separate from anything that calls a real
model: the scoring math (given a predicted answer and an expected answer,
did it match?) is tested on its own, independent of whether a model is
any good -- which matters here, since the only model available in this
offline environment is untrained and random, and would otherwise be the
only thing standing between "the eval math is right" and "the test
passes."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def normalize_answer(text: str) -> str:
    """Loosens an exact-match comparison just enough to not be defeated by
    trivial formatting differences (case, surrounding whitespace) that
    have nothing to do with whether the answer is actually right.
    """
    return text.strip().lower()


def exact_match_score(predictions: list[str], expected: list[str]) -> float:
    """The fraction of predictions that match their expected answer, after
    normalizing both. Returns 0.0 for an empty input rather than dividing
    by zero -- an empty task suite trivially "scores" nothing, which is a
    more honest answer than crashing or claiming a perfect score.
    """
    if not predictions:
        return 0.0
    matches = sum(
        1 for p, e in zip(predictions, expected)
        if normalize_answer(p) == normalize_answer(e)
    )
    return matches / len(predictions)


@dataclass
class QaPair:
    prompt: str
    answer: str


def run_exact_match_task(generate_fn: Callable[[str], str],
                          qa_pairs: list[QaPair]) -> float:
    """Runs `generate_fn` (a function that takes a prompt and returns the
    model's generated text -- typically a thin wrapper around
    kiln_py.runtime.generate.generate) over every question, and scores the
    results with exact_match_score above.
    """
    predictions = [generate_fn(pair.prompt) for pair in qa_pairs]
    expected = [pair.answer for pair in qa_pairs]
    return exact_match_score(predictions, expected)
