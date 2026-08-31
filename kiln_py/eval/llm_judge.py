"""LLM-as-a-judge: instead of an exact-match or perplexity score, ask a
second model to read a question, a reference answer, and a candidate
answer, and rate how good the candidate is. Useful for open-ended answers
where there's no single correct string to match against.

This module owns the prompt format and the response parsing -- it never
picks which model does the judging. `judge_fn` is any callable from a
prompt string to that model's raw text reply, so the same harness works
whether the judge is a real hosted model or, in a test, a small scripted
stand-in. See docs/learning/phase-24.md for why judgment quality is a
property of whichever model backs `judge_fn`, not of this file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from kiln_py.eval.bootstrap import bootstrap_confidence_interval

_SCORE_LINE = re.compile(r"SCORE:\s*([1-5])", re.IGNORECASE)
_REASON_LINE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class JudgeVerdict:
    score: int  # 1 (worst) to 5 (best)
    reasoning: str


def build_judge_prompt(question: str, reference_answer: str, candidate_answer: str) -> str:
    """A fixed rubric, not an open-ended "rate this" prompt -- forcing a
    1-5 scale and a required SCORE/REASONING format is what makes the
    reply parseable at all, regardless of which model is judging.
    """
    return (
        "You are grading a candidate answer against a reference answer.\n"
        f"Question: {question}\n"
        f"Reference answer: {reference_answer}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        "Score the candidate from 1 to 5 for how well it answers the "
        "question compared to the reference (5 = equally good or better, "
        "1 = wrong or irrelevant). Reply in exactly this format:\n"
        "SCORE: <1-5>\n"
        "REASONING: <one sentence>"
    )


def parse_judge_response(text: str) -> JudgeVerdict:
    """Raises ValueError on anything that doesn't match the required
    format, rather than guessing a default score -- a judge reply that
    can't be parsed is a real problem to surface, not one to paper over
    with a fallback number that would quietly corrupt every aggregate
    built on top of it.
    """
    score_match = _SCORE_LINE.search(text)
    if not score_match:
        raise ValueError(f"judge response had no parseable SCORE line: {text!r}")
    reason_match = _REASON_LINE.search(text)
    reasoning = reason_match.group(1).strip() if reason_match else ""
    return JudgeVerdict(score=int(score_match.group(1)), reasoning=reasoning)


def judge_answer(judge_fn: Callable[[str], str], question: str,
                  reference_answer: str, candidate_answer: str) -> JudgeVerdict:
    prompt = build_judge_prompt(question, reference_answer, candidate_answer)
    response = judge_fn(prompt)
    return parse_judge_response(response)


def judge_batch(judge_fn: Callable[[str], str], questions: list[str],
                 reference_answers: list[str], candidate_answers: list[str]) -> list[JudgeVerdict]:
    if not (len(questions) == len(reference_answers) == len(candidate_answers)):
        raise ValueError("questions, reference_answers, and candidate_answers must be the same length")
    return [
        judge_answer(judge_fn, q, ref, cand)
        for q, ref, cand in zip(questions, reference_answers, candidate_answers)
    ]


def summarize_judge_scores(verdicts: list[JudgeVerdict], confidence: float = 0.95,
                            seed: int = 0) -> dict:
    """Reuses the same bootstrap machinery Phase 13's regression gate
    already relies on -- an LLM judge's scores are noisy in exactly the
    same way a task's per-question scores are, so the same "how much
    would this wobble on a different sample" question applies here too.
    """
    scores = [float(v.score) for v in verdicts]
    if not scores:
        raise ValueError("summarize_judge_scores needs at least one verdict")
    mean_score = sum(scores) / len(scores)
    low, high = bootstrap_confidence_interval(scores, confidence=confidence, seed=seed)
    return {
        "mean_score": mean_score,
        "confidence_interval": (low, high),
        "count": len(scores),
    }
