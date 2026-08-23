"""Decides whether a candidate model/version is a statistically real
regression against a baseline, using paired bootstrap confidence intervals
on the per-question score differences -- not just comparing two averages.
See docs/learning/phase-13.md for why the averages alone aren't enough.
"""
from __future__ import annotations

from dataclasses import dataclass

from kiln_py.eval.bootstrap import bootstrap_confidence_interval


@dataclass
class RegressionCheckResult:
    mean_difference: float  # candidate score minus baseline score, averaged
    confidence_interval: tuple[float, float]
    is_regression: bool  # True only if the CI is entirely below zero


def check_for_regression(baseline_scores: list[float], candidate_scores: list[float],
                          confidence: float = 0.95, seed: int = 0) -> RegressionCheckResult:
    """baseline_scores and candidate_scores must be PAIRED -- entry i in
    each list must be the same question's score under the two versions.
    Pairing like this (rather than bootstrapping the two score lists
    independently) is what lets easy and hard questions cancel out
    correctly, instead of adding noise from "this batch happened to have
    harder questions" into the comparison.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("baseline and candidate scores must be paired (same length)")

    differences = [c - b for b, c in zip(baseline_scores, candidate_scores)]
    mean_difference = sum(differences) / len(differences)
    low, high = bootstrap_confidence_interval(differences, confidence=confidence,
                                               seed=seed)

    # Only call it a regression if we're confident the candidate is worse
    # -- meaning even the most optimistic end of the confidence interval
    # still shows a drop. If the interval straddles zero, the honest
    # answer is "we don't have enough evidence either way," not "it's
    # fine" and not "it's broken."
    is_regression = high < 0.0

    return RegressionCheckResult(mean_difference=mean_difference,
                                  confidence_interval=(low, high),
                                  is_regression=is_regression)
