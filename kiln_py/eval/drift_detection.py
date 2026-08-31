"""Drift detection: given a baseline batch of eval scores and a more
recent batch, decide whether the recent scores actually come from a
different distribution or whether the difference is just sampling noise.

There is no live production traffic to monitor here (see HANDOFF.md) --
this module is exercised with a synthetic, seeded stream of scores in its
tests, labeled honestly as synthetic. What's real is the statistics: the
same paired-bootstrap idea Phase 13's regression gate already uses, plus
a two-sample Kolmogorov-Smirnov statistic for catching a shape change
(more spread, a new mode) that a mean-only check would miss entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kiln_py.eval.bootstrap import bootstrap_confidence_interval


@dataclass
class DriftReport:
    baseline_mean: float
    recent_mean: float
    baseline_ci: tuple[float, float]
    recent_ci: tuple[float, float]
    ks_statistic: float
    mean_shifted: bool  # the two confidence intervals don't overlap at all
    shape_shifted: bool  # ks_statistic exceeds the asymptotic no-drift bound


def _ks_statistic(baseline: np.ndarray, recent: np.ndarray) -> float:
    """The two-sample KS statistic: the largest gap between the two
    samples' empirical CDFs, evaluated at every point either sample
    actually has a value. A large gap means the two samples' shapes
    disagree somewhere, not just their averages.
    """
    all_values = np.concatenate([baseline, recent])
    baseline_sorted = np.sort(baseline)
    recent_sorted = np.sort(recent)
    baseline_cdf = np.searchsorted(baseline_sorted, all_values, side="right") / len(baseline)
    recent_cdf = np.searchsorted(recent_sorted, all_values, side="right") / len(recent)
    return float(np.max(np.abs(baseline_cdf - recent_cdf)))


def _ks_no_drift_bound(n_baseline: int, n_recent: int, confidence: float = 0.95) -> float:
    """The standard asymptotic critical value for the two-sample KS test:
    above this, the two samples are unlikely (at the given confidence) to
    come from the same distribution by chance alone.
    """
    c_alpha = np.sqrt(-0.5 * np.log((1.0 - confidence) / 2.0))
    return c_alpha * np.sqrt((n_baseline + n_recent) / (n_baseline * n_recent))


def detect_drift(baseline_scores: list[float], recent_scores: list[float],
                  confidence: float = 0.95, seed: int = 0) -> DriftReport:
    if not baseline_scores or not recent_scores:
        raise ValueError("detect_drift needs at least one score in each batch")

    baseline_array = np.array(baseline_scores)
    recent_array = np.array(recent_scores)

    baseline_mean = float(np.mean(baseline_array))
    recent_mean = float(np.mean(recent_array))
    baseline_ci = bootstrap_confidence_interval(baseline_scores, confidence=confidence, seed=seed)
    recent_ci = bootstrap_confidence_interval(recent_scores, confidence=confidence, seed=seed + 1)

    # Non-overlapping confidence intervals is the same "even the most
    # cautious read still shows a real difference" bar Phase 13 uses --
    # applied here to two independent samples instead of paired ones.
    mean_shifted = (recent_ci[0] > baseline_ci[1]) or (baseline_ci[0] > recent_ci[1])

    ks_statistic = _ks_statistic(baseline_array, recent_array)
    ks_bound = _ks_no_drift_bound(len(baseline_scores), len(recent_scores), confidence)
    shape_shifted = ks_statistic > ks_bound

    return DriftReport(
        baseline_mean=baseline_mean,
        recent_mean=recent_mean,
        baseline_ci=baseline_ci,
        recent_ci=recent_ci,
        ks_statistic=ks_statistic,
        mean_shifted=mean_shifted,
        shape_shifted=shape_shifted,
    )
