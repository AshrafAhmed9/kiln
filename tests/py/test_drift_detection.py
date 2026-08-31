"""Tests for drift detection (Phase 24). There is no real production
traffic to monitor, so every score stream here is a
synthetic, seeded numpy draw -- labeled honestly as synthetic, same as
Phase 21's prefix-cache workload. What's being verified is the statistics
themselves: that a genuine shift is flagged and a same-distribution
resample is not.
"""
import numpy as np
import pytest

from kiln_py.eval.drift_detection import detect_drift


def test_no_drift_when_both_batches_are_the_same_seeded_distribution():
    rng = np.random.default_rng(20260824)
    baseline = rng.normal(loc=0.80, scale=0.05, size=200).tolist()
    recent = rng.normal(loc=0.80, scale=0.05, size=200).tolist()

    report = detect_drift(baseline, recent, seed=1)

    assert not report.mean_shifted
    assert not report.shape_shifted


def test_mean_shift_is_flagged_when_recent_scores_are_genuinely_worse():
    rng = np.random.default_rng(20260824)
    baseline = rng.normal(loc=0.85, scale=0.03, size=200).tolist()
    recent = rng.normal(loc=0.55, scale=0.03, size=200).tolist()

    report = detect_drift(baseline, recent, seed=1)

    assert report.mean_shifted
    assert report.recent_mean < report.baseline_mean


def test_shape_shift_is_flagged_even_when_the_mean_barely_moves():
    # Same mean, very different spread -- a mean-only check would miss
    # this entirely, which is exactly why the KS statistic exists here.
    rng = np.random.default_rng(20260824)
    baseline = rng.normal(loc=0.70, scale=0.02, size=300).tolist()
    recent_tight = rng.normal(loc=0.70, scale=0.02, size=150).tolist()
    recent_wide = rng.normal(loc=0.70, scale=0.30, size=150).tolist()
    recent = recent_tight + recent_wide

    report = detect_drift(baseline, recent, seed=1)

    assert report.shape_shifted


def test_detect_drift_rejects_empty_batches():
    with pytest.raises(ValueError):
        detect_drift([], [0.5])
    with pytest.raises(ValueError):
        detect_drift([0.5], [])
