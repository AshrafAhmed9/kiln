"""Bootstrap confidence intervals: a way to ask "how much would this score
have wobbled if we'd happened to test on a slightly different sample of
the same size?" without needing to collect more data. See
docs/learning/phase-13.md for why this matters for regression gating.
"""
from __future__ import annotations

import numpy as np


def bootstrap_confidence_interval(values: list[float], num_resamples: int = 1000,
                                   confidence: float = 0.95,
                                   seed: int = 0) -> tuple[float, float]:
    """Resamples `values` (with replacement, same size each time)
    `num_resamples` times, computes the mean of each resample, and returns
    the range that the middle `confidence` fraction of those resampled
    means falls into. A narrow range means the original score is a
    reliable estimate; a wide range means it easily could have come out
    quite differently by chance.
    """
    if not values:
        raise ValueError("bootstrap needs at least one value to resample")

    rng = np.random.default_rng(seed)
    values_array = np.array(values)
    resampled_means = np.empty(num_resamples)

    for i in range(num_resamples):
        resample = rng.choice(values_array, size=len(values_array), replace=True)
        resampled_means[i] = np.mean(resample)

    lower_percentile = (1.0 - confidence) / 2.0 * 100
    upper_percentile = 100 - lower_percentile
    low = float(np.percentile(resampled_means, lower_percentile))
    high = float(np.percentile(resampled_means, upper_percentile))
    return low, high
