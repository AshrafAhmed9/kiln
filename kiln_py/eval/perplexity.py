"""Perplexity: a standard way of scoring how well a model predicts real
text, lower being better. Loosely, it's "on average, how surprised was the
model by each actual next word?" -- turned into a single number so
different models or versions can be compared.
"""
from __future__ import annotations

import math

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum_exp


def perplexity_from_logits(all_logits: np.ndarray, token_ids: list[int]) -> float:
    """all_logits is [seq_len, vocab_size] -- row i is the model's
    prediction for what comes right after token_ids[i]. This scores how
    well those predictions matched what actually came next (token_ids[i+1])
    for every position but the last, since there's nothing "next" to check
    the last position's prediction against.

    The core idea: for each real next word, look up how much probability
    the model's own prediction assigned to that specific word. A model
    that's confidently right assigns that word a probability close to 1
    (a small "surprise," measured as -log(probability)); a model that's
    confidently wrong assigns it a probability close to 0 (a huge
    surprise). Averaging that surprise across every position, then
    undoing the logarithm at the very end (exponentiating), turns the
    average surprise into perplexity.
    """
    num_predictions = len(token_ids) - 1
    if num_predictions <= 0:
        raise ValueError("perplexity needs at least 2 tokens to score anything")

    total_negative_log_likelihood = 0.0
    for i in range(num_predictions):
        log_probs = _log_softmax(all_logits[i])
        actual_next_token = token_ids[i + 1]
        total_negative_log_likelihood += -log_probs[actual_next_token]

    average_negative_log_likelihood = total_negative_log_likelihood / num_predictions
    return math.exp(average_negative_log_likelihood)
