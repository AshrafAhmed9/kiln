"""Perplexity: a standard score for how well a model predicts real text.
Lower is better. Roughly, it is "how surprised was the model by each real
next word, on average".
"""
from __future__ import annotations

import math

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Turns raw scores into log-probabilities."""
    shifted = logits - np.max(logits)  # subtract the max so exp() cannot overflow
    return shifted - np.log(np.sum(np.exp(shifted)))


def perplexity_from_logits(all_logits: np.ndarray, token_ids: list[int]) -> float:
    """all_logits is [seq_len, vocab_size]: row i is the model's prediction
    for the word after token_ids[i]. Each prediction is checked against the
    word that really came next. The last row is skipped, since nothing
    follows it.

    For each position: how much probability did the model give the real next
    word? A model that was sure and right gives it close to 1 (little
    surprise). A model that was sure and wrong gives it close to 0 (huge
    surprise). Average the surprise (-log probability) over all positions,
    then undo the log with exp().
    """
    num_predictions = len(token_ids) - 1
    if num_predictions <= 0:
        raise ValueError("perplexity needs at least 2 tokens to score anything")

    total_surprise = 0.0
    for i in range(num_predictions):
        log_probs = _log_softmax(all_logits[i])
        total_surprise += -log_probs[token_ids[i + 1]]

    return math.exp(total_surprise / num_predictions)
