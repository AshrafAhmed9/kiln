"""Tests for the eval infrastructure (Phase 13). Deliberately independent
of the toy random model wherever possible -- this project's only available
model is untrained, so testing "did the eval math get the right answer"
against known, hand-computed inputs is more meaningful than testing "did
the real model score well" against a model that has no real skill to
measure.
"""
import math

import numpy as np
import pytest

from kiln_py.eval.bootstrap import bootstrap_confidence_interval
from kiln_py.eval.canary import replay_canary, summarize_canary
from kiln_py.eval.perplexity import perplexity_from_logits
from kiln_py.eval.regression_gate import check_for_regression
from kiln_py.eval.tasks import QaPair, exact_match_score, run_exact_match_task


def test_exact_match_score_counts_normalized_matches():
    predictions = ["Paris", " paris ", "London", "berlin"]
    expected = ["paris", "Paris", "Paris", "Berlin"]
    # position 0: "Paris" vs "paris" -> match after normalizing
    # position 1: " paris " vs "Paris" -> match after normalizing
    # position 2: "London" vs "Paris" -> no match
    # position 3: "berlin" vs "Berlin" -> match after normalizing
    assert exact_match_score(predictions, expected) == 3 / 4


def test_exact_match_score_on_empty_input_is_zero_not_a_crash():
    assert exact_match_score([], []) == 0.0


def test_run_exact_match_task_wires_generate_fn_to_scoring():
    # A stub "model" that always gets the first question right and the
    # second one wrong -- isolates the task-running wiring from any real
    # model behavior.
    def fake_generate(prompt: str) -> str:
        return "yes" if prompt == "is the sky blue?" else "wrong answer"

    qa_pairs = [
        QaPair(prompt="is the sky blue?", answer="yes"),
        QaPair(prompt="is grass purple?", answer="no"),
    ]
    assert run_exact_match_task(fake_generate, qa_pairs) == 0.5


def test_perplexity_matches_hand_computed_value_for_a_confident_correct_model():
    # A tiny, fully controlled 2-token vocabulary, 2 positions. The model's
    # logits put ALL of the probability mass on the correct next token at
    # every position, so the model should be completely unsurprised --
    # perplexity of exactly 1.0 is the mathematical minimum, representing
    # "always exactly right."
    logits = np.array([
        [100.0, -100.0],  # extremely confident position 0 predicts token 0
        [-100.0, 100.0],  # extremely confident position 1 predicts token 1
    ])
    token_ids = [0, 0, 1]  # first token doesn't matter; positions 0 and 1 verify predictions
    result = perplexity_from_logits(logits, token_ids)
    assert math.isclose(result, 1.0, abs_tol=1e-6)


def test_perplexity_is_higher_for_a_confidently_wrong_model():
    correct_logits = np.array([[100.0, -100.0]])
    wrong_logits = np.array([[-100.0, 100.0]])
    token_ids = [0, 0]  # the real next token is 0

    correct_perplexity = perplexity_from_logits(correct_logits, token_ids)
    wrong_perplexity = perplexity_from_logits(wrong_logits, token_ids)
    assert wrong_perplexity > correct_perplexity


def test_perplexity_requires_at_least_two_tokens():
    with pytest.raises(ValueError):
        perplexity_from_logits(np.array([[1.0, 2.0]]), [0])


def test_bootstrap_ci_is_a_single_point_when_every_value_is_identical():
    low, high = bootstrap_confidence_interval([0.5, 0.5, 0.5, 0.5], seed=1)
    assert math.isclose(low, 0.5, abs_tol=1e-9)
    assert math.isclose(high, 0.5, abs_tol=1e-9)


def test_bootstrap_ci_widens_with_more_variance():
    low_variance_ci = bootstrap_confidence_interval([0.5, 0.51, 0.49, 0.5], seed=2)
    high_variance_ci = bootstrap_confidence_interval([0.0, 1.0, 0.0, 1.0], seed=2)
    low_width = low_variance_ci[1] - low_variance_ci[0]
    high_width = high_variance_ci[1] - high_variance_ci[0]
    assert high_width > low_width


def test_regression_gate_flags_a_consistently_worse_candidate():
    baseline = [0.9, 0.9, 0.9, 0.9, 0.9]
    candidate = [0.5, 0.5, 0.5, 0.5, 0.5]  # worse on every single paired question
    result = check_for_regression(baseline, candidate, seed=3)
    assert result.is_regression


def test_regression_gate_does_not_flag_a_consistently_better_candidate():
    baseline = [0.5, 0.5, 0.5, 0.5, 0.5]
    candidate = [0.9, 0.9, 0.9, 0.9, 0.9]
    result = check_for_regression(baseline, candidate, seed=4)
    assert not result.is_regression


def test_regression_gate_does_not_flag_noisy_inconclusive_data():
    # Roughly as many questions got better as got worse -- there's no
    # confident signal either way, so this must NOT be flagged as a
    # regression just because a few individual questions got worse.
    baseline = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    candidate = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
    result = check_for_regression(baseline, candidate, seed=5)
    assert not result.is_regression


def test_regression_gate_requires_paired_equal_length_lists():
    with pytest.raises(ValueError):
        check_for_regression([0.5, 0.5], [0.5])


def test_canary_replay_detects_changed_and_unchanged_prompts():
    def baseline_fn(prompt: str) -> str:
        return f"baseline:{prompt}"

    def candidate_fn(prompt: str) -> str:
        # Deliberately answers one prompt differently from the baseline.
        if prompt == "prompt-2":
            return "a different answer"
        return f"baseline:{prompt}"

    prompts = ["prompt-1", "prompt-2", "prompt-3"]
    diffs = replay_canary(prompts, baseline_fn, candidate_fn)
    summary = summarize_canary(diffs)

    assert summary == {"total": 3, "changed": 1, "unchanged": 2}
    changed_prompts = [d.prompt for d in diffs if d.changed]
    assert changed_prompts == ["prompt-2"]
