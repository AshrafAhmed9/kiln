"""Tests for LLM-as-a-judge (Phase 24). Uses a small scripted stand-in for
`judge_fn` instead of this project's real (untrained) model -- an
untrained model's judgments would be random noise, so what's testable
and meaningful here is the harness itself: prompt construction, response
parsing, and score aggregation. See docs/learning/phase-24.md.
"""
import pytest

from kiln_py.eval.llm_judge import (JudgeVerdict, build_judge_prompt, judge_answer,
                                     judge_batch, parse_judge_response,
                                     summarize_judge_scores)


def test_build_judge_prompt_contains_all_three_texts_and_the_rubric():
    prompt = build_judge_prompt("2+2?", "4", "four")
    assert "2+2?" in prompt
    assert "4" in prompt
    assert "four" in prompt
    assert "SCORE" in prompt
    assert "REASONING" in prompt


def test_parse_judge_response_reads_score_and_reasoning():
    verdict = parse_judge_response("SCORE: 4\nREASONING: Correct but less precise wording.")
    assert verdict.score == 4
    assert verdict.reasoning == "Correct but less precise wording."


def test_parse_judge_response_rejects_missing_score_instead_of_guessing():
    with pytest.raises(ValueError):
        parse_judge_response("This candidate answer looks pretty good to me.")


def test_parse_judge_response_is_case_insensitive_and_tolerates_extra_whitespace():
    verdict = parse_judge_response("score:   2  \nreasoning:   Missed the key detail.")
    assert verdict.score == 2
    assert verdict.reasoning == "Missed the key detail."


def test_judge_answer_wires_the_prompt_to_a_scripted_judge():
    def scripted_judge(prompt: str) -> str:
        assert "2+2?" in prompt
        return "SCORE: 5\nREASONING: Exact match."

    verdict = judge_answer(scripted_judge, "2+2?", "4", "4")
    assert verdict == JudgeVerdict(score=5, reasoning="Exact match.")


def test_judge_batch_scores_each_pair_independently():
    scripted_scores = iter([5, 1, 3])

    def scripted_judge(prompt: str) -> str:
        return f"SCORE: {next(scripted_scores)}\nREASONING: scripted."

    verdicts = judge_batch(
        scripted_judge,
        questions=["q1", "q2", "q3"],
        reference_answers=["a", "b", "c"],
        candidate_answers=["a", "z", "c"],
    )
    assert [v.score for v in verdicts] == [5, 1, 3]


def test_judge_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        judge_batch(lambda p: "SCORE: 3\nREASONING: x", ["q1", "q2"], ["a"], ["a"])


def test_summarize_judge_scores_reports_mean_and_a_real_confidence_interval():
    verdicts = [JudgeVerdict(score=s, reasoning="") for s in [5, 5, 5, 1, 1]]
    summary = summarize_judge_scores(verdicts, seed=0)
    assert summary["count"] == 5
    assert summary["mean_score"] == pytest.approx(3.4)
    low, high = summary["confidence_interval"]
    assert low <= summary["mean_score"] <= high


def test_summarize_judge_scores_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_judge_scores([])
