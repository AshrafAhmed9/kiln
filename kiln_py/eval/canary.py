"""Canary comparison: replays a recorded log of real prompts through two
model/config versions and reports which ones produced a different answer,
so a human can review exactly what changed rather than trusting a single
aggregate score to catch every kind of regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CanaryDiff:
    prompt: str
    baseline_output: str
    candidate_output: str
    changed: bool


def replay_canary(prompts: list[str], baseline_fn: Callable[[str], str],
                   candidate_fn: Callable[[str], str]) -> list[CanaryDiff]:
    """Runs every prompt through both `baseline_fn` and `candidate_fn`
    (each a function from prompt to generated text) and reports, for each
    one, whether the two versions actually disagreed.
    """
    diffs = []
    for prompt in prompts:
        baseline_output = baseline_fn(prompt)
        candidate_output = candidate_fn(prompt)
        diffs.append(CanaryDiff(
            prompt=prompt,
            baseline_output=baseline_output,
            candidate_output=candidate_output,
            changed=(baseline_output != candidate_output),
        ))
    return diffs


def summarize_canary(diffs: list[CanaryDiff]) -> dict:
    """A quick top-line summary: how many prompts changed, out of how
    many total -- the first thing a human reviewing a canary run actually
    wants to know before reading every individual diff.
    """
    changed_count = sum(1 for d in diffs if d.changed)
    return {
        "total": len(diffs),
        "changed": changed_count,
        "unchanged": len(diffs) - changed_count,
    }
