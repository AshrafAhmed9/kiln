# Phase 13 — derivation notes (evaluation infrastructure)

## The problem this solves

Every earlier phase asked "does this one number match?" (parity against a
reference). This phase asks a different, longer-horizon question: "did
changing the model or its configuration make it *worse* overall?" That
needs running many test questions through the model and summarizing the
results, not comparing single numbers.

## Why regression checks need a confidence interval, not just an average

If version A scores 61% on a task and version B scores 59%, is B actually
worse, or did it just happen to get a couple of harder questions this
time? With only a few dozen or few hundred test questions, a small
difference can easily be random noise. Bootstrapping answers this
honestly: resample the *same* set of answered questions many times (with
replacement), recompute the score each time, and look at the *spread* of
those resampled scores. If two versions' spreads barely overlap, the
difference is probably real; if they overlap heavily, it probably isn't
— and this project's regression gate is built to say "not statistically
significant" rather than quietly treat noise as a real regression.

## Why canary comparison replays real traffic instead of a fixed test set

A fixed benchmark only tells you about the specific questions it contains.
Replaying a recorded log of real requests through two versions and
comparing their answers and latencies side by side catches problems a
curated benchmark might never think to ask about — this is the same
"test against real recorded behavior" idea a production incident replay
uses, applied to model evaluation instead of a live outage.

## Honest scope

A real MMLU-subset evaluation needs real MMLU data and a real trained
model; neither is available offline in this session. What's implemented
here is the machinery itself (task running, scoring, bootstrap confidence
intervals, canary replay and diffing) exercised against small, synthetic,
hand-built task suites and the toy random model already used throughout
this project — proving the *infrastructure* works, not that any particular
real model scores well on any particular real benchmark.
