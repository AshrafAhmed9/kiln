# What quantization actually costs — and what this project can and can't yet say about it

A trained model's weights are 32-bit floating point numbers by default —
precise, and large. Quantization notices that most weights in practice
cluster in a narrow range, and trades away precision most of them never
needed for a much smaller number: one small "scale" per group of weights,
plus a small integer per weight saying how many scale-steps away from
zero it sits. Reconstructing an approximate original is `integer *
scale`. Less memory, and on real hardware, faster math — at a real,
measurable accuracy cost that responsible engineering publishes rather
than assumes away.

## Two schemes, and the design choice that connects them

This project implements weight-only INT8 (one scale per output row) and
INT4 (one scale per smaller group of weights within a row, packed two
values to a byte, since the entire point of using 4-bit numbers is using
half the memory 8-bit numbers would). The shared design idea underneath
both: **the scale is computed per group, not once for the whole matrix**,
specifically so one unusually large weight anywhere in a matrix doesn't
force every other, much smaller weight in that same row to be represented
with coarse, wasteful steps. INT4's smaller groups exist because 4-bit
numbers have far fewer representable steps (15, versus INT8's 255) — so
outliers do proportionally more damage unless the protection against them
is finer-grained too.

## The honest limitation, stated plainly rather than glossed over

This project's quantizer is round-to-nearest: for each group, find the
largest weight, size the scale to it, round every weight in the group to
its nearest representable step. Production-grade schemes like GPTQ go
further — they choose each weight's rounded value partly based on how much
that specific rounding error will actually hurt the *model's real
output*, not just how close the rounded number sits to the original
weight. That's a genuinely more effective technique, and it's explicitly
not implemented here. Round-to-nearest is what teaches the core idea
cleanly; the smarter, output-aware version is real, additional
sophistication layered on top of it, named as a gap rather than quietly
assumed to be included.

## What this project can actually claim right now, and what it can't yet

**Can claim:** the quantization math is implemented correctly. The
round-trip error (quantize, then reconstruct) stays within the
mathematically expected bound (roughly half a representable step) for
both schemes. A synthetic test — quantized weights used in a real matrix
multiplication, compared against the same multiplication in full
precision — confirms the mechanism doesn't silently produce wildly wrong
output. The INT8 implementation is cross-checked against an independently
written Python reference computing the exact same thing a different way,
and the two agree exactly.

**Cannot yet claim:** a real accuracy number. The actual deliverable this
kind of work is supposed to produce — perplexity measured on a real
benchmark like WikiText-2, using a real trained checkpoint, at each
quantization level — needs both a real model and a real evaluation
dataset, neither of which is available in the offline, single-machine
environment this project was built in. What's published here instead is
the honest boundary of what's actually been measured: the mechanism
works; its real-world cost, on a real model, is a genuinely open,
explicitly deferred question — one this project's own eval infrastructure
(Phase 13: perplexity scoring, paired regression gating) is already built
to answer, the moment a real checkpoint is available to point it at.

This is the same discipline the rest of the project holds itself to:
state exactly what's been proven, exactly how, and exactly where the
proof stops — rather than let a working demo imply more than it actually
shows.
