# Phase 9 — derivation notes (quantization)

## What quantization actually is

A trained model's weights are stored as 32-bit (or 16-bit) floating-point
numbers, which can represent an enormous range of values very precisely.
Quantization notices that most of a weight matrix's actual values cluster
in a much narrower range, and trades some of that unused precision for a
much smaller number: instead of storing every weight as its own float,
store one small "scale" number per group of weights, plus one small
integer per weight that says "how many scale-units away from zero is this
weight." Reconstructing an approximate version of the original weight is
just `integer * scale`. Fewer bits per weight means less memory and, on
real hardware, faster math -- at the cost of some accuracy, which is why
the plan insists this cost always be measured, never assumed.

## Why the scale is chosen per group, not once for the whole matrix

If one single scale had to cover an entire weight matrix, a single unusually
large weight anywhere in that matrix would force every other, much smaller
weight to be represented with very coarse steps -- most of the matrix's
precision would be wasted protecting against one outlier. Computing a
separate scale for each smaller group of weights (an output channel, or an
even smaller fixed-size group of consecutive weights within a channel)
keeps each group's own outliers from ruining precision for the rest of the
matrix. This is exactly why real weight quantization schemes (this
project's per-channel INT8, and group-wise INT4) bother with per-group
scales instead of one number for everything.

## Why this project stores INT4 values two-to-a-byte

The entire point of using 4-bit numbers instead of 8-bit ones is to use
half the memory. If each 4-bit value were still stored in its own whole
byte, the memory savings this technique promises simply wouldn't exist --
it would just be an 8-bit scheme that throws away precision for nothing in
return. So two 4-bit values are packed into the two halves ("nibbles") of
one byte, and unpacked back out at read time. To keep the packing and
unpacking simple to read and simple to get right, every 4-bit value is
shifted up by 8 before storing (so it's always a small, non-negative
number that fits in 4 bits cleanly) and shifted back down by 8 when read
back out.

## Honest limitation of this implementation

This project's quantizer is round-to-nearest: for each group, take the
weights, find the biggest one, size the scale to it, and round every
weight in the group to the nearest representable step. Real production
schemes like GPTQ go further -- they choose each weight's rounded value
by also accounting for how much that particular rounding error will hurt
the *model's actual output*, not just how close the rounded number is to
the original weight. That's a meaningfully more sophisticated (and
effective) approach, and it's explicitly out of scope here: implementing
plain round-to-nearest is what teaches the core quantization idea; the
smarter, output-aware version is a research-grade technique on top of it,
named honestly as a real gap rather than quietly assumed to be included.
