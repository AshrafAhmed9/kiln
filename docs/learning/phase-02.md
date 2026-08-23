# Phase 2 — derivation notes (the forward pass)

## RMSNorm — why no mean subtraction

A normal LayerNorm centers a row of numbers around zero before scaling it
(subtract the mean, then divide by the standard deviation). RMSNorm skips
the centering step entirely: it only divides by the row's root-mean-square
(a measure of the row's overall size), then multiplies by a learned weight.
Llama-family models use RMSNorm because it's cheaper (no mean to compute)
and, empirically, dropping the mean-subtraction doesn't hurt quality for
these models. The formula, in words: take every number in the row, square
it, average those squares, take the square root of that average (adding a
tiny epsilon so we never divide by exactly zero) -- that's the row's
"typical size." Divide every original number by that typical size, then
multiply by the matching learned weight.

## RoPE — why rotation encodes position

The model needs to know how far apart two words are, but it only ever sees
each word's numbers, not their positions directly. RoPE's trick: instead of
adding a "position number" to each word (which is what older models did),
it rotates each word's numbers by an angle that depends on that word's
position. Two words at different positions end up rotated by different
amounts, and when attention later compares two words (via a dot product),
the *difference* in their rotation angles falls out of that comparison
automatically -- so attention naturally knows the relative distance between
any two words, without ever being told a position directly.

Numbers within one "head" are rotated in pairs. Llama pairs a number with
the one located exactly half the head's width away from it (not simply its
immediate neighbor) -- get this pairing wrong and the model still runs, it
just silently disagrees with the reference, which is exactly the kind of
bug the parity harness exists to catch.

## Grouped-query attention — the memory-saving trick

Normally, every attention "head" has its own set of keys and values to
compare against. Grouped-query attention has several heads share the same
keys and values instead -- head number h simply looks up shared key/value
head number (h divided by the group size, rounded down). Fewer sets of
keys and values means a much smaller cache to store, at a small cost to how
precisely attention can focus.

## What I'd get wrong without this derivation

I would have been tempted to pair RoPE's rotated numbers with their
immediate neighbor instead of the "half the head away" pairing Llama
actually uses -- a bug that runs fine and produces plausible-looking
numbers, but silently disagrees with the real model.
