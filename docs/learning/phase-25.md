# Phase 25 — ragged prefill: batching by real work, not by a padded rectangle

## The problem padding actually causes

Phase 4's padded batching runs several prompts together by making them all
the same length -- the shortest ones get filler tokens up to the longest
one's length, and the attention mask hides the filler from ever mattering
to the answer. That's correct, but it's not free: every filler token still
runs through every matmul in every layer. Three prompts of length 20, 5,
and 3 padded to a rectangle cost 3×20 = 60 rows of real matrix work to
serve 28 real tokens -- more than double the actual work, for prompts this
unevenly sized.

## The idea

Concatenate every sequence's real tokens back-to-back, with no filler at
all, and track only where each one starts and how long it is. Every matmul
in every layer (the embedding lookup, RMSNorm, the Q/K/V/output
projections, the RoPE rotation, the SwiGLU feed-forward) treats a token as
just a row in a matrix, with no idea which sequence it belongs to or where
that sequence started -- so all of those steps can run once over the
entire concatenated batch, and the row count is now exactly the real
token count, never a wasted filler row.

## The one place that isn't "just a row"

Attention is different: a token is allowed to look at earlier tokens in
*its own* sequence, never another sequence's tokens. That's a real
constraint the matmul-only steps don't have, so it can't be folded into
one giant matmul the same way. `ForwardPrefillBatch` handles this by
looping once per sequence and calling the *exact same* `Attention()`
function every other path in this codebase already uses and already has
parity tests for, pointed at that one sequence's own slice of this call's
Q/K/V arrays and its own KV cache. No new attention code was written for
this at all -- the entire ragged-prefill feature is "batch everything that
can be batched, and reuse the already-correct function for the one thing
that can't."

## Why the tests passed on the first try

Both new tests (`RaggedPrefillMatchesRunningEachSequenceAlone`,
`RaggedPrefillContinuesFromExistingCacheLength`) compare the ragged
batched call directly against running each sequence alone through the
already-tested `Forward()` path, and they passed without a single fix.
That's not luck -- it's the direct payoff of reusing `Attention()`
unmodified instead of writing a new masked-attention kernel for the
ragged case: the only genuinely new code is bookkeeping (offsets, per-row
positions, per-sequence cache append/advance), and bookkeeping bugs are
much easier to get right by inspection than numerical kernel bugs are.

## What this replaces

Phase 23's defense entry named this directly: "prompts still prefill one
at a time because the executor has no ragged prefill interface."
`kiln_py/runtime/continuous_batch.py` now concatenates every fresh
request's prompt in a scheduler step into one `forward_prefill_batch`
call instead of looping `model.forward()` once per prompt -- the same
kind of change Phase 5's continuous-batching scheduler already made for
decode steps, applied to the one place prefill was still doing it the
slow way.

## What this doesn't claim

This is still a CPU reference implementation with no measured throughput
number for the padded-vs-ragged comparison -- the claim here is
correctness (same answer, less wasted work by construction) and real API
wiring, not a benchmarked speedup. It also doesn't yet inherit Phase 8's
paged KV cache; each sequence in `ForwardPrefillBatch` still uses its own
contiguous `KVCache`, so prefix sharing across sequences prefilled
together isn't part of this path.
