# Phase 20 — derivation notes (constrained / structured decoding)

## The idea, in one sentence

At every generation step, before picking the next token, figure out which
tokens would even be *legal* right now given the schema being filled in,
and make every illegal token mathematically impossible to pick (set its
score to negative infinity before sampling) -- not merely unlikely.
Nothing about the model changes; the exact same sampler this project
already built (`csrc/executor/sampler.cpp`) is reused unmodified. The only
new piece is *computing which tokens are legal right now*, which is a
question about the schema, not about the model.

## Why this is cheap to bolt onto an existing sampler

Kiln's sampler already supports masking out tokens (that's exactly what
top-k does: keep the best few, set everything else to `-infinity` before
the softmax). Structured decoding is the same mechanism, driven by a
different source of truth: instead of "the top-k highest-scoring tokens,"
the mask is "whatever the schema says is legal at this exact position."
This is why the request specifically called this "cheap to build" --
it's not a new capability, it's a new *reason* to use a capability that
already existed.

## Why the schema walks byte-by-byte instead of parsing whole tokens

This project's tokenizer (for the demo API) is byte-level with an empty
merge list, so a "token" and a "raw byte" are the same thing already --
which makes the constraining logic simpler to reason about than it would
be for a real subword tokenizer, where one token can represent several
characters at once and a naive per-character grammar would need to
consider every token whose *text* happens to be legal at the current
position, not just single bytes. That added complexity (checking a whole
vocabulary's token strings against a grammar, not just 256 byte values)
is real, extra work a production system does; it's a genuine
simplification available here specifically because of the byte-level
tokenizer already built in Phase 1.

## What this proves, and how

The test that matters here isn't "does it look like JSON" -- it's `json.loads()`
actually succeeding on the output, every single time, across many random
seeds, using the SAME untrained random model this whole project uses
elsewhere. An untrained model choosing freely would almost never produce
valid JSON by chance; the mask is what makes validity guaranteed rather
than likely.
