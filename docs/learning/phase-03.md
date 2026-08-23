# Phase 3 — derivation notes (KV cache and generation)

## Why caching keys and values is safe

When generating text one word at a time, each new word needs to "look back"
at every earlier word (that's what attention does). Without a cache, we'd
recompute every earlier word's key and value numbers from scratch, every
single time, even though those numbers never change once a word has been
read. The cache just remembers them the first time, so each new step only
has to compute the *one new word's* key and value, then reuse everything
that came before. This is a pure speed optimization with no change in
meaning -- which is exactly why "does the cached path give the same answer
as recomputing everything?" (`Model.CachedDecodeMatchesFullRecompute`) is
the single most important test in this phase: if caching ever changed the
answer, it would mean something about *where* a word thinks it is (its
position) or *what it can see* (the causal mask) went wrong when the cache
was introduced, not that the cache itself is inherently risky.

## Why a contiguous cache has to reserve room up front

This phase's cache is one flat, pre-allocated block per layer -- it can't
be resized once created. That single fact is what later shapes the
scheduler in Phase 5: if a cache can't grow, then a request has to be
promised its worst-case room (its prompt length plus every word it might
still generate) the moment it's allowed to start, or it could run out of
room partway through with no way to make more. Phase 8's paged cache fixes
this by chopping the cache into small movable blocks instead of one fixed
slab, so reservations are no longer needed -- but that's future work; the
plan here is the honest, simpler starting point.

## Sampling -- why greedy has to be exact and everything else just has to be replayable

Greedy decoding (always pick the single best-scoring word) has one right
answer for a given set of scores, so it's the version compared against the
reference model. Every other setting (temperature, top-k, top-p,
repetition penalty) intentionally introduces randomness, so there's no
single "correct" output to compare against -- instead, the promise is
determinism: the same random-number-generator state must always produce
the same choice, so a specific generation can always be replayed and
studied later, the same principle Consensa uses for reproducing bugs.
