# Phase 4 — derivation notes (static batching)

## Why batching several sentences together is faster at all

A computer's raw number-crunching hardware is much better at doing one
huge multiplication than many small ones back to back -- there's a fixed
setup cost to each operation, and a big matrix multiply spreads that cost
over far more actual work. So instead of running the model once per
sentence, we glue several sentences into one bigger rectangle (padding the
shorter ones so they're all the same length) and run the model once for
the whole rectangle. The linear layers and normalization steps don't even
need to know a batch is happening -- they just see more rows. Only
attention needs to know where one sentence ends and another begins, so
that padding from one sentence can never leak into another sentence's
answer, and so a sentence can never accidentally attend into a completely
different sentence sharing the same batch.

## Why this isn't the same thing as continuous batching (Phase 5)

This phase's batching is static: every sentence in the batch has to be
decided on before the batch starts, and the whole batch runs together from
start to finish. If one sentence in the batch is much longer than the
others, everyone else has to wait for it, since the rectangle is only ever
as short as its longest member. Phase 5's continuous batching fixes exactly
this: it lets a finished sentence's spot be handed to a new sentence
immediately, without waiting for the whole batch to empty out first. This
phase's honest limitation is worth stating plainly rather than glossing
over: static batching's win is *hardware utilization* (bigger, more
efficient multiplications); continuous batching's win is *scheduling*
(nobody waits on the slowest sentence). They solve two different problems,
and production systems need both.
