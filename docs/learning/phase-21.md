# Phase 21 — what a prefix-cache hit rate actually means

## A cache hit needs a denominator

"Hit rate" sounds obvious until the thing being cached has several blocks.
For this workload, one **lookup** means a new sequence needs one logical
prefix block. It is a **hit** when that logical block already exists in the
paged cache and the new sequence can reference it instead of allocating a
second physical block. The rate is therefore:

```
shared logical prefix blocks / all logical prefix blocks requested
```

That definition deliberately does not call every token after a shared prompt
a hit. The cache shares blocks, not an abstract notion of similar text.

## The measured workload

The executable creates a 16-token system prompt, which is two full blocks at
the workload's block size of eight. It then creates up to four variants by
sharing those two blocks and appending four variant-specific tokens. Every
variant therefore ends with a partial third block. Eighty seeded conversations
fork one of the variants and write one to four divergent tokens.

The first divergent write has to copy that partial third block because the
variant still owns it too. This is important: a workload that only forks and
releases would make sharing look good but would never prove that later writes
remain private. The run with seed `20260824` produced 252 prefix-block
lookups, 248 hits, and 80 copy-on-write events: **98.41%**.

The four misses are the first private block for each of the four variants.
Everything else is a reference to a block already present in the cache. That
makes the result easy to inspect and gives the regression test a concrete
invariant, rather than merely checking that a percentage looks plausible.

## Why this is not playground traffic

The playground has no traffic store, no deployed users, and no request path
wired to the paged scheduler. Claiming a hit rate from it would turn a missing
measurement into a made-up one. The synthetic workload is useful because it
states exactly which access pattern it measures and exercises the real C++
allocator; it is not evidence that any future deployment will see 98.41%.

## Why the status page is intentionally small

Prometheus already owns the request and latency counters. `/status/data`
reads those collector samples, and `/status` displays them with a prominent
local-session warning. The page contains counters, latency observation count,
and total observed completion time only. It does not derive p99, QPS, uptime,
or costs from one local process because none of those numbers is present or
supportable here.
