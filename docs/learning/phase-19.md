# Phase 19 — derivation notes (chunked prefill and the scheduler policy frontier)

## The real mistake made while building this, twice

The first version of this benchmark modeled chunking as strictly better
on every axis — faster for everyone, no downside. That's wrong, and it's
wrong in an interesting way: it would have read as a plausible, confident
claim right up until someone asked "so why doesn't everyone just always
use the smallest possible chunk size?" — a question with no good answer
under that (buggy) model, because the model had accidentally given
chunking a free lunch. The fix was making interleaving cost something
real: every point where decode gets to advance is a point where the long
request's own prefill does not, which is what turns the chunk-size sweep
into an actual trade-off instead of a strictly-improving dial.

The second mistake was subtler: after fixing the free-lunch problem, an
early version of the benchmark still showed almost no difference between
policies — because the synthetic workload gave every "short" request a
long enough prompt that most of them were still waiting for their own
first turn at the single prefill slot when the long request arrived. The
mechanism being tested (interleaving with *already-decoding* requests)
was never actually exercised. Both mistakes were caught the same way: by
noticing a suspiciously clean or suspiciously flat result and asking
*why*, rather than accepting a plot that "looked reasonable."

## The metric that actually matters, found by a third correction

Even with the model fixed, the first honest comparison used *total
completion time* as the metric — and completion time barely moved across
chunk sizes. On inspection, that's a genuine, correct property of this
model: every interleave step costs exactly one step and grants exactly
one tick, a zero-sum reallocation that doesn't change when the *last*
token arrives. What it changes is the size of the *worst gap* between two
consecutive tokens along the way — the thing a user watching text stream
in actually notices. Switching the metric to "longest stall any streaming
request experiences" is what finally produced a real, monotonic,
honest frontier: smaller chunks bound the worst stall tightly, at the
cost of the interrupting request's own time-to-first-token.

## Why priority scheduling didn't help in the benchmark workload

`priority_shortest_first` produced identical results to plain FCFS in
`bench/scheduler_policy_comparison.py`. This isn't a bug -- it's a real
limitation of priority-without-preemption: reordering the *waiting*
queue does nothing once a low-priority request has already claimed the
single active slot before any higher-priority request arrives. Priority
scheduling only helps when it gets a chance to act *before* commitment;
after that, only preemption (stopping and resuming already-started work)
or chunking (voluntarily yielding partway through) can help -- which is
exactly why production schedulers combine chunked prefill with priority
policies rather than treating them as alternatives.
