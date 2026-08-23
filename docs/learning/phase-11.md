# Phase 11 — derivation notes (the perf-regression harness and testing the harness itself)

## Why a correctness check has to be tested too

A parity check that always passes -- whether or not the code it's
checking is actually right -- is worse than no check at all, because it
creates false confidence. The only way to trust that a tolerance-based
comparison ("do these two sets of numbers agree closely enough?") would
actually catch a real bug is to deliberately hand it a *known-wrong*
version of something and confirm it says no. This is the same idea as a
smoke detector test button: pressing it doesn't detect a real fire, but it
proves the detector isn't silently broken.

## The specific bug used to test it

A very real, very plausible kind of numerics bug is forgetting to divide
by the number of elements when computing an average -- easy to write by
accident, and the kind of thing that would produce numbers that are
*wrong by a consistent, large factor* rather than randomly wrong, which is
exactly the shape of bug a parity tolerance check is supposed to catch. A
deliberately broken version of RMSNorm that sums the squared values but
skips dividing by the row's length is used here as that known-wrong
reference.

## Honest scope note on nightly GPU perf-regression tracking

The plan calls for nightly performance-regression tracking on a fixed GPU.
As already recorded in ADR-009, no free, persistent GPU CI runner exists,
so this can't be built as originally scoped in this project. What can be
built and is built here is the *correctness* half of Phase 11 -- proving
the parity-checking methodology itself would catch a real class of bug --
independent of the *performance* half, which stays a manual, versioned
measurement per `BENCHMARKS.md`'s existing discipline.
