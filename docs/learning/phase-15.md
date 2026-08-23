# Phase 15 — derivation notes (the demo and the writeups)

## Why a scripted demo is itself a correctness check

A demo script that has to actually run, from a clean checkout, in order to
be shown to anyone, is a different thing from a demo that's merely
described in a README. Writing `demo.sh` and then actually running it
(not just writing it and assuming it would work) surfaced nothing broken
this time -- but the discipline of "the demo must actually execute, every
time, or it doesn't count as done" is the same discipline behind every
test in this project. A demo that silently rotted would be a worse
failure than a missing feature, because nobody would notice until the
moment it mattered most.

## Why the writeups draw from the learning notes instead of being written from scratch

Each phase's `docs/learning/phase-NN.md` was written at the time the
concept was actually being worked through -- including the mistakes,
which is exactly the material that makes a technical writeup interesting
rather than a restatement of a paper's abstract. Writing the three
writeups by drawing on those notes (rather than reconstructing the
reasoning from memory months later, which is what the plan's original
sequencing would have required) is a direct payoff of doing the learning
notes throughout, not just at the end.

## Honest scope

The three writeups here explain real, tested mechanisms this project
built (the paged allocator, the parity methodology, the quantization
tradeoffs) and are honest, in each one, about exactly where the real
project's proof stops and a deferred, GPU-dependent measurement begins.
None of them claim results that weren't actually produced in this
session.
