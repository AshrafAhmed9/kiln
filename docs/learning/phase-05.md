# Phase 5 — derivation notes (the continuous-batching scheduler)

## The core idea, in one sentence

Instead of deciding once which sentences are in the batch and running them
all to completion together, decide *every single step* who gets to run --
so the moment a sentence finishes, its spot can be handed to a new,
previously-waiting sentence on the very next step, not whenever the whole
batch happens to empty out.

## Why admission has to reserve worst-case room, not just current room

This was a real bug I found and fixed while building this phase, not just
a design point I already knew going in. My first version let a new
sentence in as long as its *current* size fit in the remaining budget. But
every running sentence grows by one word every single step -- so a batch
that fit perfectly the moment a new sentence was let in could silently
overflow the very next step, once everyone already in the batch grew a
little more. The fix: a sentence's admission is decided by its *worst-case*
final size (its prompt length plus every word it's still allowed to
generate), not its size right now. This matches how a cache that can't be
resized after the fact (this phase's contiguous cache) actually has to be
managed in a real system -- you can't discover you're out of room midway
through generating someone's answer.

## Why "first come, first served" is a defensible, simple policy

A request only ever moves from waiting to running once it actually fits;
an earlier request is never skipped over just because a later, smaller
request would fit more easily. This means a very large request can make
smaller requests behind it wait -- a real, honest tradeoff, not something
hidden. Fancier policies (priority queues, fair-share scheduling) are
explicitly out of scope for this project (see the plan's non-goals table):
they solve a real problem, but they're not where the systems lesson lives
for a from-scratch build. The one thing that can't be allowed to happen,
and which is tested directly, is a request that can *never* fit (its
worst-case size alone is bigger than the entire budget) being left to wait
forever for room that will never exist -- that request is rejected
immediately at submission time instead.
