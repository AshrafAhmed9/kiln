# Phase 8 — derivation notes (paged KV cache)

## The problem with Phase 3's cache

Phase 3's cache is one solid, pre-allocated block per sequence, sized for
the sequence's worst-case length up front. This works, but it wastes
memory: if a request is allowed to generate up to 500 words but only ends
up generating 50, the other 450 words' worth of room sat reserved and
unused the whole time. And when two requests share an identical prompt
(a common system prompt, for example), Phase 3's design has no way to
avoid storing that shared prefix twice.

## The paging idea, in one sentence

Instead of giving each sequence one solid block sized for its worst case,
carve all the cache's memory into many small, fixed-size blocks up front,
and let a sequence grow by picking up new blocks from a shared pool only
as it actually needs them -- exactly the way an operating system's virtual
memory doesn't give a program one giant contiguous chunk of RAM either; it
hands out fixed-size pages as needed, and a "page table" records which
physical pages belong to which program. Here, the "page table" is called a
block table: a list of which physical block indices belong to a given
sequence, in order.

## Why this makes prefix-sharing possible

If two sequences start with the exact same prompt, their block tables can
simply point at the exact same physical blocks for that shared part --
no data has to be duplicated. The catch is what happens when one of those
sequences needs to *change* something in a block it doesn't fully own
alone: naively writing into a shared block would silently corrupt the
other sequence's copy too. The fix is "copy-on-write": before writing into
a block that's currently shared by more than one sequence, first copy its
contents into a brand new, privately-owned block, point only the writing
sequence's block table at the new copy, and let the other sequence
continue pointing at the original. Copying only happens at the exact
moment two sequences' paths actually diverge -- not before, and not
unconditionally on every write.

## Why "no leaks, no double-free, fragmentation bounds" are the right properties to test

This design's core promise is bookkeeping-shaped, not math-shaped: it has
to never reuse a block that's still owned by someone, and it must never
lose track of a block that's been freed and should be reusable again. This
is exactly the same category of property that a general-purpose memory
allocator has to guarantee, which is why it's tested the way an allocator
is tested (randomized sequences of allocate/free operations, checking
invariants after every single one), rather than tested with hand-picked
example inputs the way pure math functions are.
