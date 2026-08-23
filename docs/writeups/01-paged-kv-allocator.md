# The paged KV cache: giving a model's memory its own virtual memory system

Every word a language model generates depends on remembering the words
that came before it. Concretely, that means storing two numbers per word,
per layer, per attention head — the "key" and "value" that let later
words look back and decide what mattered. Do this the naive way — one
solid, pre-allocated block of memory per conversation, sized for the
worst case up front — and you waste enormous amounts of memory the moment
real usage doesn't match your worst-case guess. Reserve room for 500
words and only generate 50, and 450 words' worth of memory sat there,
unusable by anyone else, the whole time.

**The fix is the same one operating systems figured out for RAM decades
ago: paging.** Instead of handing a program (or, here, a conversation) one
big contiguous slab of memory, carve all of memory into small, fixed-size
blocks up front, and hand out blocks from a shared pool only as they're
actually needed. Keep a small table — a "block table" — recording which
physical blocks belong to which conversation, in order. Nothing about the
core math changes; only *where the numbers live* changes, and a small
layer of indirection (block table lookup) sits between "the 47th word of
this conversation" and "the actual bytes for it."

## The part that makes this genuinely worth building, not just clever bookkeeping

Once memory is chopped into blocks with a shared pool, something else
becomes possible for free: **two conversations that start with the exact
same text can share the same physical blocks for that shared part.** This
matters more than it sounds — production chat systems routinely prepend
the same long system prompt to every conversation. Under the naive
design, that system prompt's memory gets duplicated once per
conversation. Under paging, it's stored once, and every conversation's
block table simply points at the same blocks.

The catch, and the interesting engineering problem, is what happens the
moment two conversations that share a block need to actually *diverge* —
one of them wants to write a new word into what was, until that instant,
shared memory. Writing into it naively would silently corrupt the other
conversation's copy too. The standard answer is **copy-on-write**: before
writing into a block more than one conversation still points at, first
copy its contents into a brand-new, privately-owned block, point only the
writing conversation's table at the new copy, decrement the old block's
share count, and leave the other conversation completely untouched,
still pointing at the original. Copying only ever happens at the exact
moment two paths genuinely diverge — never before, and never
unconditionally.

## The bug this actually caught

Writing this, I made — and caught, before running a single test — exactly
the mistake this design exists to prevent. My first draft of the
copy-on-write path allocated a fresh, private block the instant a shared
block needed writing to... and then never actually copied the old block's
numbers into it. The new block just sat there full of whatever garbage
happened to be in memory. Every piece of the *bookkeeping* was correct —
the right block got allocated, the right reference counts got
decremented, the right table entry got updated — and the actual *data*
was simply never moved. It's a genuinely easy mistake to make, because
"allocate room for a copy" and "actually perform the copy" read, at a
glance, like the same step. They aren't, and the fix was one explicit
function — `CopyBlockContents` — called at the exact moment it was
missing.

The tests that would have caught this at runtime (rather than on
careful re-reading) are the ones that matter most here: fork a
conversation from a shared prefix, have each side write its own new word,
and check that *both* the new words are correct *and* the original shared
prefix is still exactly what it was before either side touched anything.
That last check — the untouched, shared part staying untouched — is the
entire promise copy-on-write exists to keep, and it's the one a
superficial "does it crash" test would never have exercised.

## What this is, and isn't, proof of

Everything above — the block allocator, the reference counting, the
copy-on-write correctness — runs and is tested entirely on a CPU, with no
GPU involved. That's deliberate and honest: the *algorithm* (which
numbers get stored where, and the rules for sharing and copying them) is
pure bookkeeping, and bookkeeping can be verified without any special
hardware. The actual paged-attention *kernel* — the GPU code that reads
through this block table at inference time, at real speed, on real
hardware — is a separate piece of work, written to spec but not yet
compiled or run anywhere in this project, since doing that needs a real
GPU this development environment doesn't have. The allocator being
correct is a necessary foundation for that kernel to be correct; it
isn't a substitute for actually measuring it.
