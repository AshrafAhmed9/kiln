# Correctness — the "how do you know" document

This is the running catalog of what the parity harness checks, what it
can't, and every real bug it (or a test written in this same spirit) has
caught. Each entry: what broke, how it was caught, the fix, and what I
misunderstood -- the misconception, not just the bug.

## Phase 0 — the debug-only assert that made two builds disagree

**What broke:** `Arena::Allocate()` had a debug-only assert on top of its
documented "returns a null pointer when full" contract. In a normal debug
build (assertions on), running out of room crashed the program instead of
returning null.

**How it was caught:** the test `Arena.ReturnsNullWhenExhausted` expected a
null pointer and instead the whole test process aborted.

**The fix:** removed the assert. One rule, in every build: run out of
room, get a null pointer back, always.

**What I misunderstood:** I assumed "crash loudly in debug, degrade
gracefully in release" was a reasonable default without checking that it
actually matched what the class's own documented contract promised. Two
different behaviors for one function is complexity, even when each half
seems reasonable on its own -- the fix that mattered wasn't fixing the
assert's wording, it was noticing the design had two contracts instead of
one.

## Phase 5 — admission control that checked the wrong number

**What broke:** the scheduler let a new request in in as long as its
*current* size fit inside the remaining budget. Since every running
request grows by one word every step, a batch that fit perfectly the
moment a request was admitted could still overflow its budget a few steps
later, once everyone already running had grown some more.

**How it was caught:** not by a failing test at first -- by re-reading the
scheduler's own logic while writing its randomized invariant test
(`test_memory_budget_is_never_exceeded_under_random_arrivals`) and
realizing the check being written (`tokens_reserved() <= max_batch_tokens`
after every step) would only be meaningful if admission itself reserved the
right number up front.

**The fix:** admission now checks a request's worst-case final size (its
prompt length plus its full allowance of new words), not its size right
now -- matching how a cache that can't be resized after the fact actually
has to be managed.

**What I misunderstood:** I was thinking about "does it fit right now,"
which is the natural first question, without carrying through the
follow-up question: does it still fit after it's allowed to grow? Any
system where one thing's size increases over time needs its admission
decisions to be based on where that thing will end up, not where it starts.

## Phase 6 — raw bytes assumed to be text

**What broke:** the streaming API endpoint crashed with a Unicode decoding
error the first time a generated token happened to be a raw byte that
wasn't valid text on its own.

**How it was caught:** the very first real run of the streaming endpoint's
test, immediately.

**The fix:** the C++ tokenizer's decode function now hands back raw Python
bytes instead of an auto-converted string, and the Python side explicitly
decodes those bytes with a "replace anything broken with a placeholder
character" policy.

**What I misunderstood:** I treated "decode these tokens" as if it always
produces displayable text, when a byte-level tokenizer's real contract is
narrower than that -- it produces bytes, and turning bytes into text is a
separate, sometimes-lossy step that has to be done deliberately, especially
one token (one possibly-incomplete character) at a time during streaming.

## Phase 8 — copy-on-write that never actually copied

**What broke:** the first version of `PagedSequence::PrepareWriteSlot`
allocated a new, private block whenever a shared block needed to be
written to, but never actually copied the old block's numbers into the new
one -- so a sequence that diverged from a shared prefix would have started
writing into (and, worse, reading stale garbage out of) an uninitialized
block.

**How it was caught:** before any test was even run, while re-reading the
function against its own documentation comment and noticing the comment
claimed a copy that the code never performed.

**The fix:** added `PagedKVCache::CopyBlockContents`, which copies both K
and V numbers, at every layer, from the old block into the new one, and
call it at the exact moment a shared block needs to be written to.

**What I misunderstood:** I wrote the allocate-a-new-block half of
copy-on-write and treated that as if it were the whole mechanism, when
allocating room and actually copying the data into it are two separate
steps -- skipping the second one leaves the "write" correct but the
"copy" fictional. The lesson generalizes: a function's own comment
describing what it does is worth treating as a claim to verify against the
code, not just documentation to trust.
