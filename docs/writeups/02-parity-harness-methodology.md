# "How do you know it's still right?" — building a parity harness instead of hoping

Most from-scratch inference projects answer the correctness question with
a shrug: it ran, the text looked like text, ship it. That's not a
standard that survives contact with a real optimization. The moment you
touch a kernel's memory layout, change how batching works, or shrink
weights down to 4 bits, you need an actual answer to "did this change what
the model says?" — not a hopeful one.

## The core idea: never trust your own implementation alone

Every numerically meaningful piece of this project is checked against an
independent reference, computed a different way. For the forward pass,
that reference is HuggingFace's own model implementation, run once,
recorded once, and diffed against forever after. For the quantizer, it's
a second implementation, written in a different language (Python,
independent of the C++ being checked), computing the same thing from
scratch. For speculative decoding, it's a mathematical proof from the
literature about what the *output distribution* must equal, turned into
a test that checks a specific, provable special case exactly.

That last point is worth dwelling on, because it's the most interesting
lesson this project surfaced: **not every correctness claim is checked the
same way, and knowing which kind of claim you're making matters.**

## Three different shapes of "correct," and three different tests

**Exact correctness** — for things with one right answer, the check is
"do these numbers match within a tight, documented tolerance?" The
forward pass, RMSNorm, RoPE, attention: all of these have exactly one
correct output for a given input, so the test is a direct numerical
comparison against a reference.

**Exact-by-construction correctness with randomness involved** —
speculative decoding is the interesting middle case. The *algorithm* is
supposed to preserve an exact probability distribution even though real
runs involve randomness, which makes "just diff the numbers" impossible in
general. But there's a special case — always picking the single most
likely word (greedy decoding) — where all the randomness disappears and
the guarantee collapses into something fully deterministic: speculative
decoding's output must be *token-for-token identical* to running the real
model alone. That's the test this project actually leans on, because it's
checkable directly and completely, with zero statistics involved — the
strongest kind of correctness claim available, used exactly where it
applies.

**Approximate, honestly-measured correctness** — quantization doesn't
have a "right answer" to check against at all; shrinking a number from 32
bits to 4 necessarily loses some information, by design. The only honest
question is *how much*, measured and published (a perplexity delta, a
KL-divergence number), not asserted. This project's quantizer round-trip
tests check the *math* is implemented correctly (the error stays inside a
predictable, small bound); the *real* accuracy cost — against a real
trained model and a real dataset — is explicitly still an open,
deferred measurement here, not quietly assumed away.

## Testing the test itself

A correctness check that would pass no matter what is worse than no check
at all — it manufactures false confidence. This project includes one test
specifically to guard against that: a deliberately broken version of
RMSNorm (correct except for one missing division) is compared against the
real one using the exact same tolerance logic used everywhere else, and
the test asserts that comparison actually reports disagreement. It's the
software equivalent of pressing the "test" button on a smoke detector —
proving the detector isn't quietly broken, not detecting an actual fire.

## The bugs this approach actually caught, in this project, while building it

This isn't abstract. Building this project surfaced several real bugs,
each caught by exactly the kind of check described above, not by luck:

- A debug-only safety check that made a memory allocator behave
  differently in debug builds versus release builds — caught by a test
  that expected one specific, documented behavior and got a different one.
- A copy-on-write cache implementation that allocated room for a copy but
  never performed it — caught by a test checking that an *untouched*
  shared prefix stayed byte-for-byte untouched after a divergent write,
  not just that nothing crashed.
- A scheduler that checked a request's *current* memory footprint at
  admission time instead of its *worst-case future* footprint — a bug that
  wouldn't have shown up in a quick smoke test at all, since it only
  causes an overflow several steps *after* a request looked like it fit
  fine.

None of these were found by staring at the code and feeling confident.
They were found because a specific, principled check existed to catch
that specific class of mistake — which is the entire argument for
building the harness before writing the code it verifies, not after.
