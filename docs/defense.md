# Defense — interview-facing explanations, per phase

Written from memory before re-reading the code (ADR-011). One page per
phase: what the component is, why it works, what it cost.

## Phase 0 — Foundations & the oracle

**What:** an `Arena`, a bump allocator over one `std::vector<std::byte>`.
`Allocate(n)` hands back a pointer into the block and advances an offset;
`Reset()` sets the offset back to zero. `csrc/bindings.cpp` is a `pybind11`
module exposing one function, `ping()`, that proves the Python↔C++ boundary
(constitution §6) actually builds and imports before anything real crosses
it. `tools/oracle.py` loads the reference model, hooks each decoder layer
with `register_forward_hook`, runs one forward pass, and saves input ids,
every layer's output tensor, and the final logits to disk.

**Why it works:** the arena works because allocation lifetime in this
project is phase-shaped, not object-shaped — a batch of work is done and then
thrown away as a unit, so tracking "how far into the block have I gotten" is
enough; there's no need for a general allocator that supports arbitrary
free(). `pybind11` works by generating C++ glue that wraps a C++ function
pointer as a CPython callable, handling argument marshalling and reference
counting at the boundary — that glue is exactly why constitution §6 insists
on a single, narrow, audited boundary file rather than binding scattered
throughout the codebase. The oracle works because forward hooks are the
standard PyTorch mechanism for observing intermediate values without
modifying the model's source — they run after a module's forward() and
receive its output.

**What it cost:** the first version of `Allocate()` had a debug-only assert
that fired on overflow in addition to returning a null pointer, so a debug
build and a release build behaved differently for the same input. The test
for this caught it immediately — the test expected a null pointer back and
instead got a crash. That's two behaviors doing the job of one, which is
exactly the kind of avoidable complexity the project is trying to stay away
from, so the assert was removed: now there's one rule, in every build --
run out of room, get a null pointer, every time. The pybind11 dependency
itself is the one boundary-crossing cost accepted by ADR-006 — everything on
either side of it stays hand-built.

## Phase 1 — Weights & tokenizer

**What:** `SafetensorsFile` opens a model's weight file, reads one small
JSON header describing where every tensor lives, and then hands back
direct pointers into a memory-mapped view of the file for each one --
nothing is copied. `BpeTokenizer` turns text into a list of numbers (and
back) by first mapping every raw byte to a stand-in character, then
repeatedly merging pairs of pieces together in the priority order recorded
in the tokenizer's merge list, until no more merges apply.

**Why it works:** memory-mapping works because the operating system will
happily let a program treat a file on disk as if it were already sitting in
memory, loading pages of it in only as they're actually touched -- so a
multi-gigabyte weight file can be "opened" instantly without reading the
whole thing up front. The tokenizer's merge order matters because it
encodes how often each pair of pieces appeared during training -- merging
in that priority order is what makes our tokenizer agree with the
reference tokenizer's choices, rather than just producing *some* valid
tokenization.

**What it cost:** real conformance testing against the reference
tokenizer on a large set of real sentences needs a real HuggingFace
install, which this offline session doesn't have (ADR-009) -- what's tested
here is that our own encode/decode logic is internally consistent (merges
happen in the right order, decoding perfectly reverses encoding), not that
it matches the reference byte-for-byte on real text yet. Also, our
pre-tokenizer (the step that splits text into words before merging) only
correctly classifies plain ASCII letters and digits; anything non-ASCII is
treated as "part of a word" rather than checked against real Unicode letter
categories, since C++'s regular expressions don't support that check the
way the reference tokenizer's does. Both limitations are stated here rather
than hidden, and both are exactly what the deferred real-fixture testing
would catch if either were actually wrong.

## Phase 2 — The forward pass

**What:** the full stack -- turn each input word into its numbers (an
embedding lookup), then repeatedly: normalize (RMSNorm), figure out what
matters by comparing every word to every earlier word (attention, with
positions baked in via RoPE), add that result back in, then run a small
"think about it" network (SwiGLU), add that back in too. After all layers,
normalize one more time and produce one score per possible next word
(logits).

**Why it works:** each layer only ever *adds* its result back onto the
running total (a "residual connection") rather than replacing it -- this is
what lets the numbers survive many layers deep without vanishing or
exploding, since even if one layer computes something close to zero, the
running total from every earlier layer is still there afterward.

**What it cost:** GEMM (matrix multiplication) here is a plain triple loop
with a cache-friendly ordering, not a fully tiled, hand-tuned kernel -- a
deliberate choice (ADR-004/010), since the Part II GPU path replaces this
entirely with cuBLAS anyway, so more effort here would be thrown away
later. No real checkpoint has been loaded through this code yet in this
session (see the Phase 1 entry above) -- everything tested so far uses
small, deterministically random weights, which is enough to prove the
*shapes and internal properties* are right (causal masking, residual
addition, batching correctness) but not yet that the numbers match a real
trained model's numbers.

## Phase 3 — KV cache & generation

**What:** `KVCache` is one pre-allocated block per layer that remembers
every earlier word's key and value numbers, so generating each new word
only costs the work for that one new word, not the whole sentence over
again. The sampler turns a row of scores into one chosen word, either
always picking the best one (greedy) or picking with some controlled
randomness (temperature, top-k, top-p, repetition penalty).

**Why it works:** see docs/learning/phase-03.md for the full reasoning --
in short, caching is purely a speed trick with no change in meaning, which
is exactly why the cached path is tested against the no-cache path and
required to match exactly.

**What it cost:** nothing scoped out here, but this is where the most
important test in the whole project so far lives
(`Model.CachedDecodeMatchesFullRecompute`) -- if a future change to
attention, RoPE, or the cache ever broke this test, it would mean
generated text had quietly started depending on whether the cache was used,
which is the single easiest correctness bug in this kind of system to miss
by eye.

## Phase 4 — Static batching

**What:** several sentences, padded to the same length, run through the
model together as one bigger rectangle instead of one sentence at a time.

**Why it works:** see docs/learning/phase-04.md -- the short version is
that most of the model's layers don't need to know a batch is happening at
all; only attention needs an explicit boundary so padding and other
sentences never leak into a sentence's real answer.

**What it cost:** the real §6 boundary-cost number this phase's DoD asks
for was measured on a small toy model, on this machine's CPU, not on the
real production hardware and checkpoint the finished project will
eventually use (see BENCHMARKS.md phase 4 row) -- it's a genuine,
reproducible measurement (~0.06% overhead), just not yet the final number.

## Phase 5 — Continuous batching (the scheduler)

**What:** a Python scheduler that decides, every single step, which
requests are currently allowed to run, admitting new ones the instant
there's room rather than waiting for the whole batch to finish.

**Why it works and what it cost:** see docs/learning/phase-05.md for the
real bug found and fixed while building this (admission checking a
request's worst-case final size, not its current size) -- this is the
strongest interview story in Part I so far: a bug that would have caused a
production cache overflow a few steps after looking correct, caught by a
seeded randomized test before it ever ran against a real model.

## Phase 6 — The API & Part I assembly

**What:** a FastAPI server exposing `/v1/completions`, matching the shape
of OpenAI's API (including streaming), backed by the real C++ model and
scheduler-free single-request generation loop from Phase 3.

**Why it works and what it cost:** see docs/learning/phase-06.md for the
real UTF-8 decoding bug this phase caught -- a byte-level tokenizer's
output isn't automatically valid text, and assuming otherwise crashed the
first real end-to-end test of the streaming endpoint. Also worth stating
plainly: this session's API serves an untrained, randomly-initialized
model (no real checkpoint was available offline), and it calls the
single-sequence generation loop directly rather than being wired through
the Phase 5 scheduler for true concurrent multi-request continuous
batching -- that wiring (an async request queue feeding the scheduler,
with the scheduler driving batched calls into the executor) is real,
non-trivial engineering earmarked as follow-up work, not something this
session quietly skipped without saying so.
