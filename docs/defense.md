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

## Phase 7 — CUDA port

**Status, upfront:** written on a machine with no NVIDIA GPU. None of
`csrc/kernels/cuda/*.cu` or `csrc/kernels/triton/rope.py` has been
compiled or run in this session -- see docs/learning/phase-07.md.

**What:** hand-written raw CUDA kernels for the three "headline" kernels
(attention, RMSNorm, greedy argmax sampling), each deliberately mirroring
the already-tested CPU version's algorithm rather than trying anything
novel; RoPE written twice, once in raw CUDA and once in Triton, so there's
a real basis for the "why not raw CUDA for everything?" comparison instead
of an assertion.

**Why it works:** each kernel follows the standard GPU pattern of "many
threads each do a small piece of the work, then combine their partial
results" -- RMSNorm and the sampler combine partial sums/maxes across
threads using a warp shuffle (a fast, no-shared-memory way for nearby
threads to exchange values); attention uses on-chip shared memory to hold
one query's full row of scores, since every thread needs to read that same
row more than once (once to find its max, once to sum it, once to use it
as weights).

**What it cost:** the first version of the attention kernel used a
different, "streaming" style of combining partial results across threads
with a raw atomic add -- while writing it, I recognized that approach has
a real race condition when many threads update a shared accumulator
concurrently without the specific synchronization that pattern actually
requires, so I rewrote it to mirror the CPU version's simpler, already
correct two-pass structure instead of shipping something merely
plausible-looking. This is exactly the kind of mistake that's cheap to
catch by careful reading before hardware is involved, and expensive to
debug after.

## Phase 8 — Paged KV cache

**What:** all of the cache's memory is divided into small, fixed-size
blocks up front. Each sequence keeps a list (a "block table") of which
blocks belong to it, and picks up a new block from a shared free pool only
when it actually needs more room. Two sequences that start with an
identical prompt can share the same blocks for that shared part, and only
get their own private copy of a block the moment either one needs to
change something in it (copy-on-write).

**Why it works:** see docs/learning/phase-08.md -- the short version is
that this is the same idea as an operating system's virtual memory pages,
applied to a model's cache instead of a program's RAM.

**What it cost:** the first version of the copy-on-write path allocated a
brand new private block but never actually copied the old block's numbers
into it -- a half-finished implementation that would have silently handed
back a block full of garbage the moment two sequences diverged. This was
caught before ever running a single test, just by re-reading what the
function actually did versus what its own comment claimed it did, and
fixed by adding a real `CopyBlockContents` step. It's a useful reminder
that "the code runs without crashing" and "the code does what it claims to
do" are different bars, and only one of them is checked by the compiler.

## Phase 9 — Quantization

**What:** two weight-only quantization schemes: INT8 with one scale per
output row, and INT4 with one scale per smaller group of weights within a
row, packed two values to a byte. Both are cross-checked against an
independently written Python reference implementation
(`tools/quantize_ref.py`), through the same pybind11 boundary the real
model uses.

**Why it works:** see docs/learning/phase-09.md -- the short version is
that a small integer plus one scale number per group can stand in for a
whole group of floats, trading some precision for a lot less memory, and
grouping the scale more finely (INT4's smaller groups vs INT8's per-row
scale) protects against any one outlier ruining precision for its
neighbors.

**What it cost, stated honestly:** the real deliverable this phase asks
for -- a measured perplexity/KL-divergence table against WikiText-2, using
a real trained checkpoint -- isn't possible in this offline session (no
real model, no real dataset). What's verified instead is a synthetic proxy
(quantized weights used in a real matmul stay close to the full-precision
result on random data) and the round-trip error bound math, which prove
the *mechanism* is implemented correctly, not the *real-world accuracy
cost* the finished project ultimately needs to publish. This project's
quantizer is also plain round-to-nearest, not the more sophisticated,
output-aware rounding real schemes like GPTQ use -- a real and named gap,
not an implied equivalence.

## Phase 10 — Speculative decoding

**What:** a small "draft" model guesses several words ahead; a single pass
through the real "target" model checks all of those guesses at once, using
a specific accept-or-replace rule (rejection sampling) so the final output
has exactly the same odds of occurring as if the target model alone had
generated every word.

**Why it works:** see docs/learning/phase-10.md for the full rule; the
short version is that "accept with probability min(1, p/q)" plus "on
rejection, resample from the leftover difference max(0, p-q)" is the
specific, provable choice that keeps the output distribution exactly
unchanged, which is what makes speculative decoding a genuine speedup
rather than an approximation.

**What was actually proven here, and how:** in greedy mode (always pick
the best word, no randomness), the accept-or-replace rule collapses to
something fully deterministic -- and the test in
`tests/py/test_speculative_decode.py` proves, by direct token-for-token
comparison (not a statistical approximation), that speculative decoding's
output exactly matches running the target model alone, across several
different draft lengths. That's the strongest kind of correctness claim in
this project: not "close enough," but bit-for-bit identical, checked
directly. The general randomized case is exercised for "does it run
correctly end to end" but its exact distribution-preservation isn't
statistically proven in this session -- that would need a much larger
sample than a quick CPU check can reasonably run, and is better done once
real GPU time is available.

**What it cost:** this implementation deliberately doesn't use the KV
cache at all (every round recomputes the whole context from scratch),
because integrating it with the cache would require the cache to be able
to roll back when a guess is rejected -- a capability Phase 3's
append-only cache doesn't have. This is a real, named simplification, not
a hidden one: it makes the implementation slower than real speculative
decoding needs to be, in exchange for keeping the part actually being
taught here (the acceptance rule) correct and cleanly separated from a
cache feature that doesn't exist yet.


## Phase 11 — Testing the parity harness itself

**What:** a deliberately broken RMSNorm (correct except it forgets to
divide by the row length before taking the square root) is compared
against the real, already-tested RMSNorm using the same kind of
tolerance check the parity harness uses everywhere else.

**Why it works:** see docs/learning/phase-11.md -- the short version is
that a correctness check which always passes regardless of whether the
underlying code is right is worse than having no check at all, so the
check itself has to be shown to actually fail on a known-wrong input, not
just assumed to work because it's written in the same style as the real
ones.

**What it cost:** the plan's real Phase 11 deliverable -- nightly
performance-regression tracking on a fixed GPU -- isn't achievable free
(ADR-009, no free persistent GPU CI runner exists). What's built instead
is the correctness half only: proof that the tolerance-checking
methodology would catch a real class of bug if one were introduced.

## Phase 12 — Tensor parallelism

**What:** column-parallel and row-parallel matrix multiplication,
simulated across a configurable number of "ranks" purely in Python/numpy,
proven to produce exactly the same numbers as the unsharded computation
for 1, 2, and 4 simulated ranks -- including the real paired pattern
(column-parallel into row-parallel, needing only one combine step for the
whole two-layer block, not one per layer).

**Why it works:** see docs/learning/phase-12.md -- splitting the *output*
side of a matmul needs no communication until the results are placed side
by side; splitting the *input* (contraction) side means every rank only
has a partial answer, so the partial answers have to be added together.

**What it cost, stated plainly:** this proves the sharding *algorithm* is
correct -- which numbers get split where, and how they combine back into
the right answer -- using ordinary local addition to stand in for a real
network all-reduce. It does not, and can't, prove anything about real
multi-GPU behavior (NCCL communication, actual scaling efficiency, real
communication overhead), since no multi-GPU hardware was available in this
session. That real measurement is genuinely deferred to when GPU time
(Kaggle, per ADR-009) is available.

## Phase 13 — Evaluation infrastructure

**What:** exact-match task scoring, a perplexity calculator, bootstrap
confidence intervals, a paired regression gate (candidate vs. baseline),
and a canary-replay diffing tool -- the machinery for deciding whether a
model change made things better, worse, or just noisily different.

**Why it works:** see docs/learning/phase-13.md -- the short version is
that a single average score can't tell noise from a real regression, but
resampling the same results many times and looking at how much the
average wobbles can.

**What it cost, and how it was tested:** every piece here is tested
against small, hand-controlled inputs rather than the real model, and
deliberately so -- the only model in this project is untrained and random,
so testing "does the eval math get the right answer" against known,
constructed cases is a stronger, more honest test than running the real
model and hoping the numbers look plausible. The regression gate is
specifically checked both ways: it must flag a candidate that's worse on
every paired question, and it must NOT flag a candidate whose results are
just noisily mixed (some better, some worse) -- a gate that only knows how
to say "yes" isn't a real gate.

## Phase 14 — Adapter-aware serving (fine-tuning itself out of scope)

**What:** given an already-trained LoRA adapter (two small matrices, A and
B), fold it into a served model's weight matrix once, at load time, so the
rest of the forward pass runs completely unmodified afterward.

**Why it works:** see docs/learning/phase-14.md -- merging once at load
time means there's no separate "adapter-aware" code path in the forward
pass to build and re-verify; the already-tested executor just sees a
slightly different weight matrix.

**What it cost, stated as plainly as possible:** this phase does **not**
include actually training a LoRA adapter, a real data pipeline, or the
multi-GPU scaling study the plan's Phase 14 is centered on -- none of
those are achievable without a real PyTorch training setup and real
GPUs, which this session doesn't have. What's built and tested is
narrower and named precisely: given trained adapter matrices (however
they were produced), merge them in correctly. The test proves this two
ways -- a hand-computed matrix check, and confirmation that the merge
changes a real model's actual output through the real, unmodified forward
pass, not just at the matrix level in isolation.

## Phase 16 — Multi-tenancy and the control plane

**What:** API keys (generated once, stored only as a SHA-256 hash),
per-tenant daily token quotas and per-second rate limits (both enforced
*before* a request runs, not after), and usage metering -- built as its
own FastAPI service, separate from the plain demo API, sitting in front
of the same engine.

**Why it works:** see docs/learning/phase-16.md -- hashing means a leaked
key database is useless on its own; checking quota before doing the work
is what actually stops a request rather than just recording that it
happened.

**What was actually proven, and how:** the plan's own two named
requirements for this phase are both directly tested: hammering one
tenant's key with repeated requests until its quota trips leaves a second,
completely unrelated tenant's usage at exactly zero (not corrupted, not
partially shared) -- and a simulated leaked-key drill (use the key
successfully, revoke it, confirm the very next request with that same key
is rejected) passes.

**What it cost:** tenant and usage state lives in memory for this
session, not a real Postgres database as the plan specifies -- sufficient
to prove the enforcement logic is correct, not yet a production-ready
persistence layer. Abuse controls are limited to the quota/rate-limit
mechanism itself; real content-policy filtering is out of scope (it's
effectively its own project).

## Phase 17 — The playground

**What:** a single, self-contained static HTML page (no build step, no
framework) that talks to the real `/v1/completions` endpoint and runs two
different temperature settings side by side, showing both outputs and
real measured latency for each.

**Why it works:** it's genuinely the same request the curl/Python
quickstart examples on the same page show -- there's no separate,
special-cased code path for the playground versus any other API client.

**What it cost, stated directly:** the plan calls for a Next.js app with
live quantization-level and speculative-decode toggles. Node was actually
available while building this; the deviation was a judgment call, not a
limitation -- see docs/learning/phase-17.md. What's built is a plain
static page (simpler, zero dependencies, fully readable in one sitting)
comparing temperature settings, which genuinely is wired end to end right
now. Comparing quantization levels or speculative decoding live isn't
possible yet because the API doesn't currently expose more than one
servable model configuration to switch between -- that's real, named,
future backend work, not something this page fakes with a toggle that
does nothing.

## Phase 18 — Launch, users, and operations

**What is real:** the engine builds and runs correctly inside a real
Docker container (verified on this machine, catching two real bugs in
the process -- see docs/correctness.md), with Prometheus metrics wired
to a real `/metrics` endpoint and a docker-compose stack tying the
engine, Prometheus, and Grafana together.

**What is explicitly, deliberately NOT done, and why that's stated this
plainly:** there is no public deployment, no real users, no uptime
history, no incident, and no retention data. Fabricating any of these
would be a different kind of dishonesty than every other named gap in
this project (an untrained model, a CPU proxy for a GPU measurement) --
those are honest substitutes for something real; a fake user count or
invented postmortem would just be a fabricated claim. See
docs/learning/phase-18.md for the full reasoning. `docs/postmortems/`
contains a template only, with no incident in it, because there has been
none.

## Phase 21 — Synthetic prefix-cache measurement and local status

**What:** a small C++ executable runs a fixed, seeded cache workload against
the actual `PagedKVCache`. It creates one 16-token shared system prompt,
warms four 20-token prompt variants, then starts 80 conversations from those
variants and makes each one write one to four divergent tokens. The program
counts a prefix-block hit only when a logical prefix-block lookup reuses an
already materialized physical cache block. In the checked workload, it
measured 248 hits in 252 lookups (98.41%) and 80 copy-on-write events. The
API also serves `/status`, a plain HTML page that asks `/status/data` for the
same real Prometheus counters exposed at `/metrics`.

**Why it works:** `PagedSequence::Fork` increments the reference count of
each existing prefix block instead of allocating it again, so counting those
references directly measures the sharing the cache actually performed. Each
variant ends mid-block; the first divergent write therefore has to take the
cache's copy-on-write branch, proving the workload exercises both sharing and
the safe split afterward. The status data reads Prometheus collector samples,
not a second set of application counters, so the status page and `/metrics`
cannot drift into reporting different values.

**What it cost:** the hit rate is intentionally not a general claim about
users, the playground, or a deployed service. It describes one favorable,
named synthetic workload and will change with its prompt-sharing pattern.
The status page also avoids p99, QPS, uptime, and cost figures because this
local process has neither the traffic nor the measurement history needed to
support them. Streaming requests are not included in its completion/latency
counters because those counters were not previously wired for the streaming
path; the page names the scope rather than quietly implying otherwise.
