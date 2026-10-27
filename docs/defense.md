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

**Status, upfront:** local development had no NVIDIA GPU, but revision
`dc792e1` was compiled on Kaggle's Tesla P100-PCIE-16GB (sm_60) with NVCC
12.8. The raw-CUDA attention, RMSNorm, greedy-argmax, and RoPE kernels
passed four small CPU-vs-GPU tests; the complete CTest suite passed 57/57.
This is a correctness/build check, not a speed claim. The later T4 run
verified Triton RoPE and a device-resident CUDA prefill path: revision
`80bf15e` passed 60/60 CTest checks, including full CPU-vs-GPU logits for a
small model. Revision `3c02523` later passed 61/61, adding cached GPU decode
parity. The pybind probe, profiler metrics, and end-to-end throughput remain
unverified or unfinished as named below.

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

**Upgraded (Phase 26) — the real speed measurement, and the hardware line
it hit:** everything above only ever ran a dequantize-then-FP32 GEMM, which
can prove accuracy but can never show a speed win. `Int8GemmBT` (CPU) and
`Int8GemmBTCuda` (GPU, via cuBLAS's `CUBLAS_COMPUTE_32I`) accumulate real
INT8xINT8 products in INT32 and scale once at the end -- the actual
computation GPU tensor cores accelerate. The CPU path is tested against
the existing dequantize-then-FP32 path directly. The CUDA path needs
compute capability 6.1+; every free GPU session reached this pass landed
one step short of that (Kaggle: P100, 6.0, on every attempt; five separate
GCE T4 attempts across different zones all failed on
`ZONE_RESOURCE_POOL_EXHAUSTED` before a VM was even created -- see
docs/correctness.md and docs/learning/phase-26.md). The test and benchmark
detect this and report a clear `SKIPPED`, not a false pass or an
unexplained crash. Code, wiring, and CPU verification are done; the one
number still missing is the real GPU speedup, blocked purely on GPU
access, not on anything left to build.

## Phase 14 — Adapter-aware serving (fine-tuning itself out of scope)

**What:** given an already-trained LoRA adapter (two small matrices, A and
B), fold it into a served model's weight matrix once, at load time, so the
rest of the forward pass runs completely unmodified afterward.

**Why it works:** see docs/learning/phase-14.md -- merging once at load
time means there's no separate "adapter-aware" code path in the forward
pass to build and re-verify; the already-tested executor just sees a
slightly different weight matrix.

**What it cost, stated as plainly as possible:** `tools/train_lora.py` now
trains and exports a standard PEFT `q_proj` adapter, and the C++ merge is
available through the narrow Python binding for training tooling. A 20-step
SmolLM2 run on four authored fixture records produced an adapter; this proves
the pipeline, not quality or usefulness. Multi-GPU DDP/FSDP scaling remains
separate work.

**Upgraded (Phase 27) — a real licensed dataset and a real held-out number.**
`tools/prepare_banking77.py` fetches the real BANKING77 intent-classification
CSVs (CC-BY-4.0, PolyAI-LDN) directly, `tools/train_lora.py` ran 1000 real
optimizer steps on a real Kaggle CPU session (loss fell from ~4.5-5.0 over the
first five steps to ~1.4-2.2 over the last five -- genuine learning, not
noise), and `tools/eval_lora_intent.py` scored greedy exact-match accuracy
against 300 held-out validation examples never seen during training: the
untrained base SmolLM2-135M scored **0/300 (0.0%)** -- it doesn't know the
BANKING77 label vocabulary at all -- and the LoRA-adapted model scored
**18/300 (6.0%)**. Stated honestly: 6% exact-match on a 77-way intent
classification task is a real, non-zero, above-random-guessing-baseline-in-
kind result from 1000 steps on one `q_proj` matrix with no hyperparameter
search, not a claim of a production-quality classifier -- the samples in
`lora_eval_result.json` show the adapted model consistently produces
BANKING77-shaped label strings (`new_passcode_requested`, `lost_card`,
`cash_fee`) instead of the base model's non-answers (empty strings, `"i"`),
which is the qualitative signal that the adapter learned the task's output
format even where its answer wasn't an exact match to the specific label.

## Phase 22 — First real Hugging Face comparison

**What:** the oracle tooling now has a reproducible CPU-only route using
`HuggingFaceTB/SmolLM2-135M-Instruct`, a small Llama-family checkpoint. Kiln
loads its BF16 safetensors weights, including tied input/output embeddings,
then compares its final next-token logits with Hugging Face FP32. For the
prompt `Kiln checks its own math.`, the measured maximum absolute difference
was 7.44×10⁻⁵, mean absolute difference was 1.23×10⁻⁵, and both implementations
chose the same top token. A 10,000-string, seeded tokenizer comparison also
now runs against the checkpoint's real tokenizer JSON.

**Why it works:** the checkpoint uses the same Llama tensor names and model
dimensions the C++ executor expects. Some Llama checkpoints omit
`lm_head.weight` because it is tied to the input embedding table; the loader
now deliberately recognizes that format and uses the converted embeddings as
the output head. The comparison sends the exact Hugging Face token IDs to both
models, isolating forward-pass numerics from tokenizer disagreement.

**What it cost:** this is evidence, not a completed parity program. It checks
one prompt's final row rather than every layer or a broad prompt fixture. The
tokenizer gap is now closed for the named fixture: ICU supplies the Unicode
letter, number, punctuation, and whitespace categories that the prior ASCII
splitter could not express, and the seeded 10,000-string run matches 10,000
of 10,000 cases. That adds a native dependency (ADR-014), but does not prove
conformance for every possible tokenizer configuration or replace broad model
numerical parity.

**Upgraded (Phase 28) — a real second checkpoint, all 10 fixture prompts, both
tokenizers.** The 10-prompt fixture (`tools/fixtures/hf_parity_prompts.txt`,
covering short/long prompts, Unicode, punctuation, and multi-token positions)
had existed but never actually been run in full -- only the single-prompt
result above was ever measured. Running it for real against
`HuggingFaceTB/SmolLM2-135M-Instruct`: all 10 prompts top-1 match, worst-case
final-logit max absolute difference 6.63×10⁻⁵. A second, structurally
different checkpoint, `HuggingFaceTB/SmolLM2-360M-Instruct` (hidden size 960,
32 layers, 15 query / 5 KV heads -- grouped-query attention with a different
group ratio than the 135M model), ran the exact same 10-prompt fixture with no
code changes: all 10 top-1 match, worst-case max absolute difference
5.07×10⁻⁵. Its seeded 10,000-string tokenizer comparison also matched 10,000
of 10,000. This is still not exhaustive numerical parity across every prompt
and checkpoint that exists, but it is now real evidence across two different
model shapes and their two different tokenizer vocabularies, not one prompt
against one checkpoint.

## Phase 29 — Per-layer parity becomes a real check, not a printed number

**What:** `tools/hf_parity.py` already computed a per-layer hidden-state
difference against a real Hugging Face reference on every run (via
`Model::forward_hidden_states`), but nothing ever looked at the number --
it just sat in the printed JSON. Added a `--layer-diff-threshold` flag
(default `5e-2`) and a real pass/fail: the script now exits nonzero and
prints which prompts failed if any layer's difference from the reference
exceeds it.

**Why the threshold is 5e-2, not something tighter:** the final-logit
comparison (Phase 22/28) holds to ~10⁻⁵ on both checkpoints, but per-layer
hidden-state differences compound across depth -- FP32-vs-BF16 rounding
in one layer feeds into the next layer's computation instead of being
independent each time. Measured directly on both checkpoints: the worst
per-layer difference was ~2.7×10⁻² (135M) while the worst final-logit
difference stayed ~6×10⁻⁵. A threshold at final-logit precision would fail
every real run for a reason that isn't a bug; 5e-2 is set from the actual
observed accumulation, not picked arbitrarily.

**What it cost:** zero new numerical code -- the per-layer diffing already
existed. The only work was picking a threshold grounded in a real
measurement instead of a guess, and running it against both cached
checkpoints (`SmolLM2-135M-Instruct`, `SmolLM2-360M-Instruct`) to confirm
both pass: 10/10 prompts each. No GPU needed for this phase.

## Phase 30 -- Real GPU numbers: INT8 GEMM speedup and Nsight Compute profiling

**What was blocked, and why:** two numbers had been missing since Phase 7 --
a real INT8-vs-FP32 GEMM speedup measured on GPU hardware, and Nsight
Compute profiler output for the model's own CUDA kernels. Kaggle blocks
Nsight structurally (its shared, multi-tenant notebooks don't grant the
performance-counter access `ncu` requires, regardless of GPU). Eleven+
attempts at a GCP GPU VM across many zones and two GPU types hit either a
`GPUS_ALL_REGIONS` quota of 0 (fixed after a support request) or, once
that was fixed, `ZONE_RESOURCE_POOL_EXHAUSTED` / preemption every time --
a preemptible T4 in `us-central1-b` got reclaimed four separate times
mid-build, once after finally clearing a real `nvcc`/GCC host-compiler
incompatibility (CUDA 11.5's frontend fails to parse Ubuntu 22.04's
patched `std_function.h` under GCC 11.4, not just under GCC 12 as first
suspected).

**What actually closed it:** a personal machine with a real NVIDIA GPU
(GeForce GTX 1660 Ti Mobile, Turing, compute capability 7.5 -- the same
generation as a T4, but a DP4A-based consumer part with no dedicated INT8
tensor cores). CUDA 12.0 was already installed; the only real snag was
the same class of host-compiler mismatch seen on GCP (Ubuntu 24.04 ships
GCC 13 by default, and CUDA 12.0 only supports up to GCC 12) -- fixed by
installing `g++-12`/`gcc-12` alongside the system default and passing
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12` to the CMake configure step,
rather than touching the system's default compiler.

**The INT8 GEMM number:** `kiln_int8_gemm_cuda_benchmark`, run for real on
that hardware: `rows=512 k=576 n=1536 fp32_median_ms=0.488616
int8_median_ms=0.305316 speedup=1.600360` -- **1.60x**, INT8 vs. FP32, on
real silicon. Lower than a datacenter GPU with dedicated INT8 tensor cores
would show (this consumer Turing part accelerates INT8 via DP4A dot-product
instructions, not tensor cores) -- reported exactly as measured, not
adjusted to look better.

**The Nsight Compute output:** `ncu --set default` (this Nsight Compute
version, 2022.4.1, uses the identifier `default` where a newer version
would use `basic` -- the first attempt with `--set basic` silently
collected nothing and printed `No metrics to collect found in sections.`,
which is itself a small, worth-naming gotcha) against
`CudaModel.FullPrefillMatchesCpuReference`, all 33 kernel launches across
a 3-token prefill. Every kernel showed Compute (SM) Throughput and Memory
Throughput both under ~3.5%, and Nsight's own launch-statistics warning
flagged it directly: *"This kernel grid is too small to fill the available
resources on this device, resulting in only 0.0 full waves across all
SMs."* Example, `AttentionKernel`: Duration 3.81us, Elapsed Cycles 4,909,
Compute Throughput 0.90%, Memory Throughput 3.43%. This is an honest,
expected result for a correctness-test-sized fixture (3 tokens), not a
disappointing one -- it's launch-overhead-bound, not compute-bound, which
is precisely the mechanism this project's own answer to "why isn't
everything in C++" already describes: per-iteration overhead is real, but
it isn't a per-token compute cost.

**What it cost:** real time chasing infrastructure (GCP quota, zone
capacity, preemption, two separate nvcc/GCC version incompatibilities)
rather than code -- once on hardware that actually stayed up, both numbers
came out cleanly on the first real attempt, and the CUDA test suite passed
67/67 on real physical hardware for the first time in this project's
history (the earlier 61/61 CPU-only and Kaggle-remote runs never included
a locally-run, root-accessible GPU).
