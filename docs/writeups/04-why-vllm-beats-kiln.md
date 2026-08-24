# Why vLLM beats Kiln here — named directly, not glossed over

Kiln implements the *ideas* behind several of vLLM's headline features
(paged attention, continuous batching, quantization, speculative
decoding) and proves each one's core algorithm is correct. It does not
match vLLM's actual production performance, and pretending otherwise
would be the same kind of dishonesty this project refuses everywhere
else. Here is specifically why, ranked roughly by how much of the gap
each one explains.

## 1. FlashAttention-style tiling — the single biggest gap

Kiln's attention (both the tested CPU version and the raw CUDA kernel,
which passed a small CPU-vs-GPU check on a Kaggle P100) computes and holds the *entire* row of
attention scores for a query before turning it into an output — read
every key, store every score, then reduce. FlashAttention's actual
contribution is an IO-aware *tiling* scheme: process keys and values in
small blocks that fit in on-chip memory, maintaining a running (online)
softmax so the full score matrix is never materialized in slow memory at
all. For long contexts, this is the difference between attention's memory
traffic scaling with context length squared versus scaling linearly.
Kiln's kernel is a straightforward, correct, *un-tiled* implementation —
exactly the difference between "the algorithm is right" and "the
algorithm is fast at scale."

## 2. CUDA graphs — every decode step pays launch overhead Kiln doesn't hide

A decode step is a fixed, repeating sequence of kernel launches. Real
serving engines capture that sequence once into a CUDA graph and replay
it, paying the CPU-side launch/dispatch overhead exactly once instead of
once per step. Kiln launches everything fresh, every single decode step,
through the same Python↔C++ boundary measured in Phase 4. That boundary
overhead was measured at ~0.06% of one decode step on this project's toy
CPU config — genuinely small *here*, but CUDA graph capture is precisely
the production technique for keeping that category of overhead small at
real GPU decode speeds, and Kiln doesn't do it.

## 3. Chunked prefill — Kiln processes a whole prompt in one blocking pass

A long prompt in Kiln's scheduler runs to completion in prefill before
any decode step for *any* request (including other users) can proceed —
one long prompt blocks everyone else's decode progress for its entire
duration. Splitting a long prefill into smaller chunks, interleaved with
other requests' decode steps between chunks, is what keeps decode latency
low for concurrent users even when someone submits a very long prompt.
Kiln's scheduler (see `kiln_py/scheduler/`) has a real chunked-prefill
policy implemented and compared against plain FCFS (see
`bench/scheduler_policy_comparison.py`), but it isn't the default, and
vLLM's version is substantially more battle-tested under adversarial
mixes of prompt lengths.

## 4. Scheduler policy sophistication

Kiln's scheduler admits requests first-come-first-served, reserving each
request's full worst-case memory footprint up front (a real, deliberate,
tested design — see `docs/learning/phase-05.md`). vLLM's scheduler adds
preemption with swapping (moving a request's KV cache out to CPU memory
and back rather than only ever blocking admission), priority-aware
ordering, and iteration-level rebalancing tuned against real production
traffic patterns. Kiln's admission math is provably correct; it isn't
tuned for the traffic patterns a real multi-tenant deployment sees.

## 5. Kernel maturity, generally

Beyond attention specifically: vLLM ships fused dequantization-GEMM
kernels tuned per quantization format (AWQ, GPTQ, Marlin-style kernels),
warp-level-tuned paged-attention variants for different head
configurations, and years of profiling-driven micro-optimization. Kiln's
raw CUDA kernels (Phase 7) are written once and mirror the already-tested
CPU structure for correctness. They have compiled and passed small
reference tests on a P100, but have not been tuned against a profiler,
measured for throughput, or integrated into the model executor.

## 6. Speculative decoding: single draft chain vs. tree verification

Kiln's speculative decoding verifies one linear sequence of draft-model
guesses per round. vLLM (and the broader research literature past the
original papers this project implements) supports *tree*-structured
speculation — verifying several candidate continuations at once, not
just one guessed sequence — which raises the achievable acceptance rate
per expensive target-model call. Kiln implements the provably-exact
single-chain case; the tree-based generalization is real, additional
algorithmic work this project doesn't attempt.

## The honest summary

Every gap above is a *known, named* piece of engineering — not a mystery,
not an oversight discovered by an interviewer. That's the actual
difference this list is trying to draw: overclaiming ("Kiln is
competitive with vLLM") reads as not understanding what production
inference engines actually do. Naming the specific gap, and why it
matters, reads as someone who built enough of the real thing to know
exactly where the remaining engineer-years went.
