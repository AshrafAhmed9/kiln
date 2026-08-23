# Why is decode slow? A roofline analysis

This is the question every inference-systems interview asks. The answer
doesn't need a GPU to derive — it's arithmetic on the model's own
dimensions, checked against what a GPU can actually do. What it *does*
need a GPU for is measuring the real, achieved bandwidth utilization
against the theoretical peak — that part is stated honestly as pending
(see the last section) rather than guessed at.

## The roofline model, in one paragraph

Every piece of hardware has two limits: how many floating-point
operations it can do per second (FLOPs), and how many bytes it can move
between memory and compute per second (bandwidth). A computation's
**arithmetic intensity** is FLOPs performed per byte moved. If a
computation's intensity is *below* the hardware's own FLOPs-per-byte
ratio (the "ridge point"), the hardware sits idle waiting for data —
**memory-bound**. If it's *above* the ridge point, the hardware sits
busy computing and idles waiting on nothing — **compute-bound**. The
whole question "why is decode slow" is really "which side of the ridge
point is decode on, and why."

## Worked arithmetic, using realistic dimensions

Using representative published dimensions for a small (~1B-parameter)
Llama-architecture model — hidden size 2048, 16 layers, 32 query heads,
8 KV heads (head dim 64), FFN size 8192, FP16 weights (2 bytes/weight) —
not this project's own toy test config, which is deliberately tiny and
would give a misleadingly small answer.

For one linear layer's weight matrix `[K, N]`, processing `M` tokens at
once in one matmul:

- **FLOPs** = `2 · M · K · N` (a multiply and an add per weight, per token)
- **Bytes moved** (weights, the dominant cost) = `K · N · 2` (FP16)
- **Arithmetic intensity** = FLOPs / bytes = `2·M·K·N / (2·K·N)` = **`M`**

That's the whole result, and it's worth sitting with: **arithmetic
intensity is approximately equal to the number of tokens processed in
one pass** (for a weight-bound linear layer — attention adds K/V cache
traffic on top, discussed below, which only makes the memory-bound case
stronger).

### Prefill: many tokens at once → compute-bound

A typical prompt is hundreds to thousands of tokens, all processed
together in one forward pass. `M` is large (say 512–2048), so arithmetic
intensity is ~512–2048 FLOP/byte. A representative inference GPU's ridge
point (peak FP16 FLOPs ÷ peak memory bandwidth):

- **T4**: ~65 TFLOPS ÷ 320 GB/s ≈ **203 FLOP/byte**
- **A100**: ~312 TFLOPS ÷ 2039 GB/s ≈ **153 FLOP/byte**

A 512+ token prompt sails past either ridge point — **prefill is
compute-bound**, and stays that way for any realistic prompt length. The
GPU's math units are the bottleneck, not its memory bus.

### Decode: one new token at a time → memory-bound

Autoregressive decode processes exactly one new token per forward pass
per sequence (`M = 1`, absent batching). Arithmetic intensity ≈ **1
FLOP/byte** — one to two *orders of magnitude* below either GPU's ridge
point. **Decode is deeply memory-bandwidth-bound**: the GPU spends nearly
all of its time waiting to read the model's weights from memory, not
computing with them. This is the standard, well-known industry result,
and the arithmetic above is the actual reason for it, not an assertion.

**Attention makes this worse, not better, as context grows.** Beyond the
weight matrices, decode's attention step also has to read every cached
key and value for every token seen so far — bytes that grow linearly with
context length, for the exact same one token's worth of new compute. A
longer conversation doesn't just cost more memory (Phase 8's whole
reason for existing) — it drives arithmetic intensity *down further*,
making decode progressively more memory-bound as a conversation gets
longer.

### Where continuous batching fits into this

Continuous batching's speed win (measured in `BENCHMARKS.md` /
`bench/run_benchmarks.py`) isn't from changing any single request's
arithmetic intensity — it's from **raising `M` for the same one weight
read** by processing several sequences' one-new-token-each requests
together in a single batched pass. Batching 32 concurrent decode
requests together turns `M` from 1 to 32 — still likely below the ridge
point for a large model, but 32× fewer separate weight reads for the
same total work, which is exactly why serving throughput scales with
batch size even though single-request latency doesn't improve.

## What this predicts about where Kiln's own levers help

- **Quantization (Phase 9)** attacks decode's actual bottleneck directly:
  fewer bytes to move per weight (INT8: 4× fewer than FP32) means less
  time waiting on memory bandwidth, even with identical FLOP count. This
  is *why* INT8 is worth doing for decode specifically, independent of
  any GPU tensor-core speedup on the compute side.
- **Paged attention (Phase 8)** doesn't change arithmetic intensity at
  all — it changes how much *can run at once* within a fixed memory
  budget, which is a capacity lever, not a compute-vs-memory lever. Both
  matter; they're different axes.
- **Speculative decoding (Phase 10)** effectively raises `M` for the
  expensive target model's verification pass (checking several draft
  tokens in one forward call, the same trick as batching), which is
  exactly why it helps decode's memory-bound regime specifically — one
  weight read now verifies several tokens instead of one.

## What's honestly still missing

Everything above is analytical — real arithmetic against real model
dimensions, correct regardless of what hardware eventually runs it. What
it is **not** is a *measured* bandwidth-utilization number (what fraction
of a specific GPU's 320 GB/s or 2039 GB/s a real, profiled Kiln decode
step actually achieves). That number needs a real GPU and a real
profiler (Nsight Compute), neither of which is available in this
session — it's deferred to the Kaggle T4 path this project's zero-budget
plan already accounts for (ADR-009), and it belongs here, filled in, the
moment that measurement exists.
