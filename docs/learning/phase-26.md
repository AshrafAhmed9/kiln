# Phase 26 — the actual GEMM that makes INT8 fast, and the hardware line it hits

## What Phase 9 actually built, versus what "INT8 is fast" requires

Phase 9's quantizer produces real INT8 weights with real per-channel
scales, and the accuracy/memory tradeoff table is real. But every GEMM
Phase 9 ever ran was: dequantize the INT8 weights back to FP32, then run
an ordinary FP32 matmul. That's a legitimate way to test a quantizer's
*accuracy*, and it's an honest way to get the *memory* win (a quantized
checkpoint on disk and in RAM is smaller), but it can never show a *speed*
win, because the actual multiplication happening in the CPU is FP32 either
way. The GEMM never got faster; only the storage did.

## The real thing: accumulate in INT32, scale once

A true INT8 GEMM keeps both operands as INT8 all the way through the
multiply-accumulate step, summing into an INT32 accumulator (127×127 =
16,129, summed over a whole row -- nowhere near overflowing 32 bits for
any realistic hidden size), and only converts back to a real number once,
at the very end, scaled by that row's and that column's quantization
scale. `Int8GemmBT` (CPU) and `Int8GemmBTCuda` (GPU, via cuBLAS's
`CUBLAS_COMPUTE_32I` path) both do exactly this. This is the version of
"INT8 matmul" that GPU tensor cores are actually built to accelerate --
Phase 9's dequantize-then-FP32 approach was never going to benefit from
that hardware at all, no matter how well-optimized.

## Why the activation side has to be quantized fresh, every call

Weights are quantized once, at load time -- they don't change between
calls. Activations (the input to a layer) are different every single
forward pass, so there's no "quantize once" option for them; they get
quantized right before the GEMM that consumes them, using the exact same
`QuantizeInt8PerChannel` function Phase 9 already built and tested for
weights. Treating "the input row" as just another kind of row to quantize
-- rather than writing a second, separate quantization function -- is why
no new quantization code exists for this feature at all.

## The hardware line this hit, and why that's worth recording precisely

cuBLAS's `CUBLAS_COMPUTE_32I` INT8 tensor-core path requires compute
capability 6.1 or newer. Every free GPU session reachable this session
landed on hardware one step below that line: Kaggle assigned a Tesla P100
(compute capability 6.0) on every attempt, and five separate attempts at a
GCE T4 (compute capability 7.5, comfortably above the line) across
different zones all failed with `ZONE_RESOURCE_POOL_EXHAUSTED` before a
VM was even created (see `docs/correctness.md`). The code is real, it
compiles, its CPU reference is fully tested, and its CUDA path has been
run on real hardware -- what's missing is specifically a GPU new enough to
exercise the one code path this whole feature exists for. The test and
benchmark detect this and report `SKIPPED` with the exact reason, rather
than either crashing without explanation or silently reporting a
misleading "0 tests failed" that would hide the fact that the actual
speed number was never measured.

## What would resolve this

Any GPU with compute capability 6.1+ -- a Kaggle T4 session (7.5), a GCE
T4 once capacity frees up, or genuinely any consumer/cloud GPU from the
last several years. The moment one of those is available, the exact same
code and benchmark already committed to this repo produces the real
FP32-vs-INT8 speed number with no further changes needed.
