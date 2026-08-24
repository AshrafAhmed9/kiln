# Phase 7 — derivation notes (the CUDA port)

## Honest starting point

This phase's code was written on a machine with no NVIDIA GPU at all, then
validated on Kaggle. Revision `dc792e1` compiled with NVCC 12.8 on a Tesla
P100-PCIE-16GB (sm_60); all four raw-CUDA CPU-vs-GPU checks (RMSNorm,
greedy argmax, RoPE, and attention) passed, as did the complete 57-test
CTest suite. That proves this narrow claim: these raw kernels build and
match their CPU references on small inputs on that hardware.

It does **not** prove they are fast, robust over production shapes, or used
by the model executor. No Nsight profile or throughput measurement exists,
and the Triton RoPE implementation had not yet been exercised at that point.
The P100's older sm_60 architecture also means the result should not be
generalized to newer T4/A10/A100 hardware without rerunning it there.

The same Kaggle P100 later ran the Triton RoPE check after pinning a
CUDA-12.1 PyTorch wheel that still includes Pascal kernels. It matched the
CPU reference on three positions and two heads. Kaggle's default current
PyTorch wheel did not support this P100, so that compatibility pin is part
of the reproducible notebook setup rather than a claim that all PyTorch
versions support sm_60.

Nsight Compute was invoked on the raw attention test in the same notebook.
The test itself passed, but Kaggle denied access to GPU performance counters
with `ERR_NVGPUCTRPERM`. That means no profiler metric was collected; enabling
that permission or using a different GPU environment is an external
requirement, not a source-code failure.

## Device-resident model prefill (Kaggle version 14)

Revision `80bf15e` added `CudaModel`: it copies one model's weights to device
memory once, keeps temporary activations on device, reuses one cuBLAS handle,
and composes the existing CUDA RMSNorm, RoPE, attention, residual/SwiGLU, and
GEMM operations into a complete uncached forward pass. On Kaggle's T4 pair,
the full CTest suite passed 60/60, including `CudaModel.FullPrefillMatchesCpuReference`.
That test compares every logit for a two-layer random model against
`Model::Forward` at `1e-4` tolerance. It is end-to-end correctness evidence
for the narrow prefill contract, not a throughput claim or a real-checkpoint
benchmark.

The next committed extension adds an executor-owned contiguous GPU KV cache
and a conditional pybind interface. Those changes were not part of version 14
and must not be treated as GPU-validated until their queued remote run passes.

## T4 kernel comparison (Kaggle version 9)

A later Kaggle run received two Tesla T4s (sm_75, CUDA 12.8). All 57 CTest
checks passed, and Triton RoPE again matched the small CPU reference. This
finally exercised ADR-007's one direct raw-CUDA-versus-Triton comparison:
the same FP32 RoPE rotation was event-timed on one T4 at the
SmolLM2-135M-shaped workload of 512 tokens, 9 heads, and head dimension 64.
Each implementation used 100 warm-up launches, then 21 samples of 1,000
launches; allocation, Python setup, and CPU timing were excluded.

| implementation | median time / launch | effective bandwidth |
| --- | ---: | ---: |
| raw CUDA | 0.011774 ms | 200.38 GB/s |
| Triton | 0.020402 ms | 115.64 GB/s |

For this one unfused kernel and shape, the straightforward raw CUDA version
was faster. That is a measured comparison, not evidence that raw CUDA is
always faster, nor an end-to-end model/serving number. Nsight still failed
with the same Kaggle counter-permission error, so this result explains no
hardware-counter-level cause.

## Why a GPU changes how attention has to be written, even though the math is identical

On a CPU, one core computes one number at a time, in whatever order is
convenient. A GPU instead runs thousands of tiny threads at once, and it
only goes fast if those threads read memory in a coordinated pattern
("coalesced" access) and share fast, on-chip memory instead of constantly
reaching out to slow, off-chip memory. So a GPU attention kernel isn't just
"the same loop, but parallel" -- it has to be restructured so that nearby
threads read nearby memory at the same time, and intermediate results
(like the running maximum and running sum used in a numerically stable
softmax) get kept in fast on-chip memory and combined across threads using
a "warp reduction" (a hardware-supported way for a small group of threads
to combine their values without going through slow shared memory at all).

## Why RMSNorm is a clean teaching example for warp reductions

RMSNorm needs one number per row: the average of every value in that row,
squared. On a GPU, many threads each hold one piece of that row, and a
warp reduction is the standard trick for combining all of their partial
sums into one final sum efficiently, without any thread having to wait on
a slow round-trip to shared memory. This is a simpler, smaller version of
exactly the same coordination problem attention has (combining many
threads' partial results into one answer), which is why it's a good first
kernel to actually understand the pattern on before tackling attention.

## Why RoPE and the "everything else" kernels are Triton, not raw CUDA

Raw CUDA is worth writing by hand for the handful of kernels where the
performance and the learning both live (attention, RMSNorm, the sampling
kernel). Beyond those, writing every remaining small kernel (rotation,
residual addition, small fused steps) by hand in raw CUDA teaches
diminishing returns -- the 8th hand-written elementwise kernel doesn't
teach anything the 1st one didn't, and it's still surface area someone
could ask about in an interview with no extra insight behind the answer.
Triton (a Python-embedded language that compiles down to GPU code) covers
that long tail with a fraction of the ceremony, and it's what PyTorch
itself actually compiles into internally -- using it is current practice,
not a shortcut taken to avoid real work. RoPE gets written both ways (once
by hand in raw CUDA, once in Triton) specifically so there's a real,
measured comparison to point to, instead of just asserting Triton is fine.
