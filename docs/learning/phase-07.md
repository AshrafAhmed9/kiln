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
and the Triton RoPE implementation remains uncompiled and unrun. The
P100's older sm_60 architecture also means the result should not be
generalized to newer T4/A10/A100 hardware without rerunning it there.

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
