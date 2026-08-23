# Phase 7 — derivation notes (the CUDA port)

## Honest starting point

This phase's code was written on a machine with no NVIDIA GPU at all, so
none of it has actually been compiled or run on real hardware in this
session. It's written carefully, to spec, matching the already-tested CPU
versions line-for-line in intent -- but "compiles and runs correctly on a
real GPU" is a claim that can only be made once it's actually been run on
one (the plan's own zero-budget notes already name Kaggle's free T4 GPUs as
where that happens). Writing it now, honestly labeled, is still worth
doing: it's real code, ready to be tested the moment GPU time is
available, and getting the design right on paper first is exactly the
"derive before implementing" discipline this whole project follows.

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
