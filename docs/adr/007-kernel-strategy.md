# ADR-007: Kernel strategy — raw CUDA for the headline three, Triton for the tail

**Status:** Accepted (Phase 7 will fill in measured numbers)

**Decision:** Attention, RMSNorm, and the sampling kernel are hand-written in
raw CUDA C++, each profiled naive→optimized with Nsight Compute. Everything
else GPU-side (RoPE, residual adds, elementwise/fused epilogues, dequant
paths) is written in Triton. RoPE is implemented **both** ways once, and
benchmarked, so "why Triton?" has a data-backed answer instead of a
defensive one.

**Why:** the 8th hand-written elementwise kernel teaches nothing the 1st
didn't and is pure defense surface with no learning behind it. The headline
three are where CUDA depth is genuinely earned and worth defending
line-by-line. Triton is what PyTorch itself compiles into — using it is
current practice.

**Honesty rule:** the README and resume state precisely which kernels are
raw CUDA and which are Triton. Never imply the whole kernel layer is
hand-written CUDA — see project memory `kiln-minimal-defensible-code` on
precision as a defensibility requirement.
