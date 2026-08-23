# Phase 12 — derivation notes (tensor parallelism)

## The problem this solves

Some models are simply too big for one GPU's memory to hold at all.
Tensor parallelism splits individual weight matrices across several GPUs,
so each one only ever holds and computes with its own slice -- no single
GPU needs to fit the whole layer.

## Column-parallel vs row-parallel, and why they're paired

There are two natural ways to split a matrix multiplication across
workers ("ranks"):

- **Column-parallel:** split the *output* side of a matrix. Each rank
  computes a different slice of the output columns, using the *same, full*
  input. No communication is needed until the very end, when every rank's
  output slice is simply placed side by side to form the full result.
- **Row-parallel:** split the *input* side (the contraction dimension)
  instead. Each rank only has a slice of the input's numbers and a
  matching slice of the weight matrix, so each rank can only compute a
  partial sum toward the true answer -- every rank's partial result has to
  be added together (an "all-reduce") to get the real, complete output.

Real transformer layers pair these two styles deliberately: the first
matrix multiply in a block (say, projecting into a larger hidden size)
is done column-parallel, and the second one (projecting back down) is
done row-parallel. That pairing means the *only* communication needed for
the whole block is one all-reduce at the very end, instead of one after
every single matrix multiply -- communication between GPUs is slow
compared to the math itself, so minimizing how often it happens is the
entire point of choosing this particular pairing.

## Why this can be proven correct on a CPU, with no real GPUs involved

The actual hard part of real tensor parallelism -- coordinating real GPUs
over a real network connection (NCCL) -- can't be tested without real
GPUs. But the *algorithm* (which numbers get split where, and how they get
recombined) is pure arithmetic, and arithmetic can be checked on a CPU: if
splitting a matrix into N pieces, computing each piece separately, and
recombining them always produces the exact same numbers as just doing the
whole computation at once, that proves the *sharding math* is correct.
What isn't proven this way is that real communication between real GPUs
actually behaves the way this project's simulated "add these pieces
together" stand-in assumes -- that's real, deferred, GPU-dependent work,
named honestly rather than implied to be covered by these CPU-only tests.
