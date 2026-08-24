// RoPE on the GPU, written by hand in raw CUDA. UNVERIFIED IN THIS SESSION
// -- see docs/learning/phase-07.md. This kernel exists specifically to be
// benchmarked against csrc/kernels/triton/rope.py (ADR-007's "one kernel,
// both ways" comparison) -- everything past this one is Triton, not raw
// CUDA, and this is the evidence for why that split is reasonable rather
// than just asserted.
#include <cuda_runtime.h>
#include <cstdint>

namespace kiln {

// One thread handles one (token, head, pair-of-numbers) triple -- RoPE
// naturally parallelizes this way, since every pair of numbers is rotated
// completely independently of every other pair.
__global__ void RopeKernel(float* x, const int64_t* positions,
                            int64_t n_heads, int64_t head_dim, float theta) {
  int64_t token = blockIdx.x;
  int64_t head = blockIdx.y;
  int64_t j = threadIdx.x;  // which pair, within this head
  int64_t half = head_dim / 2;
  if (j >= half) return;

  float pos = static_cast<float>(positions[token]);
  float freq = powf(theta, -2.0f * static_cast<float>(j) /
                               static_cast<float>(head_dim));
  float angle = pos * freq;
  float cos_a = cosf(angle);
  float sin_a = sinf(angle);

  float* pair = x + token * n_heads * head_dim + head * head_dim;
  float x0 = pair[j];
  float x1 = pair[j + half];
  pair[j] = x0 * cos_a - x1 * sin_a;
  pair[j + half] = x0 * sin_a + x1 * cos_a;
}

void ApplyRopeCuda(float* x, const int64_t* positions, int64_t n_tokens,
                    int64_t n_heads, int64_t head_dim, float theta) {
  dim3 grid(n_tokens, n_heads);
  int threads_per_block = static_cast<int>(head_dim / 2);
  RopeKernel<<<grid, threads_per_block>>>(x, positions, n_heads, head_dim,
                                           theta);
}

}  // namespace kiln
