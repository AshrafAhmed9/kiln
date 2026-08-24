// RMSNorm on the GPU. UNVERIFIED IN THIS SESSION: written on a machine with
// no NVIDIA GPU, so this has not been compiled or run yet -- see
// docs/learning/phase-07.md and docs/defense.md for the honest status.
// The math is identical to csrc/executor/rmsnorm.cpp; only the way work is
// split across threads is different.
#include <cuda_runtime.h>
#include <cstdint>

namespace kiln {

// One GPU "block" (a group of threads that can cooperate) handles one row.
// Each thread in the block first adds up the squares of the numbers it's
// personally responsible for, and then all the threads in the block
// combine their partial sums together using a "warp shuffle" -- a
// hardware instruction that lets threads within a group of 32 (a "warp")
// pass a value directly to each other without going through slower shared
// memory at all. This is the standard, fast way to answer "what's the sum
// of all these numbers, spread across many threads?" on a GPU.
__global__ void RmsNormKernel(const float* x, const float* weight,
                               float* out, int64_t dim, float eps) {
  int64_t row = blockIdx.x;
  const float* row_in = x + row * dim;
  float* row_out = out + row * dim;

  float thread_sum_sq = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = row_in[i];
    thread_sum_sq += v * v;
  }

  // Combine every thread's partial sum into one total, first within each
  // warp (using the fast shuffle instruction), then across warps (using a
  // small shared-memory buffer, since shuffles only work within one warp).
  for (int offset = 16; offset > 0; offset /= 2) {
    thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
  }

  __shared__ float warp_sums[32];
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) warp_sums[warp_id] = thread_sum_sq;
  __syncthreads();

  __shared__ float total_sum_sq;
  if (threadIdx.x == 0) {
    float total = 0.0f;
    int num_warps = (blockDim.x + 31) / 32;
    for (int w = 0; w < num_warps; ++w) total += warp_sums[w];
    total_sum_sq = total;
  }
  __syncthreads();

  float rms = sqrtf(total_sum_sq / static_cast<float>(dim) + eps);
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    row_out[i] = (row_in[i] / rms) * weight[i];
  }
}

void RmsNormCuda(const float* x, const float* weight, float* out,
                  int64_t n_rows, int64_t dim, float eps) {
  int threads_per_block = 256;
  RmsNormKernel<<<n_rows, threads_per_block>>>(x, weight, out, dim, eps);
}

}  // namespace kiln
