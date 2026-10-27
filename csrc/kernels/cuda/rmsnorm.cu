// RMSNorm on the GPU. Same math as csrc/executor/rmsnorm.cpp, with the work
// split across threads.
#include <cuda_runtime.h>
#include <cstdint>

namespace kiln {

// One GPU block (a group of cooperating threads) handles one row. Each
// thread adds up the squares of its share of the numbers. Then the threads
// combine their partial sums with a "warp shuffle", a hardware instruction
// that lets the 32 threads of a warp pass values to each other directly,
// without going through slower shared memory.
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
