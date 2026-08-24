// The greedy (argmax) sampling kernel on the GPU. It passed a small
// CPU-vs-GPU check on Kaggle's P100 -- see docs/learning/phase-07.md. Only greedy argmax is written
// as a hand-written CUDA kernel here (the third of the three headline
// kernels this project commits to writing by hand, per ADR-007); the
// randomized settings (temperature, top-k, top-p, repetition penalty) use
// the exact same reduction pattern as this kernel but were left as CPU-side
// work in this pass, since the interesting, defend-in-an-interview idea --
// "how do many threads agree on the single largest value across a whole
// row, without racing each other?" -- is already fully present here.
#include <cuda_runtime.h>
#include <cstdint>
#include <float.h>

namespace kiln {

// Every thread scans a share of the row and remembers the best (index,
// score) pair it personally saw. Then, the same warp-shuffle trick used in
// rmsnorm.cu combines every thread's candidate down to one final answer --
// except here we have to carry an index along with each value, not just a
// number, so the comparison has to look at both together at each step.
__global__ void ArgmaxKernel(const float* logits, int64_t vocab_size,
                              int32_t* out_index) {
  float best_score = -FLT_MAX;
  int32_t best_index = 0;

  for (int64_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    if (logits[i] > best_score) {
      best_score = logits[i];
      best_index = static_cast<int32_t>(i);
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    float other_score = __shfl_down_sync(0xffffffff, best_score, offset);
    int32_t other_index = __shfl_down_sync(0xffffffff, best_index, offset);
    if (other_score > best_score) {
      best_score = other_score;
      best_index = other_index;
    }
  }

  __shared__ float warp_best_scores[32];
  __shared__ int32_t warp_best_indices[32];
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (lane_id == 0) {
    warp_best_scores[warp_id] = best_score;
    warp_best_indices[warp_id] = best_index;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float final_score = warp_best_scores[0];
    int32_t final_index = warp_best_indices[0];
    int num_warps = (blockDim.x + 31) / 32;
    for (int w = 1; w < num_warps; ++w) {
      if (warp_best_scores[w] > final_score) {
        final_score = warp_best_scores[w];
        final_index = warp_best_indices[w];
      }
    }
    *out_index = final_index;
  }
}

void ArgmaxCuda(const float* logits, int64_t vocab_size, int32_t* out_index) {
  int threads_per_block = 256;
  ArgmaxKernel<<<1, threads_per_block>>>(logits, vocab_size, out_index);
}

}  // namespace kiln
