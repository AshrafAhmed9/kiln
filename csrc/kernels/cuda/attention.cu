// Causal grouped-query attention on the GPU. UNVERIFIED IN THIS SESSION --
// see docs/learning/phase-07.md. Deliberately mirrors the already-tested
// CPU version (csrc/executor/attention.cpp) step for step -- compute every
// visible score, find the largest, turn scores into normalized weights,
// then take the weighted average of the values -- just with each step
// spread across threads instead of a single-threaded loop. One GPU block
// handles one (query token, attention head) pair.
#include <cuda_runtime.h>
#include <float.h>

namespace kiln {

__global__ void AttentionKernel(const float* q, const float* k,
                                 const float* v, float* out, int64_t kv_len,
                                 int64_t n_heads, int64_t n_kv_heads,
                                 int64_t head_dim, int64_t query_start_pos,
                                 int64_t valid_kv_len) {
  int64_t query_row = blockIdx.x;
  int64_t head = blockIdx.y;
  int64_t group_size = n_heads / n_kv_heads;
  int64_t kv_head = head / group_size;

  int64_t query_pos = query_start_pos + query_row;
  int64_t max_valid_kv = (valid_kv_len >= 0) ? valid_kv_len : kv_len;
  int64_t last_visible = min(query_pos, max_valid_kv - 1);

  const float* q_row = q + query_row * n_heads * head_dim + head * head_dim;
  float scale = rsqrtf(static_cast<float>(head_dim));

  // Shared memory is fast, on-chip scratch space every thread in this
  // block can see -- we use it to hold this one query's score against
  // every key it's allowed to look at, since every thread needs to read
  // the whole row of scores later (once to find the largest, once to sum
  // exponentials, once to take the weighted average of values).
  extern __shared__ float scores[];

  for (int64_t j = threadIdx.x; j <= last_visible; j += blockDim.x) {
    const float* k_row = k + j * n_kv_heads * head_dim + kv_head * head_dim;
    float dot = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
    scores[j] = dot * scale;
  }
  __syncthreads();

  // Finding the largest score first, and subtracting it before taking an
  // exponential, is what keeps this numerically stable -- without it, a
  // large raw score could overflow when exponentiated. A single thread
  // does this scan; it's a small amount of work compared to the score
  // computation above, so it's not worth parallelizing further.
  __shared__ float max_score;
  __shared__ float sum_exp;
  if (threadIdx.x == 0) {
    float m = -FLT_MAX;
    for (int64_t j = 0; j <= last_visible; ++j) m = fmaxf(m, scores[j]);
    max_score = m;
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j <= last_visible; j += blockDim.x) {
    scores[j] = expf(scores[j] - max_score);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float s = 0.0f;
    for (int64_t j = 0; j <= last_visible; ++j) s += scores[j];
    sum_exp = s;
  }
  __syncthreads();

  float* out_row = out + query_row * n_heads * head_dim + head * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float weighted_sum = 0.0f;
    for (int64_t j = 0; j <= last_visible; ++j) {
      const float* v_row = v + j * n_kv_heads * head_dim + kv_head * head_dim;
      weighted_sum += (scores[j] / sum_exp) * v_row[d];
    }
    out_row[d] = weighted_sum;
  }
}

void AttentionCuda(const float* q, const float* k, const float* v,
                    float* out, int64_t seq_len, int64_t kv_len,
                    int64_t n_heads, int64_t n_kv_heads, int64_t head_dim,
                    int64_t query_start_pos, int64_t valid_kv_len) {
  dim3 grid(seq_len, n_heads);
  int threads_per_block = 128;
  size_t shared_bytes = kv_len * sizeof(float);
  AttentionKernel<<<grid, threads_per_block, shared_bytes>>>(
      q, k, v, out, kv_len, n_heads, n_kv_heads, head_dim, query_start_pos,
      valid_kv_len);
}

}  // namespace kiln
