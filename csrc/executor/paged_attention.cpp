#include "executor/paged_attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kiln {

void PagedAttention(const float* q, const PagedKVCache& cache, int64_t layer,
                    const std::vector<int64_t>& block_table, float* out,
                    int64_t seq_len, int64_t kv_len, int64_t n_heads,
                    int64_t n_kv_heads, int64_t head_dim,
                    int64_t query_start_pos) {
  int64_t group_size = n_heads / n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  int64_t block_size = cache.block_size();

  // The caller has to have handed us enough blocks to actually cover
  // kv_len positions -- if not, reading past the end of block_table below
  // would be silent undefined behavior instead of a clear error pointing
  // at the actual mistake (the caller under-sized the block table).
  int64_t blocks_needed = (kv_len + block_size - 1) / block_size;
  if (static_cast<int64_t>(block_table.size()) < blocks_needed) {
    throw std::invalid_argument(
        "PagedAttention: block_table has fewer blocks than kv_len needs");
  }

  // Translates a logical key position (0, 1, 2, ...) into "which physical
  // block, which slot within that block" -- this one line is the entire
  // difference between paged and contiguous attention. Everything else
  // below is identical to the contiguous version.
  auto locate = [&](int64_t position, int64_t* block_id, int64_t* slot) {
    *block_id = block_table[position / block_size];
    *slot = position % block_size;
  };

  std::vector<float> scores(kv_len);

  for (int64_t h = 0; h < n_heads; ++h) {
    int64_t kv_head = h / group_size;

    for (int64_t i = 0; i < seq_len; ++i) {
      const float* q_row = q + i * n_heads * head_dim + h * head_dim;
      int64_t query_pos = query_start_pos + i;
      int64_t last_visible = std::min(query_pos, kv_len - 1);

      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j <= last_visible; ++j) {
        int64_t block_id, slot;
        locate(j, &block_id, &slot);
        const float* k_row = cache.K(layer, block_id) +
                             slot * n_kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }

      float sum_exp = 0.0f;
      for (int64_t j = 0; j <= last_visible; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum_exp += scores[j];
      }

      float* out_row = out + i * n_heads * head_dim + h * head_dim;
      for (int64_t d = 0; d < head_dim; ++d) out_row[d] = 0.0f;

      for (int64_t j = 0; j <= last_visible; ++j) {
        int64_t block_id, slot;
        locate(j, &block_id, &slot);
        const float* v_row = cache.V(layer, block_id) +
                             slot * n_kv_heads * head_dim + kv_head * head_dim;
        float weight = scores[j] / sum_exp;
        for (int64_t d = 0; d < head_dim; ++d) out_row[d] += weight * v_row[d];
      }
    }
  }
}

}  // namespace kiln
