#include "executor/attention.h"

#include <cmath>
#include <limits>
#include <vector>

namespace kiln {

void Attention(const float* q, const float* k, const float* v, float* out,
               int64_t seq_len, int64_t kv_len, int64_t n_heads,
               int64_t n_kv_heads, int64_t head_dim, int64_t query_start_pos) {
  int64_t group_size = n_heads / n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> scores(kv_len);

  for (int64_t h = 0; h < n_heads; ++h) {
    int64_t kv_head = h / group_size;

    for (int64_t i = 0; i < seq_len; ++i) {
      const float* q_row = q + i * n_heads * head_dim + h * head_dim;
      int64_t query_pos = query_start_pos + i;

      // This word can see itself and everything before it, nothing later.
      int64_t last_visible = query_pos;

      // Step 1: score how well this word's query matches each earlier key.
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j <= last_visible; ++j) {
        const float* k_row = k + j * n_kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }

      // Step 2: turn scores into weights that add up to 1 (softmax). Taking
      // away the biggest score first avoids overflow in exp().
      float sum_exp = 0.0f;
      for (int64_t j = 0; j <= last_visible; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum_exp += scores[j];
      }

      float* out_row = out + i * n_heads * head_dim + h * head_dim;
      for (int64_t d = 0; d < head_dim; ++d) out_row[d] = 0.0f;

      // Step 3: the output is the weighted average of the values.

      for (int64_t j = 0; j <= last_visible; ++j) {
        float weight = scores[j] / sum_exp;
        const float* v_row = v + j * n_kv_heads * head_dim + kv_head * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) out_row[d] += weight * v_row[d];
      }
    }
  }
}

}  // namespace kiln
