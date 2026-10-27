#pragma once
#include <cstdint>
#include <vector>

#include "kv/paged_kv_cache.h"

namespace kiln {

// The same attention as Attention() in attention.h, but reading keys and
// values from a paged cache. Paging only changes where the numbers sit in
// memory, so the result must be identical. paged_attention_test.cpp checks
// that.
void PagedAttention(const float* q, const PagedKVCache& cache, int64_t layer,
                    const std::vector<int64_t>& block_table, float* out,
                    int64_t seq_len, int64_t kv_len, int64_t n_heads,
                    int64_t n_kv_heads, int64_t head_dim,
                    int64_t query_start_pos);

}  // namespace kiln
