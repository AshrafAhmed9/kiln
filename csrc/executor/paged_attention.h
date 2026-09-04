#pragma once
#include <cstdint>
#include <vector>

#include "kv/paged_kv_cache.h"

namespace kiln {

// Attention, reading keys and values out of a paged cache instead of one
// contiguous buffer. Mathematically this must produce exactly the same
// answer as Attention() (csrc/executor/attention.cpp) given the same
// underlying numbers -- paging is purely a matter of *where the numbers
// live in memory*, never a change to the attention math itself. The
// `Model.PagedAttentionMatchesContiguousAttention`-style test (Phase 8's
// "paged vs contiguous parity" requirement) is what proves that.
void PagedAttention(const float* q, const PagedKVCache& cache, int64_t layer,
                    const std::vector<int64_t>& block_table, float* out,
                    int64_t seq_len, int64_t kv_len, int64_t n_heads,
                    int64_t n_kv_heads, int64_t head_dim,
                    int64_t query_start_pos);

}  // namespace kiln
