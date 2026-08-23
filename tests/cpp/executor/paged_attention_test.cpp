#include "executor/paged_attention.h"

#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

#include "executor/attention.h"

namespace kiln {
namespace {

// Paging is only supposed to change WHERE the key/value numbers live in
// memory, never what attention computes from them. This test builds the
// exact same key/value numbers two ways -- once as one contiguous buffer,
// once spread across several small paged blocks -- and checks that
// PagedAttention and the plain Attention() function agree exactly.
TEST(PagedAttention, MatchesContiguousAttentionExactly) {
  int64_t n_heads = 2, n_kv_heads = 1, head_dim = 2;
  int64_t seq_len = 4;
  int64_t block_size = 2;

  // q has seq_len * n_heads * head_dim = 4 * 2 * 2 = 16 numbers.
  float q[16] = {1, 0, 0, 1, 1, 1, 0.5f, 0.5f, 0.2f, 0.8f, 0.9f, 0.1f, 1, -1, -1, 1};
  // k and v have seq_len * n_kv_heads * head_dim = 4 * 1 * 2 = 8 numbers each.
  float k[8] = {1, 0, 0.5f, 0.5f, 0.3f, 0.7f, -1, 1};
  float v[8] = {2, 2, 4, 4, 6, 6, 8, 8};

  std::vector<float> contiguous_out(seq_len * n_heads * head_dim);
  Attention(q, k, v, contiguous_out.data(), seq_len, seq_len, n_heads,
            n_kv_heads, head_dim, /*query_start_pos=*/0);

  PagedKVCache cache(/*n_layers=*/1, /*num_blocks=*/4, block_size, n_kv_heads,
                     head_dim);
  std::vector<int64_t> block_table;
  for (int64_t pos = 0; pos < seq_len; pos += block_size) {
    int64_t block = cache.AllocateBlock();
    block_table.push_back(block);
    for (int64_t slot = 0; slot < block_size; ++slot) {
      int64_t src_row = pos + slot;
      std::memcpy(cache.K(0, block) + slot * n_kv_heads * head_dim,
                  k + src_row * n_kv_heads * head_dim,
                  n_kv_heads * head_dim * sizeof(float));
      std::memcpy(cache.V(0, block) + slot * n_kv_heads * head_dim,
                  v + src_row * n_kv_heads * head_dim,
                  n_kv_heads * head_dim * sizeof(float));
    }
  }

  std::vector<float> paged_out(seq_len * n_heads * head_dim);
  PagedAttention(q, cache, /*layer=*/0, block_table, paged_out.data(),
                 seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                 /*query_start_pos=*/0);

  for (size_t i = 0; i < contiguous_out.size(); ++i) {
    EXPECT_NEAR(contiguous_out[i], paged_out[i], 1e-5f);
  }
}

// If a caller hands PagedAttention fewer blocks than kv_len actually
// needs, reading past the end of the block table would otherwise be
// silent undefined behavior -- this must be a clear, catchable error
// instead.
TEST(PagedAttention, TooFewBlocksForKvLenThrows) {
  int64_t n_heads = 1, n_kv_heads = 1, head_dim = 2;
  PagedKVCache cache(1, 4, /*block_size=*/2, n_kv_heads, head_dim);
  std::vector<int64_t> block_table = {cache.AllocateBlock()};  // only 1 block: room for 2 kv positions

  float q[2] = {1, 0};
  std::vector<float> out(2);

  EXPECT_THROW(
      PagedAttention(q, cache, 0, block_table, out.data(), /*seq_len=*/1,
                     /*kv_len=*/4, n_heads, n_kv_heads, head_dim, 0),
      std::invalid_argument);
}

}  // namespace
}  // namespace kiln
