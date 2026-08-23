#include "kv/kv_cache.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

TEST(KVCache, StartsEmpty) {
  KVCache cache(/*n_layers=*/2, /*max_seq_len=*/8, /*n_kv_heads=*/1,
                /*head_dim=*/2);
  EXPECT_EQ(cache.length(), 0);
}

// After writing 3 tokens and moving the cache's length forward by 3, the
// numbers we just wrote should still be sitting exactly where we put them.
TEST(KVCache, AppendedValuesAreReadableAfterAdvance) {
  KVCache cache(1, 8, 1, 2);
  float k[6] = {1, 2, 3, 4, 5, 6};  // 3 tokens, 2 numbers each
  float v[6] = {10, 20, 30, 40, 50, 60};

  cache.Append(0, k, v, 3);
  cache.Advance(3);

  EXPECT_EQ(cache.length(), 3);
  const float* stored_k = cache.K(0);
  for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(stored_k[i], k[i]);
}

// Writing tokens in two separate steps (like a prompt, then one new word
// generated after it) should place the second step's tokens right after
// the first step's, not overwrite them.
TEST(KVCache, SecondAppendGoesAfterFirst) {
  KVCache cache(1, 8, 1, 2);
  float k1[4] = {1, 2, 3, 4};  // 2 tokens
  float v1[4] = {10, 20, 30, 40};
  cache.Append(0, k1, v1, 2);
  cache.Advance(2);

  float k2[2] = {100, 200};  // 1 more token
  float v2[2] = {1000, 2000};
  cache.Append(0, k2, v2, 1);
  cache.Advance(1);

  EXPECT_EQ(cache.length(), 3);
  const float* stored_k = cache.K(0);
  EXPECT_FLOAT_EQ(stored_k[4], 100);
  EXPECT_FLOAT_EQ(stored_k[5], 200);
}

}  // namespace
}  // namespace kiln
