#include "executor/attention.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

// With one token, one head, and one key/value head, attention has nothing
// to choose between -- it can only look at itself. So the output must
// simply equal that one value vector, unchanged. This is the simplest
// possible case and a good sanity check before trusting anything more
// complicated.
TEST(Attention, SingleTokenReturnsItsOwnValue) {
  float q[2] = {1.0f, 0.0f};
  float k[2] = {1.0f, 0.0f};
  float v[2] = {7.0f, 9.0f};
  float out[2];

  Attention(q, k, v, out, /*seq_len=*/1, /*kv_len=*/1, /*n_heads=*/1,
            /*n_kv_heads=*/1, /*head_dim=*/2, /*query_start_pos=*/0);

  EXPECT_NEAR(out[0], 7.0f, 1e-4f);
  EXPECT_NEAR(out[1], 9.0f, 1e-4f);
}

// Causal masking means "you can't look at the future." So if we compute
// attention for token 0 out of a 2-token sentence, changing token 1's
// value should have no effect on token 0's output at all -- token 0 isn't
// allowed to see it. This is the property that makes generation
// consistent: today's answer doesn't retroactively change because of a
// word written after it.
TEST(Attention, EarlierTokenIsUnaffectedByLaterToken) {
  float q[4] = {1.0f, 0.0f, 1.0f, 0.0f};   // 2 tokens, head_dim=2
  float k[4] = {1.0f, 0.0f, 1.0f, 0.0f};
  float v_first[4] = {7.0f, 9.0f, 100.0f, 100.0f};
  float v_second[4] = {7.0f, 9.0f, -5.0f, -5.0f};  // only token 1's value differs
  float out_first[4];
  float out_second[4];

  Attention(q, k, v_first, out_first, /*seq_len=*/2, /*kv_len=*/2, 1, 1, 2, 0);
  Attention(q, k, v_second, out_second, 2, 2, 1, 1, 2, 0);

  // Row 0 (token 0's output) must be identical in both runs.
  EXPECT_NEAR(out_first[0], out_second[0], 1e-4f);
  EXPECT_NEAR(out_first[1], out_second[1], 1e-4f);
}

// valid_kv_len lets a padded, unused position be ignored entirely, as if
// it weren't there. So attending with 2 real keys plus 1 ignored padding
// key should give exactly the same answer as attending with just the 2
// real keys and no padding at all.
TEST(Attention, PaddingBeyondValidLengthIsIgnored) {
  float q[2] = {1.0f, 0.0f};
  float k_two_real[4] = {1.0f, 0.0f, 1.0f, 0.0f};
  float v_two_real[4] = {2.0f, 2.0f, 4.0f, 4.0f};
  float out_no_padding[2];
  Attention(q, k_two_real, v_two_real, out_no_padding, 1, 2, 1, 1, 2, 5);

  float k_with_padding[6] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  float v_with_padding[6] = {2.0f, 2.0f, 4.0f, 4.0f, 999.0f, 999.0f};
  float out_with_padding[2];
  Attention(q, k_with_padding, v_with_padding, out_with_padding, 1, 3, 1, 1, 2,
            5, /*valid_kv_len=*/2);

  EXPECT_NEAR(out_no_padding[0], out_with_padding[0], 1e-4f);
  EXPECT_NEAR(out_no_padding[1], out_with_padding[1], 1e-4f);
}

}  // namespace
}  // namespace kiln
