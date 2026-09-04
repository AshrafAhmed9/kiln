#include "executor/rope.h"

#include <gtest/gtest.h>

#include <cmath>

namespace kiln {
namespace {

// A token at position 0 should come out completely unchanged. The rotation
// angle for position 0 is always 0 * frequency = 0, and rotating by an
// angle of 0 (cos=1, sin=0) is the same as not rotating at all -- so this
// is really testing that the "do nothing at position zero" case works,
// which is an easy thing to get wrong with an off-by-one in the position
// numbering.
TEST(Rope, PositionZeroIsUnchanged) {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // one token, one head, head_dim=4
  int64_t positions[1] = {0};

  ApplyRope(x, positions, /*n_tokens=*/1, /*n_heads=*/1, /*head_dim=*/4,
            /*theta=*/10000.0f);

  EXPECT_NEAR(x[0], 1.0f, 1e-5f);
  EXPECT_NEAR(x[1], 2.0f, 1e-5f);
  EXPECT_NEAR(x[2], 3.0f, 1e-5f);
  EXPECT_NEAR(x[3], 4.0f, 1e-5f);
}

// Rotating always keeps a pair's combined length the same -- rotating a
// point never moves it closer to or further from the center, only around
// it. So no matter what angle position 5 works out to, (x0, x2)'s combined
// length before and after must match. This test doesn't need to know the
// exact angle to be a real check: if the rotation math were wrong (say, a
// sign flipped or a wrong pairing was used), this length would almost
// certainly change.
TEST(Rope, RotationPreservesPairLength) {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float original_len_pair0 = std::sqrt(x[0] * x[0] + x[2] * x[2]);

  int64_t positions[1] = {5};
  ApplyRope(x, positions, 1, 1, 4, 10000.0f);

  float rotated_len_pair0 = std::sqrt(x[0] * x[0] + x[2] * x[2]);
  EXPECT_NEAR(original_len_pair0, rotated_len_pair0, 1e-4f);
}

}  // namespace
}  // namespace kiln
