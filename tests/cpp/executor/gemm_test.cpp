#include "executor/gemm.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

// [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]], worked out by hand.
// GemmBT takes B flipped on its side, so B's columns are written as rows.
TEST(Gemm, GemmBTMatchesHandComputedResult) {
  float a[4] = {1, 2, 3, 4};
  float b_transposed[4] = {5, 7, 6, 8};
  float c[4] = {0, 0, 0, 0};

  GemmBT(a, b_transposed, c, 2, 2, 2);

  EXPECT_FLOAT_EQ(c[0], 19);
  EXPECT_FLOAT_EQ(c[1], 22);
  EXPECT_FLOAT_EQ(c[2], 43);
  EXPECT_FLOAT_EQ(c[3], 50);
}

}  // namespace
}  // namespace kiln
