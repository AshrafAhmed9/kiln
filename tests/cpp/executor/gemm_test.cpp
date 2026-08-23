#include "executor/gemm.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

// A tiny 2x2 times 2x2 multiply, checked by hand:
// [[1,2],[3,4]] * [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
//                                = [[19, 22], [43, 50]]
TEST(Gemm, MatchesHandComputedResult) {
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};

  Gemm(a, b, c, 2, 2, 2);

  EXPECT_FLOAT_EQ(c[0], 19);
  EXPECT_FLOAT_EQ(c[1], 22);
  EXPECT_FLOAT_EQ(c[2], 43);
  EXPECT_FLOAT_EQ(c[3], 50);
}

// GemmBT takes B already "flipped on its side" (each output row stored as
// one row of B, instead of one column) -- this is the layout real model
// weight files use. If we flip our small example's B by hand, GemmBT
// should give the exact same answer as plain Gemm did above.
TEST(Gemm, GemmBTMatchesGemmWithTransposedInput) {
  float a[4] = {1, 2, 3, 4};
  float b_transposed[4] = {5, 7, 6, 8};  // this is b's columns written as rows
  float c[4] = {0, 0, 0, 0};

  GemmBT(a, b_transposed, c, 2, 2, 2);

  EXPECT_FLOAT_EQ(c[0], 19);
  EXPECT_FLOAT_EQ(c[1], 22);
  EXPECT_FLOAT_EQ(c[2], 43);
  EXPECT_FLOAT_EQ(c[3], 50);
}

}  // namespace
}  // namespace kiln
