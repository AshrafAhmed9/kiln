#include "executor/rmsnorm.h"

#include <gtest/gtest.h>

#include <cmath>

namespace kiln {
namespace {

// RMSNorm rescales a row of numbers so its "typical size" is 1, then
// multiplies each number by a learned weight. By hand, for the row [3, 4]
// with weight [1, 1] and a tiny eps: the typical size is
// sqrt((3*3 + 4*4) / 2) = sqrt(12.5) = 3.5355..., so the result should be
// [3/3.5355, 4/3.5355] = [0.8485, 1.1314].
TEST(RmsNorm, MatchesHandComputedResult) {
  float x[2] = {3.0f, 4.0f};
  float weight[2] = {1.0f, 1.0f};
  float out[2];

  RmsNorm(x, weight, out, /*n_rows=*/1, /*dim=*/2, /*eps=*/1e-5f);

  EXPECT_NEAR(out[0], 0.8485f, 1e-3f);
  EXPECT_NEAR(out[1], 1.1314f, 1e-3f);
}

// If the weight isn't all 1s, the result should just be the same
// normalized numbers, each additionally multiplied by its own weight.
TEST(RmsNorm, AppliesWeightAfterNormalizing) {
  float x[2] = {3.0f, 4.0f};
  float weight[2] = {2.0f, 0.5f};
  float out[2];

  RmsNorm(x, weight, out, 1, 2, 1e-5f);

  EXPECT_NEAR(out[0], 0.8485f * 2.0f, 1e-3f);
  EXPECT_NEAR(out[1], 1.1314f * 0.5f, 1e-3f);
}

}  // namespace
}  // namespace kiln
