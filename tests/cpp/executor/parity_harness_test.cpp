#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "executor/rmsnorm.h"

namespace kiln {
namespace {

// A deliberately wrong RMSNorm: it sums the squared values but forgets to
// divide by how many numbers are in the row before taking the square root.
// This is a realistic, easy-to-make bug (an "off by a factor," not a
// typo), used here only to prove the parity-style tolerance check below
// would actually notice it.
void BuggyRmsNormMissingMeanDivision(const float* x, const float* weight,
                                     float* out, int64_t dim, float eps) {
  float sum_sq = 0.0f;
  for (int64_t i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
  float wrong_rms = std::sqrt(sum_sq + eps);  // missing "/ dim" before the sqrt
  for (int64_t i = 0; i < dim; ++i) out[i] = (x[i] / wrong_rms) * weight[i];
}

// Tests the tolerance check itself. Give it a correct result and a
// deliberately wrong one, and confirm it flags the wrong one. A check that
// always says "close enough" would be useless.
TEST(ParityHarness, ToleranceCheckCatchesADeliberatelyInjectedBug) {
  int64_t dim = 8;
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> weight(dim, 1.0f);
  float eps = 1e-5f;

  std::vector<float> correct(dim);
  RmsNorm(x.data(), weight.data(), correct.data(), 1, dim, eps);

  std::vector<float> buggy(dim);
  BuggyRmsNormMissingMeanDivision(x.data(), weight.data(), buggy.data(), dim,
                                  eps);

  // The project's real per-dtype tolerance is on the order of
  // 1e-3 to 1e-5 for FP32 -- comfortably tighter than the gap this bug
  // produces, which is why a real parity gate would fail it loudly rather
  // than let it through as noise.
  const float real_tolerance = 1e-3f;
  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < dim; ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(correct[i] - buggy[i]));
  }

  EXPECT_GT(max_abs_diff, real_tolerance)
      << "the deliberately broken implementation should disagree with the "
         "correct one by more than the real tolerance -- if this ever "
         "fails, it means the injected bug accidentally produced the same "
         "numbers anyway, which would need a different bug to test with";
}

}  // namespace
}  // namespace kiln
