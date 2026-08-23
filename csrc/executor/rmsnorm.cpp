#include "executor/rmsnorm.h"

#include <cmath>

namespace kiln {

void RmsNorm(const float* x, const float* weight, float* out, int64_t n_rows,
             int64_t dim, float eps) {
  for (int64_t r = 0; r < n_rows; ++r) {
    const float* row = x + r * dim;
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < dim; ++i) sum_sq += row[i] * row[i];
    float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);

    float* out_row = out + r * dim;
    for (int64_t i = 0; i < dim; ++i) {
      out_row[i] = (row[i] / rms) * weight[i];
    }
  }
}

}  // namespace kiln
