#pragma once
#include <cstdint>

namespace kiln {

// RMSNorm keeps each row's numbers at a steady size. For every row of
// length `dim`: out = x / rms(x) * weight, where rms(x) is the square root
// of the average of x squared (plus a tiny eps so it is never zero). This is
// the normalization Llama models use.
void RmsNorm(const float* x, const float* weight, float* out, int64_t n_rows,
             int64_t dim, float eps);

}  // namespace kiln
