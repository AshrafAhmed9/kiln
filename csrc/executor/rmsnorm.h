#pragma once
#include <cstdint>

namespace kiln {

// RMSNorm (Llama's norm, simpler than LayerNorm -- no mean-subtraction, no
// bias): for each of the n_rows rows of length dim, out = x / rms(x) *
// weight, where rms(x) = sqrt(mean(x_i^2) + eps). See
// docs/learning/phase-02.md for why skipping the mean-subtraction is a
// deliberate design choice in Llama-family models, not a shortcut here.
void RmsNorm(const float* x, const float* weight, float* out, int64_t n_rows,
             int64_t dim, float eps);

}  // namespace kiln
