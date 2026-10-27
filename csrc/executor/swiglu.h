#pragma once
#include <cstdint>

namespace kiln {

// SwiGLU feed-forward block: out = (silu(x @ Wg) * (x @ Wu)) @ Wd, where
// silu(z) = z * sigmoid(z) is a smooth version of ReLU. Wg and Wu are
// [ffn_hidden, dim]; Wd is [dim, ffn_hidden].
void SwiGlu(const float* x, const float* w_gate, const float* w_up,
            const float* w_down, float* out, int64_t n_rows, int64_t dim,
            int64_t ffn_hidden);

}  // namespace kiln
