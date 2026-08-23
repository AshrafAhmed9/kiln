#pragma once
#include <cstdint>

namespace kiln {

// SwiGLU MLP: out = (silu(x @ Wg) * (x @ Wu)) @ Wd. silu(z) = z * sigmoid(z)
// -- a smooth alternative to ReLU that Llama-family models use in place of
// a plain gated linear unit. Wg, Wu are [ffn_hidden, dim] (HF layout,
// out_features-first); Wd is [dim, ffn_hidden].
void SwiGlu(const float* x, const float* w_gate, const float* w_up,
            const float* w_down, float* out, int64_t n_rows, int64_t dim,
            int64_t ffn_hidden);

}  // namespace kiln
