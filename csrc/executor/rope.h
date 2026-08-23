#pragma once
#include <cstdint>

namespace kiln {

// Rotary position embedding, applied in place to Q or K. `x` is
// [n_tokens, n_heads * head_dim]; `positions[i]` is the absolute sequence
// position of token i (not just its index in this call -- this is what
// lets RoPE be applied correctly to a single new token during KV-cached
// decode, where its absolute position is far past `n_tokens`).
// See docs/learning/phase-02.md for the derivation of the per-pair
// rotation angle.
void ApplyRope(float* x, const int64_t* positions, int64_t n_tokens,
               int64_t n_heads, int64_t head_dim, float theta);

}  // namespace kiln
