#pragma once
#include <cstdint>

namespace kiln {

// RoPE (rotary position embedding) stamps each word's position into its
// query or key numbers by rotating pairs of them, so attention can tell how
// far apart two words are. Changes `x` in place.
//
// x is [n_tokens, n_heads * head_dim]. positions[i] is word i's place in the
// whole text, not just in this call. That matters when the cache is in use
// and one new word arrives at, say, position 50.
void ApplyRope(float* x, const int64_t* positions, int64_t n_tokens,
               int64_t n_heads, int64_t head_dim, float theta);

}  // namespace kiln
