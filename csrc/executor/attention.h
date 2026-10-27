#pragma once
#include <cstdint>

namespace kiln {

// Causal grouped-query attention: each new word looks back at the earlier
// words and takes a weighted average of what it finds there.
//
//   q   [seq_len, n_heads * head_dim]    what each new word is looking for
//   k,v [kv_len, n_kv_heads * head_dim]  what every word so far offers
//                                        (kv_len = cached words + new words)
//
// "Grouped-query" means several query heads share one key/value head, which
// makes the KV cache smaller. Head h uses key/value head h / group_size.
//
// query_start_pos is where the first new word sits in the whole text (0 for
// a fresh prompt, or the number of cached words when adding one more word).
// A word may only look at itself and earlier words, never later ones. That
// is what "causal" means, and it stops the model peeking at the answer.
void Attention(const float* q, const float* k, const float* v, float* out,
               int64_t seq_len, int64_t kv_len, int64_t n_heads,
               int64_t n_kv_heads, int64_t head_dim, int64_t query_start_pos);

}  // namespace kiln
