#pragma once
#include <cstdint>

namespace kiln {

// Causal grouped-query attention, one batch item at a time.
//
// q is [seq_len, n_heads*head_dim]; k and v are [kv_len, n_kv_heads*head_dim]
// (kv_len covers everything already cached plus this call's new tokens).
//
// Normally each attention head has its own key and value. "Grouped-query
// attention" saves memory by having several query heads share one key/value
// head instead -- head h simply looks up which shared key/value head it
// belongs to (h divided by the group size, rounding down). This is the
// entire idea: fewer key/value heads to store means a much smaller cache,
// at a small cost in how precise the attention can be.
//
// query_start_pos says where in the whole sentence this batch of queries
// starts (0 the first time we read a prompt, or however many tokens we've
// already generated when we're adding one new word after that). A token
// is only allowed to look at earlier or same-position tokens, never later
// ones -- that's what "causal" means here, and it's what stops the model
// from cheating by peeking at words it's supposed to be predicting.
//
// valid_kv_len, when set, tells the function to ignore any key past that
// point, as if it didn't exist. That's how padded sentences in the same
// batch (Phase 4) avoid attending to each other's filler padding, without
// needing a whole separate code path for the padded case.
void Attention(const float* q, const float* k, const float* v, float* out,
               int64_t seq_len, int64_t kv_len, int64_t n_heads,
               int64_t n_kv_heads, int64_t head_dim, int64_t query_start_pos,
               int64_t valid_kv_len = -1);

}  // namespace kiln
