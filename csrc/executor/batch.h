#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// Takes several sentences (as lists of token ids, possibly different
// lengths) and lines them up into one rectangular block, padding the
// shorter ones with a filler token so every sentence is the same length.
// This is what "batching" means at the input level: instead of running the
// model once per sentence, we run it once for the whole rectangle, which
// makes much better use of the computer's raw math throughput (the same
// reason it's faster to wash ten plates at once than one at a time).
//
// Returns the padded tokens (batch_size * max_len, one sentence after
// another) and fills `valid_lengths` with each sentence's real, unpadded
// length, so the model knows which of the padded positions are real words
// and which are just filler.
std::vector<int32_t> PadSequences(const std::vector<std::vector<int32_t>>& sequences,
                                   int32_t pad_token,
                                   std::vector<int64_t>* valid_lengths,
                                   int64_t* out_max_len);

}  // namespace kiln
