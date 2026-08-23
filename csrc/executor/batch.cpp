#include "executor/batch.h"

#include <algorithm>

namespace kiln {

std::vector<int32_t> PadSequences(
    const std::vector<std::vector<int32_t>>& sequences, int32_t pad_token,
    std::vector<int64_t>* valid_lengths, int64_t* out_max_len) {
  int64_t max_len = 0;
  for (const auto& seq : sequences) {
    max_len = std::max<int64_t>(max_len, static_cast<int64_t>(seq.size()));
  }

  std::vector<int32_t> padded(sequences.size() * max_len, pad_token);
  valid_lengths->resize(sequences.size());

  for (size_t b = 0; b < sequences.size(); ++b) {
    const auto& seq = sequences[b];
    (*valid_lengths)[b] = static_cast<int64_t>(seq.size());
    std::copy(seq.begin(), seq.end(),
              padded.begin() + static_cast<int64_t>(b) * max_len);
  }

  *out_max_len = max_len;
  return padded;
}

}  // namespace kiln
