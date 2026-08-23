#include "executor/batch.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

TEST(PadSequences, PadsShorterSequenceToMatchLongest) {
  std::vector<std::vector<int32_t>> sequences = {{1, 2, 3}, {4, 5}};
  std::vector<int64_t> valid_lengths;
  int64_t max_len;

  std::vector<int32_t> padded =
      PadSequences(sequences, /*pad_token=*/0, &valid_lengths, &max_len);

  EXPECT_EQ(max_len, 3);
  EXPECT_EQ(valid_lengths[0], 3);
  EXPECT_EQ(valid_lengths[1], 2);
  // First sentence, unchanged: [1, 2, 3]
  EXPECT_EQ(padded[0], 1);
  EXPECT_EQ(padded[1], 2);
  EXPECT_EQ(padded[2], 3);
  // Second sentence, padded with the filler token at the end: [4, 5, 0]
  EXPECT_EQ(padded[3], 4);
  EXPECT_EQ(padded[4], 5);
  EXPECT_EQ(padded[5], 0);
}

}  // namespace
}  // namespace kiln
