#include "executor/sampler.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

TEST(Sampler, GreedyPicksHighestScore) {
  float logits[4] = {1.0f, 5.0f, 3.0f, -2.0f};
  EXPECT_EQ(GreedyArgmax(logits, 4), 1);
}

// With temperature set to 0, Sample should behave exactly like
// GreedyArgmax -- "always pick the best word" is what temperature 0 means.
TEST(Sampler, ZeroTemperatureIsGreedy) {
  float logits[4] = {1.0f, 5.0f, 3.0f, -2.0f};
  SamplerConfig config;
  config.temperature = 0.0f;
  std::mt19937 rng(42);

  int32_t result = Sample(logits, 4, config, /*previous_tokens=*/{}, rng);
  EXPECT_EQ(result, 1);
}

// With top_k set to 1, only the single best word is allowed through, no
// matter what the temperature is -- so the answer should still always be
// the best-scoring word, every single time, even with randomness enabled.
TEST(Sampler, TopKOneAlwaysPicksBestWord) {
  float logits[4] = {1.0f, 5.0f, 3.0f, -2.0f};
  SamplerConfig config;
  config.temperature = 1.0f;
  config.top_k = 1;
  std::mt19937 rng(7);

  for (int i = 0; i < 10; ++i) {
    int32_t result = Sample(logits, 4, config, {}, rng);
    EXPECT_EQ(result, 1);
  }
}

// Two runs that start with the exact same random-number state should make
// the exact same random choice -- this is the "replayable" property the
// whole project depends on for debugging.
TEST(Sampler, SameSeedGivesSameResult) {
  float logits[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // all equally likely
  SamplerConfig config;
  config.temperature = 1.0f;

  std::mt19937 rng_a(123);
  std::mt19937 rng_b(123);
  int32_t result_a = Sample(logits, 4, config, {}, rng_a);
  int32_t result_b = Sample(logits, 4, config, {}, rng_b);

  EXPECT_EQ(result_a, result_b);
}

}  // namespace
}  // namespace kiln
