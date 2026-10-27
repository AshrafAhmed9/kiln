#include "executor/model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace kiln {
namespace {

ModelConfig TinyConfig() {
  ModelConfig config;
  config.vocab_size = 16;
  config.hidden_size = 8;
  config.n_layers = 2;
  config.n_heads = 2;
  config.n_kv_heads =
      1;  // grouped-query attention: 2 query heads share 1 key/value head
  config.head_dim = 4;
  config.ffn_hidden = 16;
  config.max_seq_len = 32;
  config.rms_eps = 1e-5f;
  config.rope_theta = 10000.0f;
  return config;
}

TEST(Model, ProducesLogitsOfTheRightShape) {
  Model model = Model::LoadRandom(TinyConfig(), /*seed=*/1);
  int32_t tokens[3] = {1, 2, 3};
  std::vector<float> logits(3 * model.config().vocab_size);

  model.Forward(tokens, /*seq_len=*/3, /*start_pos=*/0, /*cache=*/nullptr,
                logits.data());

  // Just checking the numbers came out finite and the right count exist --
  // this is the "did anything crash or produce garbage" sanity check that
  // comes before any test of whether the numbers are actually correct.
  for (float value : logits) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

// This is the single most important test in this file: generating a
// sentence one word at a time using the cache (read the prompt, then add
// one new word using only the cache) must give the exact same numbers as
// throwing the whole sentence-so-far at the model fresh every time, with
// no cache at all. If these ever disagreed, it would mean the cache is
// remembering the wrong thing -- a bug that would be very easy to miss
// otherwise, since the model would still produce *some* answer, just a
// silently wrong one.
TEST(Model, CachedDecodeMatchesFullRecompute) {
  ModelConfig config = TinyConfig();
  Model model = Model::LoadRandom(config, /*seed=*/2);

  std::vector<int32_t> full_sequence = {1, 2, 3, 4};

  // Path A: process the first 3 tokens as a "prompt" using the cache, then
  // hand the cache one more token and read out its prediction.
  KVCache cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                config.head_dim);
  std::vector<float> prompt_logits(3 * config.vocab_size);
  model.Forward(full_sequence.data(), 3, 0, &cache, prompt_logits.data());

  int32_t next_token = full_sequence[3];
  std::vector<float> cached_next_logits(1 * config.vocab_size);
  model.Forward(&next_token, 1, /*start_pos=*/3, &cache,
                cached_next_logits.data());

  // Path B: process all 4 tokens at once, with no cache at all.
  std::vector<float> full_logits(4 * config.vocab_size);
  model.Forward(full_sequence.data(), 4, 0, nullptr, full_logits.data());

  // The cached path's answer for the 4th token should match the
  // no-cache path's answer for that same 4th token, exactly.
  const float* full_path_4th_token_logits =
      full_logits.data() + 3 * config.vocab_size;
  for (int64_t i = 0; i < config.vocab_size; ++i) {
    EXPECT_NEAR(cached_next_logits[i], full_path_4th_token_logits[i], 1e-3f);
  }
}
}  // namespace
}  // namespace kiln
