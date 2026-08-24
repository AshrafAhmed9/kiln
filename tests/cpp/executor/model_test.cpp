#include "executor/model.h"

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "executor/batch.h"

namespace kiln {
namespace {

ModelConfig TinyConfig() {
  ModelConfig config;
  config.vocab_size = 16;
  config.hidden_size = 8;
  config.n_layers = 2;
  config.n_heads = 2;
  config.n_kv_heads = 1;  // grouped-query attention: 2 query heads share 1 key/value head
  config.head_dim = 4;
  config.ffn_hidden = 16;
  config.max_seq_len = 32;
  config.rms_eps = 1e-5f;
  config.rope_theta = 10000.0f;
  return config;
}

// A cache only makes sense for one sentence at a time in this project's
// current design -- passing both a cache and more than one sentence at
// once must fail loudly rather than silently ignore the cache the caller
// asked for (see the comment at the top of Model::Forward).
TEST(Model, PassingACacheWithMoreThanOneSequenceThrows) {
  ModelConfig config = TinyConfig();
  Model model = Model::LoadRandom(config, /*seed=*/4);
  KVCache cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                config.head_dim);
  int32_t tokens[4] = {1, 2, 3, 4};
  std::vector<float> logits(4 * config.vocab_size);

  EXPECT_THROW(
      model.Forward(tokens, /*batch_size=*/2, /*seq_len=*/2, nullptr, 0,
                    &cache, logits.data()),
      std::invalid_argument);
}

TEST(Model, ProducesLogitsOfTheRightShape) {
  Model model = Model::LoadRandom(TinyConfig(), /*seed=*/1);
  int32_t tokens[3] = {1, 2, 3};
  std::vector<float> logits(3 * model.config().vocab_size);

  model.Forward(tokens, /*batch_size=*/1, /*seq_len=*/3, nullptr,
                /*start_pos=*/0, /*cache=*/nullptr, logits.data());

  // Just checking the numbers came out finite and the right count exist --
  // this is the "did anything crash or produce garbage" sanity check that
  // comes before any test of whether the numbers are actually correct.
  for (float value : logits) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

// This is the single most important test in Phase 3: generating a
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
  model.Forward(full_sequence.data(), 1, 3, nullptr, 0, &cache,
                prompt_logits.data());

  int32_t next_token = full_sequence[3];
  std::vector<float> cached_next_logits(1 * config.vocab_size);
  model.Forward(&next_token, 1, 1, nullptr, /*start_pos=*/3, &cache,
                cached_next_logits.data());

  // Path B: process all 4 tokens at once, with no cache at all.
  std::vector<float> full_logits(4 * config.vocab_size);
  model.Forward(full_sequence.data(), 1, 4, nullptr, 0, nullptr,
                full_logits.data());

  // The cached path's answer for the 4th token should match the
  // no-cache path's answer for that same 4th token, exactly.
  const float* full_path_4th_token_logits =
      full_logits.data() + 3 * config.vocab_size;
  for (int64_t i = 0; i < config.vocab_size; ++i) {
    EXPECT_NEAR(cached_next_logits[i], full_path_4th_token_logits[i], 1e-3f);
  }
}

TEST(Model, BatchedCachedDecodeMatchesIndependentCachedSequences) {
  ModelConfig config = TinyConfig();
  Model model = Model::LoadRandom(config, /*seed=*/21);
  KVCache first_cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                      config.head_dim);
  KVCache second_cache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                       config.head_dim);
  int32_t first_prompt[] = {1, 2};
  int32_t second_prompt[] = {3, 4, 5};
  std::vector<float> ignored_first(2 * config.vocab_size);
  std::vector<float> ignored_second(3 * config.vocab_size);
  model.Forward(first_prompt, 1, 2, nullptr, 0, &first_cache,
                ignored_first.data());
  model.Forward(second_prompt, 1, 3, nullptr, 0, &second_cache,
                ignored_second.data());

  int32_t decode_tokens[] = {6, 7};
  int64_t positions[] = {2, 3};
  KVCache* caches[] = {&first_cache, &second_cache};
  std::vector<float> batched_logits(2 * config.vocab_size);
  model.ForwardDecodeBatch(decode_tokens, 2, positions, caches,
                           batched_logits.data());

  KVCache first_reference(config.n_layers, config.max_seq_len,
                          config.n_kv_heads, config.head_dim);
  KVCache second_reference(config.n_layers, config.max_seq_len,
                           config.n_kv_heads, config.head_dim);
  model.Forward(first_prompt, 1, 2, nullptr, 0, &first_reference,
                ignored_first.data());
  model.Forward(second_prompt, 1, 3, nullptr, 0, &second_reference,
                ignored_second.data());
  std::vector<float> first_logits(config.vocab_size);
  std::vector<float> second_logits(config.vocab_size);
  model.Forward(decode_tokens, 1, 1, nullptr, 2, &first_reference,
                first_logits.data());
  model.Forward(decode_tokens + 1, 1, 1, nullptr, 3, &second_reference,
                second_logits.data());

  for (int64_t i = 0; i < config.vocab_size; ++i) {
    EXPECT_NEAR(batched_logits[i], first_logits[i], 1e-3f);
    EXPECT_NEAR(batched_logits[config.vocab_size + i], second_logits[i], 1e-3f);
  }
  EXPECT_EQ(first_cache.length(), 3);
  EXPECT_EQ(second_cache.length(), 4);
}

// Phase 4's whole point: running two sentences together, padded into one
// rectangle, should give each sentence exactly the answer it would have
// gotten running alone -- padding must never leak into a real answer.
TEST(Model, BatchedForwardMatchesRunningEachSequenceAlone) {
  ModelConfig config = TinyConfig();
  Model model = Model::LoadRandom(config, /*seed=*/3);

  std::vector<std::vector<int32_t>> sequences = {{1, 2, 3}, {4, 5}};
  std::vector<int64_t> valid_lengths;
  int64_t max_len;
  std::vector<int32_t> padded =
      PadSequences(sequences, /*pad_token=*/0, &valid_lengths, &max_len);

  std::vector<float> batched_logits(2 * max_len * config.vocab_size);
  model.Forward(padded.data(), /*batch_size=*/2, max_len,
                valid_lengths.data(), /*start_pos=*/0, /*cache=*/nullptr,
                batched_logits.data());

  // Run sentence 0 alone and compare its real (non-padded) positions.
  std::vector<float> alone_logits_0(3 * config.vocab_size);
  model.Forward(sequences[0].data(), 1, 3, nullptr, 0, nullptr,
                alone_logits_0.data());
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t i = 0; i < config.vocab_size; ++i) {
      EXPECT_NEAR(batched_logits[row * config.vocab_size + i],
                  alone_logits_0[row * config.vocab_size + i], 1e-3f);
    }
  }

  // Run sentence 1 alone and compare its real (non-padded) positions,
  // which live in the second row-block of the batched output.
  std::vector<float> alone_logits_1(2 * config.vocab_size);
  model.Forward(sequences[1].data(), 1, 2, nullptr, 0, nullptr,
                alone_logits_1.data());
  int64_t second_item_offset = max_len * config.vocab_size;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t i = 0; i < config.vocab_size; ++i) {
      EXPECT_NEAR(
          batched_logits[second_item_offset + row * config.vocab_size + i],
          alone_logits_1[row * config.vocab_size + i], 1e-3f);
    }
  }
}

}  // namespace
}  // namespace kiln
