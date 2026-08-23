#include "executor/lora.h"

#include <stdexcept>

#include <gtest/gtest.h>

#include "executor/model.h"

namespace kiln {
namespace {

// A hand-computable case: a 2x2 weight, rank 1, so lora_b @ lora_a is a
// single outer product we can check by hand.
// lora_b = [2, 3] (a column, out_features=2, rank=1)
// lora_a = [4, 5] (a row, rank=1, in_features=2)
// lora_b @ lora_a = [[2*4, 2*5], [3*4, 3*5]] = [[8, 10], [12, 15]]
// scale = 0.5 -> delta = [[4, 5], [6, 7.5]]
TEST(MergeLoraAdapter, MatchesHandComputedOuterProduct) {
  float weight[4] = {1, 1, 1, 1};
  float lora_b[2] = {2, 3};
  float lora_a[2] = {4, 5};

  MergeLoraAdapter(weight, lora_a, lora_b, /*out_features=*/2,
                   /*in_features=*/2, /*rank=*/1, /*scale=*/0.5f);

  EXPECT_FLOAT_EQ(weight[0], 1 + 4.0f);
  EXPECT_FLOAT_EQ(weight[1], 1 + 5.0f);
  EXPECT_FLOAT_EQ(weight[2], 1 + 6.0f);
  EXPECT_FLOAT_EQ(weight[3], 1 + 7.5f);
}

TEST(MergeLoraAdapter, ZeroScaleLeavesWeightUnchanged) {
  float weight[4] = {1, 2, 3, 4};
  float lora_b[2] = {10, 20};
  float lora_a[2] = {30, 40};

  MergeLoraAdapter(weight, lora_a, lora_b, 2, 2, 1, /*scale=*/0.0f);

  EXPECT_FLOAT_EQ(weight[0], 1);
  EXPECT_FLOAT_EQ(weight[1], 2);
  EXPECT_FLOAT_EQ(weight[2], 3);
  EXPECT_FLOAT_EQ(weight[3], 4);
}

// Proves the merge actually takes effect through the real, already-tested
// forward pass -- not just at the matrix level in isolation. A model that
// has had a (deliberately large, so its effect isn't lost in the noise of
// small random weights) adapter merged into one layer's wq must produce
// DIFFERENT logits than the same model before the merge.
TEST(MergeLoraAdapter, ChangesModelOutputThroughARealForwardPass) {
  ModelConfig config;
  config.vocab_size = 16;
  config.hidden_size = 8;
  config.n_layers = 2;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.head_dim = 4;
  config.ffn_hidden = 16;
  config.max_seq_len = 32;
  config.rms_eps = 1e-5f;
  config.rope_theta = 10000.0f;

  Model model = Model::LoadRandom(config, /*seed=*/11);
  int32_t tokens[3] = {1, 2, 3};
  std::vector<float> logits_before(3 * config.vocab_size);
  model.Forward(tokens, 1, 3, nullptr, 0, nullptr, logits_before.data());

  int64_t q_dim = config.n_heads * config.head_dim;  // 8
  int64_t rank = 2;
  std::vector<float> lora_a(rank * config.hidden_size, 1.0f);  // [rank, hidden_size]
  std::vector<float> lora_b(q_dim * rank, 1.0f);               // [q_dim, rank]
  model.MergeLoraIntoLayer(/*layer_idx=*/0, "wq", lora_a.data(), lora_b.data(),
                           rank, /*scale=*/2.0f);

  std::vector<float> logits_after(3 * config.vocab_size);
  model.Forward(tokens, 1, 3, nullptr, 0, nullptr, logits_after.data());

  bool any_different = false;
  for (size_t i = 0; i < logits_before.size(); ++i) {
    if (logits_before[i] != logits_after[i]) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST(MergeLoraAdapter, UnknownMatrixNameThrows) {
  ModelConfig config;
  config.vocab_size = 8;
  config.hidden_size = 4;
  config.n_layers = 1;
  config.n_heads = 1;
  config.n_kv_heads = 1;
  config.head_dim = 4;
  config.ffn_hidden = 8;
  config.max_seq_len = 8;
  Model model = Model::LoadRandom(config, 1);

  std::vector<float> lora_a(4, 1.0f), lora_b(4, 1.0f);
  EXPECT_THROW(
      model.MergeLoraIntoLayer(0, "not_a_real_matrix", lora_a.data(),
                               lora_b.data(), 1, 1.0f),
      std::invalid_argument);
}

}  // namespace
}  // namespace kiln
