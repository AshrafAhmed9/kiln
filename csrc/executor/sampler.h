#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace kiln {

// Settings for picking the next word from the model's scores (one score per
// possible word). Greedy decoding (temperature 0) always picks the top
// word, so the same input always gives the same output. The Hugging Face
// comparison relies on that.
struct SamplerConfig {
  float temperature =
      1.0f;  // higher = more random; 0 means "always pick the best one"
  int32_t top_k = 0;   // if > 0, only consider the top_k highest-scoring words
  float top_p = 1.0f;  // if < 1, only consider the smallest set of top words
                       // whose probabilities add up to at least top_p (this is
                       // "nucleus sampling")
  float repetition_penalty =
      1.0f;  // > 1 makes already-used words less likely to repeat
};

// Returns the single highest-scoring word ("greedy" decoding).
int32_t GreedyArgmax(const float* logits, int64_t vocab_size);

// Picks one word using the settings above. The same rng state always gives
// the same word, so a run can be repeated exactly.
int32_t Sample(const float* logits, int64_t vocab_size,
               const SamplerConfig& config,
               const std::vector<int32_t>& previous_tokens, std::mt19937& rng);

}  // namespace kiln
