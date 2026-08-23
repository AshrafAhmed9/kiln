#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace kiln {

// Settings for turning a row of logits (one raw "how likely" score per
// possible next word) into one chosen word. All the knobs below make the
// output more or less random on purpose -- greedy decoding (temperature 0,
// no top_k/top_p) always gives the exact same output for the exact same
// input, which is what the parity tests rely on.
struct SamplerConfig {
  float temperature = 1.0f;    // higher = more random; 0 means "always pick the best one"
  int32_t top_k = 0;           // if > 0, only consider the top_k highest-scoring words
  float top_p = 1.0f;          // if < 1, only consider the smallest set of top words whose
                                // probabilities add up to at least top_p (this is "nucleus sampling")
  float repetition_penalty = 1.0f;  // > 1 makes already-used words less likely to repeat
};

// Always returns the single highest-scoring word. This is what "greedy"
// decoding means, and it's the version we compare against the reference
// model, since it has no randomness to account for.
int32_t GreedyArgmax(const float* logits, int64_t vocab_size);

// Picks one word according to the settings above, using `rng` for any
// randomness needed. Passing the same rng state twice gives the same
// answer twice -- this is what makes generation replayable, which matters
// both for debugging and for the project's determinism promise.
int32_t Sample(const float* logits, int64_t vocab_size,
               const SamplerConfig& config,
               const std::vector<int32_t>& previous_tokens, std::mt19937& rng);

}  // namespace kiln
