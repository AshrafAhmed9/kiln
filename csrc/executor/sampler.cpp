#include "executor/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace kiln {

int32_t GreedyArgmax(const float* logits, int64_t vocab_size) {
  int32_t best_index = 0;
  float best_score = logits[0];
  for (int64_t i = 1; i < vocab_size; ++i) {
    if (logits[i] > best_score) {
      best_score = logits[i];
      best_index = static_cast<int32_t>(i);
    }
  }
  return best_index;
}

int32_t Sample(const float* logits, int64_t vocab_size,
               const SamplerConfig& config,
               const std::vector<int32_t>& previous_tokens,
               std::mt19937& rng) {
  if (config.temperature <= 0.0f) return GreedyArgmax(logits, vocab_size);

  std::vector<float> scores(logits, logits + vocab_size);

  // If a word has already been used, make it a little less attractive to
  // pick again. A positive score gets divided down; a negative score gets
  // multiplied down (which also makes it smaller) -- either way, a word
  // that's already appeared becomes less likely than it was before.
  if (config.repetition_penalty != 1.0f) {
    for (int32_t used : previous_tokens) {
      float& s = scores[used];
      s = (s > 0.0f) ? s / config.repetition_penalty
                      : s * config.repetition_penalty;
    }
  }

  // Temperature stretches or squashes the differences between scores
  // before we turn them into probabilities. A low temperature makes the
  // best word even more dominant; a high temperature makes the choice more
  // even across many words.
  for (float& s : scores) s /= config.temperature;

  // top_k: keep only the best top_k words, and treat every other word as
  // impossible (by setting its score to negative infinity).
  if (config.top_k > 0 && config.top_k < vocab_size) {
    std::vector<int32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + config.top_k, indices.end(),
        [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
    std::vector<bool> keep(vocab_size, false);
    for (int32_t i = 0; i < config.top_k; ++i) keep[indices[i]] = true;
    for (int64_t i = 0; i < vocab_size; ++i) {
      if (!keep[i]) scores[i] = -std::numeric_limits<float>::infinity();
    }
  }

  // Turn scores into real probabilities (softmax): exponentiate each score
  // and divide by the total, so everything adds up to 1 and can be treated
  // as "percent chance of picking this word."
  float max_score = *std::max_element(scores.begin(), scores.end());
  std::vector<float> probs(vocab_size);
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp(scores[i] - max_score);
    sum += probs[i];
  }
  for (float& p : probs) p /= sum;

  // top_p ("nucleus sampling"): sort words from most to least likely, and
  // keep adding words to our shortlist until their probabilities add up to
  // at least top_p. Every word left out of the shortlist gets its
  // probability set to zero, then we rescale so the shortlist alone adds
  // up to 1 again.
  if (config.top_p < 1.0f) {
    std::vector<int32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

    float cumulative = 0.0f;
    size_t cutoff = 0;
    for (; cutoff < indices.size(); ++cutoff) {
      cumulative += probs[indices[cutoff]];
      if (cumulative >= config.top_p) {
        ++cutoff;
        break;
      }
    }
    std::vector<float> filtered(vocab_size, 0.0f);
    float kept_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
      filtered[indices[i]] = probs[indices[i]];
      kept_sum += probs[indices[i]];
    }
    for (float& p : filtered) p /= kept_sum;
    probs = filtered;
  }

  std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
  return dist(rng);
}

}  // namespace kiln
