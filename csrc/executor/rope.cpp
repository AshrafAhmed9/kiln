#include "executor/rope.h"

#include <cmath>

namespace kiln {

void ApplyRope(float* x, const int64_t* positions, int64_t n_tokens,
               int64_t n_heads, int64_t head_dim, float theta) {
  int64_t half = head_dim / 2;

  for (int64_t t = 0; t < n_tokens; ++t) {
    float pos = static_cast<float>(positions[t]);
    for (int64_t h = 0; h < n_heads; ++h) {
      float* head = x + t * n_heads * head_dim + h * head_dim;
      for (int64_t j = 0; j < half; ++j) {
        // Number j is paired with number j + half and the pair is rotated
        // by an angle that grows with the word's position. Llama pairs
        // numbers this way; other models pair neighbors instead. Using the
        // wrong pairing still runs but gives wrong answers, which the
        // Hugging Face comparison would catch.
        float freq = std::pow(theta, -2.0f * static_cast<float>(j) /
                                         static_cast<float>(head_dim));
        float angle = pos * freq;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        float x0 = head[j];
        float x1 = head[j + half];
        head[j] = x0 * cos_a - x1 * sin_a;
        head[j + half] = x0 * sin_a + x1 * cos_a;
      }
    }
  }
}

}  // namespace kiln
