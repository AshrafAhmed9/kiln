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
        // Each number in a head is paired with the number half the head's
        // width away from it (position j pairs with position j+half), and
        // that pair gets rotated together by an angle that depends on the
        // token's position in the sentence and on which pair (j) this is.
        // Llama pairs numbers this way (far apart); some other models pair
        // neighbors (j and j+1) instead. The code still runs either way if
        // you pick the wrong pairing -- it just quietly disagrees with the
        // reference model, which is exactly the kind of bug the parity
        // checks exist to catch.
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
