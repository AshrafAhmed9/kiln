#include "executor/lora.h"

namespace kiln {

void MergeLoraAdapter(float* weight, const float* lora_a, const float* lora_b,
                      int64_t out_features, int64_t in_features,
                      int64_t rank, float scale) {
  for (int64_t out = 0; out < out_features; ++out) {
    for (int64_t in = 0; in < in_features; ++in) {
      float delta = 0.0f;
      for (int64_t r = 0; r < rank; ++r) {
        delta += lora_b[out * rank + r] * lora_a[r * in_features + in];
      }
      weight[out * in_features + in] += scale * delta;
    }
  }
}

}  // namespace kiln
