#pragma once
#include <cstdint>

namespace kiln {

// Adds a trained LoRA adapter into a weight matrix, in place:
// weight += scale * (lora_b @ lora_a).
// weight is [out_features, in_features], lora_a is [rank, in_features], and
// lora_b is [out_features, rank]. Done once up front, so serving the merged
// model costs nothing extra.
void MergeLoraAdapter(float* weight, const float* lora_a, const float* lora_b,
                      int64_t out_features, int64_t in_features, int64_t rank,
                      float scale);

}  // namespace kiln
