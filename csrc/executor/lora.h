#pragma once
#include <cstdint>

namespace kiln {

// Folds a trained LoRA adapter into a weight matrix, in place:
// weight += scale * (lora_b @ lora_a). weight is [out_features, in_features]
// (the same out-features-first layout every other weight in this project
// uses); lora_a is [rank, in_features]; lora_b is [out_features, rank].
// See docs/learning/phase-14.md for why merging happens once, here, rather
// than being computed fresh on every forward pass.
void MergeLoraAdapter(float* weight, const float* lora_a, const float* lora_b,
                      int64_t out_features, int64_t in_features, int64_t rank,
                      float scale);

}  // namespace kiln
