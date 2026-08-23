#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// INT8, one scale per row (per output channel -- a weight matrix here is
// [rows, cols] = [out_features, in_features], so "per channel" means "per
// row"). See docs/learning/phase-09.md for why a separate scale per group
// keeps one outlier from ruining precision for everything else.
//
// weights is [rows, cols]. out_quantized (same shape) holds each weight's
// nearest representable integer step, in [-127, 127]. out_scales (one per
// row) holds how much one integer step is worth, in the original units.
void QuantizeInt8PerChannel(const float* weights, int64_t rows, int64_t cols,
                            int8_t* out_quantized, float* out_scales);

void DequantizeInt8PerChannel(const int8_t* quantized, const float* scales,
                              int64_t rows, int64_t cols, float* out_weights);

// INT4, grouped every `group_size` consecutive weights within a row (a
// smaller group than "per row" trades a little more overhead for finer
// protection against outliers -- see docs/learning/phase-09.md). Each
// value is packed two-to-a-byte, since fitting two 4-bit numbers in one
// byte is the entire reason to use 4-bit numbers instead of 8-bit ones.
// `group_size` must be even, so every group packs into whole bytes with
// nothing left over.
void QuantizeInt4GroupWise(const float* weights, int64_t rows, int64_t cols,
                          int64_t group_size, uint8_t* out_packed,
                          float* out_scales);

void DequantizeInt4GroupWise(const uint8_t* packed, const float* scales,
                            int64_t rows, int64_t cols, int64_t group_size,
                            float* out_weights);

}  // namespace kiln
