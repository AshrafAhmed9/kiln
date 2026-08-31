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

// A real INT8xINT8 GEMM: both operands are already-quantized INT8 (with one
// scale per row on each side -- QuantizeInt8PerChannel produces exactly this
// shape for either the weight side or, applied to activations instead of
// weights, the input side too, since the math is identical either way).
// The dot products accumulate in INT32 (so 127*127 summed many times never
// overflows an INT8 or even an INT16), and only the final sum gets
// converted back to a float and scaled -- this is what actually lets INT8
// tensor cores do the work on a GPU, unlike this project's existing
// quantize-then-dequantize-then-FP32-matmul path (Phase 9), which only ever
// saves memory, never GEMM time, because it does the real multiplication in
// FP32 either way. See docs/learning/phase-26.md.
//
// a_quantized is [M, K], one scale per row (out_scales_a, M entries) --
// this is the *activation* side, quantized fresh per call, since unlike
// weights an activation isn't known ahead of time.
// b_quantized_transposed is [N, K] (HF/GemmBT's usual out-features-first
// layout), one scale per row (out_scales_b, N entries) -- this is the
// weight side, already quantized once at load time.
// c is [M, N].
void Int8GemmBT(const int8_t* a_quantized, const float* a_scales,
                const int8_t* b_quantized_transposed, const float* b_scales,
                float* c, int64_t M, int64_t K, int64_t N);

}  // namespace kiln
