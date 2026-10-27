#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// INT8 quantization stores each weight as a whole number from -127 to 127
// plus one scale per row, which says how big one step is. That uses 1 byte
// per weight instead of 4. Giving every row its own scale stops one large
// weight from ruining the precision of the others.
//
// weights is [rows, cols]. out_quantized has the same shape. out_scales has
// one entry per row.
void QuantizeInt8PerChannel(const float* weights, int64_t rows, int64_t cols,
                            int8_t* out_quantized, float* out_scales);

void DequantizeInt8PerChannel(const int8_t* quantized, const float* scales,
                              int64_t rows, int64_t cols, float* out_weights);

// INT4 quantization: whole numbers from -7 to 7, two packed into each byte.
// Each group of `group_size` weights in a row gets its own scale (smaller
// groups are more precise but need more scales). group_size must be even
// and divide the row length.
void QuantizeInt4GroupWise(const float* weights, int64_t rows, int64_t cols,
                           int64_t group_size, uint8_t* out_packed,
                           float* out_scales);

void DequantizeInt4GroupWise(const uint8_t* packed, const float* scales,
                             int64_t rows, int64_t cols, int64_t group_size,
                             float* out_weights);

// Matrix multiply on INT8 numbers: C[M,N] = A[M,K] * B^T, where B is [N,K].
//
// The products are added up as whole numbers (INT32, so nothing overflows),
// and only the final sum is turned back into a decimal and multiplied by
// the two scales. This is what lets a GPU's INT8 hardware do the heavy work.
// Dequantizing first and multiplying in FP32 saves memory but no time.
//
// a is the input, quantized fresh on every call (a_scales has M entries).
// b is the weights, quantized once when loaded (b_scales has N entries).
void Int8GemmBT(const int8_t* a_quantized, const float* a_scales,
                const int8_t* b_quantized_transposed, const float* b_scales,
                float* c, int64_t M, int64_t K, int64_t N);

}  // namespace kiln
