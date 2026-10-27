#pragma once

#include <cstdint>

namespace kiln {

// Plain-C++ names for the GPU kernels. Each one matches a CPU function in
// csrc/executor (or csrc/quant), and tests/cpp/cuda_kernels_test.cu checks
// that they give the same answers.
//
// GPU version of GemmBT: A[M,K] times B_transposed[N,K]. cuBLAS does the
// matrix multiply, since beating it by hand is not worth the effort.
void GemmBTCuda(const float* a, const float* b_transposed, float* c, int64_t m,
                int64_t k, int64_t n);
void AddCuda(float* in_out, const float* addend, int64_t count);
void SwiGluActivateCuda(const float* gate, const float* up, float* out,
                        int64_t count);
void RmsNormCuda(const float* x, const float* weight, float* out,
                 int64_t n_rows, int64_t dim, float eps);
void ArgmaxCuda(const float* logits, int64_t vocab_size, int32_t* out_index);
void ApplyRopeCuda(float* x, const int64_t* positions, int64_t n_tokens,
                   int64_t n_heads, int64_t head_dim, float theta);
void AttentionCuda(const float* q, const float* k, const float* v, float* out,
                   int64_t seq_len, int64_t kv_len, int64_t n_heads,
                   int64_t n_kv_heads, int64_t head_dim,
                   int64_t query_start_pos);

// GPU version of Int8GemmBT: INT8 matrix multiply through cuBLAS with INT32
// totals, then a custom kernel scales the result back to decimals.
// a_quantized is [M,K], b_quantized_transposed is [N,K], c is [M,N].
void Int8GemmBTCuda(const int8_t* a_quantized, const float* a_scales,
                    const int8_t* b_quantized_transposed, const float* b_scales,
                    float* c, int64_t m, int64_t k, int64_t n);

}  // namespace kiln
