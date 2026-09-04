#pragma once

#include <cstdint>

namespace kiln {

// Device-resident equivalent of GemmBT: A[M,K] times B_transposed[N,K].
// cuBLAS owns dense GEMM by the project's deliberate kernel strategy.
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
                   int64_t query_start_pos, int64_t valid_kv_len);

// The real INT8xINT8 GEMM on GPU tensor cores, via cublasGemmEx's INT32
// accumulation path -- the GPU counterpart of Int8GemmBT (csrc/quant/
// quantize.h), and the piece this project's CPU-only INT8 story has been
// missing: a genuine speed measurement, not just a memory-and-accuracy
// tradeoff table. a_quantized is [M,K], b_quantized_transposed is [N,K]
// (both already INT8, one scale per row on each side), c is [M,N].
void Int8GemmBTCuda(const int8_t* a_quantized, const float* a_scales,
                    const int8_t* b_quantized_transposed, const float* b_scales,
                    float* c, int64_t m, int64_t k, int64_t n);

}  // namespace kiln
