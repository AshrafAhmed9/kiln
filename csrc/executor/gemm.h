#pragma once
#include <cstddef>
#include <cstdint>

namespace kiln {

// C[M,N] = A[M,K] * B[K,N], all row-major. Honest and not world-class
// (ADR-004/010): loop order i-k-j keeps the inner loop's B/C access
// sequential, which is the one cache-blocking lesson worth having in a CPU
// reference path -- the GPU path (Part II) replaces this entirely with
// cuBLAS, so a fuller tiling scheme here would be effort spent on code that
// gets thrown away.
void Gemm(const float* A, const float* B, float* C, int64_t M, int64_t K,
          int64_t N);

// C[M,N] = A[M,K] * B^T, where B is stored [N,K] (row-major) -- the layout
// HF/safetensors uses for linear-layer weights (out_features, in_features).
// Avoids materializing a transposed copy of every weight matrix.
void GemmBT(const float* A, const float* B_transposed, float* C, int64_t M,
            int64_t K, int64_t N);

}  // namespace kiln
