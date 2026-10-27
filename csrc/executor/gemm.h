#pragma once
#include <cstddef>
#include <cstdint>

namespace kiln {

// Matrix multiply: C[M,N] = A[M,K] * B^T, where B is stored as [N,K].
//
// Model weights are stored one output per row ([out_features, in_features]),
// which is exactly this "B transposed" layout, so no weight needs to be
// flipped first. This is the plain CPU version. On the GPU, cuBLAS does the
// same job much faster.
void GemmBT(const float* A, const float* B_transposed, float* C, int64_t M,
            int64_t K, int64_t N);

}  // namespace kiln
