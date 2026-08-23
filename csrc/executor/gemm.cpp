#include "executor/gemm.h"

#include <cstring>

namespace kiln {

void Gemm(const float* A, const float* B, float* C, int64_t M, int64_t K,
          int64_t N) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M * N));
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t k = 0; k < K; ++k) {
      float a_ik = A[i * K + k];
      const float* b_row = B + k * N;
      float* c_row = C + i * N;
      for (int64_t j = 0; j < N; ++j) {
        c_row[j] += a_ik * b_row[j];
      }
    }
  }
}

void GemmBT(const float* A, const float* B_transposed, float* C, int64_t M,
            int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i) {
    const float* a_row = A + i * K;
    for (int64_t j = 0; j < N; ++j) {
      const float* b_row = B_transposed + j * K;
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) sum += a_row[k] * b_row[k];
      C[i * N + j] = sum;
    }
  }
}

}  // namespace kiln
