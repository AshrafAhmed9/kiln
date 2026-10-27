#include "executor/gemm.h"

namespace kiln {

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
