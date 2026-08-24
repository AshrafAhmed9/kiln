// Dense matrix multiplication is deliberately delegated to cuBLAS. Kiln's
// hand-written CUDA work is attention, RMSNorm, and sampling; replacing
// cuBLAS here would add code without adding an honest learning result.
#include <cublas_v2.h>

#include <cstdint>
#include <stdexcept>

namespace kiln {

void GemmBTCuda(const float* a, const float* b_transposed, float* c,
                int64_t m, int64_t k, int64_t n) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasCreate failed");
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // cuBLAS is column-major. Reinterpreting our row-major C=A*B^T as its
  // transpose gives C^T=B*A^T, so the operands swap and dimensions reverse.
  cublasStatus_t status = cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha, b_transposed,
      static_cast<int>(k), a, static_cast<int>(k), &beta, c,
      static_cast<int>(n));
  cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasSgemm failed");
  }
}

}  // namespace kiln
