// INT8 matrix multiply on the GPU. The multiplication itself is cuBLAS with
// INT32 totals, mirroring Int8GemmBT in csrc/quant/quantize.h. Then a small
// custom kernel multiplies each total by the two scales to get decimals
// back. Keeping the work in INT8 until the end is what makes it faster than
// FP32 (measured 1.60x on a GTX 1660 Ti).
//
// Every pointer is already in GPU memory; the caller does the copying.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace kiln {

namespace {

// The one genuinely new piece of device code this feature needs: turning
// each INT32 dot-product sum back into a real number, scaled by its row's
// activation scale and its column's weight scale. This runs once per
// output element, after every INT8 multiply-add already happened inside
// cublasGemmEx -- exactly mirroring where Int8GemmBT's CPU version does
// its scaling, just as a device kernel instead of a host loop, so the
// result never has to leave the GPU.
__global__ void DequantScaleKernel(const int32_t* acc, const float* a_scales,
                                   const float* b_scales, float* out,
                                   int64_t m, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  int64_t row = idx / n;
  int64_t col = idx % n;
  out[idx] = static_cast<float>(acc[idx]) * a_scales[row] * b_scales[col];
}

}  // namespace

void Int8GemmBTCuda(const int8_t* a_quantized, const float* a_scales,
                    const int8_t* b_quantized_transposed,
                    const float* b_scales, float* c, int64_t m, int64_t k,
                    int64_t n) {
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasCreate failed");
  }

  int32_t* acc = nullptr;
  if (cudaMalloc(&acc, m * n * sizeof(int32_t)) != cudaSuccess) {
    cublasDestroy(handle);
    throw std::runtime_error("cudaMalloc failed for INT8 GEMM accumulator");
  }

  // cuBLAS expects column-major matrices and ours are row-major, so we ask
  // for C^T = B * A^T instead, which swaps the operands. For INT32 output,
  // cuBLAS wants alpha and beta as INT32 too, not float.
  const int32_t alpha = 1;
  const int32_t beta = 0;
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha, b_quantized_transposed,
      CUDA_R_8I, static_cast<int>(k), a_quantized, CUDA_R_8I,
      static_cast<int>(k), &beta, acc, CUDA_R_32I, static_cast<int>(n),
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(acc);
    throw std::runtime_error("cublasGemmEx (INT8) failed");
  }

  int threads_per_block = 256;
  int64_t total = m * n;
  int blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
  DequantScaleKernel<<<blocks, threads_per_block>>>(acc, a_scales, b_scales, c,
                                                     m, n);
  cudaFree(acc);
}

}  // namespace kiln
