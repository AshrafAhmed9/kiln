// Small device-side vector operations used between cuBLAS projections.
#include <cuda_runtime.h>

#include <cstdint>

namespace kiln {

__global__ void AddKernel(float* in_out, const float* addend, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) in_out[index] += addend[index];
}

__global__ void SwiGluActivateKernel(const float* gate, const float* up,
                                     float* out, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    float g = gate[index];
    out[index] = (g / (1.0f + expf(-g))) * up[index];
  }
}

void AddCuda(float* in_out, const float* addend, int64_t count) {
  constexpr int kThreads = 256;
  AddKernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(in_out, addend,
                                                               count);
}

void SwiGluActivateCuda(const float* gate, const float* up, float* out,
                        int64_t count) {
  constexpr int kThreads = 256;
  SwiGluActivateKernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
      gate, up, out, count);
}

}  // namespace kiln
