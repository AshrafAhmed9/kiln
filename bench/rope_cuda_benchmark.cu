// Measures the raw-CUDA half of ADR-007's one-kernel comparison. The Python
// Triton companion uses the identical SmolLM2-135M shape and event-timing
// protocol, so the two reported medians are directly comparable.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/cuda/kernels.h"

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

int ParsePositive(const char* text, const char* name) {
  const int value = std::stoi(text);
  if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  // SmolLM2-135M uses 9 attention heads with a head dimension of 64. A 512
  // token prefill is big enough to amortize timer noise but small enough to
  // fit on every Kaggle GPU that has run Kiln's validation notebook.
  int tokens = 512;
  int heads = 9;
  int head_dim = 64;
  int iterations = 1'000;
  if (argc == 5) {
    tokens = ParsePositive(argv[1], "tokens");
    heads = ParsePositive(argv[2], "heads");
    head_dim = ParsePositive(argv[3], "head_dim");
    iterations = ParsePositive(argv[4], "iterations");
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0]
              << " [tokens heads head_dim iterations]\n";
    return 2;
  }
  if (head_dim % 2 != 0) {
    std::cerr << "head_dim must be even\n";
    return 2;
  }

  const size_t elements = static_cast<size_t>(tokens) * heads * head_dim;
  float* x = nullptr;
  int64_t* positions = nullptr;
  CheckCuda(cudaMalloc(&x, elements * sizeof(float)), "cudaMalloc(x)");
  CheckCuda(cudaMalloc(&positions, tokens * sizeof(int64_t)), "cudaMalloc(positions)");
  CheckCuda(cudaMemset(x, 0, elements * sizeof(float)), "cudaMemset(x)");
  std::vector<int64_t> host_positions(tokens);
  std::iota(host_positions.begin(), host_positions.end(), int64_t{0});
  CheckCuda(cudaMemcpy(positions, host_positions.data(), tokens * sizeof(int64_t),
                       cudaMemcpyHostToDevice), "cudaMemcpy(positions)");

  for (int i = 0; i < 100; ++i) {
    kiln::ApplyRopeCuda(x, positions, tokens, heads, head_dim, 10000.0f);
  }
  CheckCuda(cudaGetLastError(), "warmup launch");
  CheckCuda(cudaDeviceSynchronize(), "warmup synchronize");

  cudaEvent_t start;
  cudaEvent_t end;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  CheckCuda(cudaEventCreate(&end), "cudaEventCreate(end)");
  constexpr int kSamples = 21;
  std::vector<float> milliseconds;
  milliseconds.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
      kiln::ApplyRopeCuda(x, positions, tokens, heads, head_dim, 10000.0f);
    }
    CheckCuda(cudaEventRecord(end), "cudaEventRecord(end)");
    CheckCuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float elapsed = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsed, start, end), "cudaEventElapsedTime");
    milliseconds.push_back(elapsed / iterations);
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const float median_ms = milliseconds[milliseconds.size() / 2];
  // One rotation reads and writes every float once. This is an effective
  // bandwidth figure for comparison only, not a claim about total DRAM use.
  const double bytes_per_call = static_cast<double>(elements) * 2 * sizeof(float);
  const double effective_gb_s = bytes_per_call / (median_ms * 1e6);
  std::cout << std::fixed << std::setprecision(6)
            << "implementation=raw_cuda tokens=" << tokens << " heads=" << heads
            << " head_dim=" << head_dim << " iterations=" << iterations
            << " median_ms=" << median_ms
            << " effective_gb_s=" << effective_gb_s << "\n";

  cudaEventDestroy(end);
  cudaEventDestroy(start);
  cudaFree(positions);
  cudaFree(x);
}
