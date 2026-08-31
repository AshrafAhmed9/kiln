// Times the real INT8 GEMM (Int8GemmBTCuda) against the existing FP32
// GEMM (GemmBTCuda) on the same shape -- this is the measurement Phase 9's
// CPU-only INT8 story has been missing: the actual GPU speed win the
// existing README table names as "not yet measured," not another
// memory/accuracy number.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
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

// See tests/cpp/cuda_kernels_test.cu's DeviceSupportsInt8TensorCores for
// why this specific version check exists (a P100's compute capability 6.0
// is one minor version short of cuBLAS's INT8 tensor-core path).
bool DeviceSupportsInt8TensorCores() {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp props;
  CheckCuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
  return (props.major > 6) || (props.major == 6 && props.minor >= 1);
}

template <typename T>
float TimeMedianMs(int iterations, T&& launch_and_sync) {
  cudaEvent_t start, end;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  CheckCuda(cudaEventCreate(&end), "cudaEventCreate(end)");
  constexpr int kSamples = 15;
  std::vector<float> milliseconds;
  milliseconds.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) launch_and_sync();
    CheckCuda(cudaEventRecord(end), "cudaEventRecord(end)");
    CheckCuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float elapsed = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsed, start, end), "cudaEventElapsedTime");
    milliseconds.push_back(elapsed / iterations);
  }
  cudaEventDestroy(end);
  cudaEventDestroy(start);
  std::sort(milliseconds.begin(), milliseconds.end());
  return milliseconds[milliseconds.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (!DeviceSupportsInt8TensorCores()) {
    std::cout << "SKIPPED: this GPU's compute capability is below 6.1 -- "
                 "cuBLAS's INT8 tensor-core path isn't available here. "
                 "Real, named hardware gap, not a benchmark failure.\n";
    return 0;
  }

  // SmolLM2-135M's own dimensions (Phase 22's real reference checkpoint):
  // hidden_size 576, one MLP projection at 1536. K must be a multiple of 4
  // for cuBLAS's INT8 IMMA path; both these already are.
  int rows = 512;
  int k = 576;
  int n = 1536;
  int iterations = 200;
  if (argc == 5) {
    rows = ParsePositive(argv[1], "rows");
    k = ParsePositive(argv[2], "k");
    n = ParsePositive(argv[3], "n");
    iterations = ParsePositive(argv[4], "iterations");
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [rows k n iterations]\n";
    return 2;
  }

  std::mt19937 rng(1);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> a(static_cast<size_t>(rows) * k);
  for (float& v : a) v = dist(rng);
  std::vector<float> b(static_cast<size_t>(n) * k);
  for (float& v : b) v = dist(rng);

  // FP32 path -- identical to what the served model actually runs today.
  float *d_a_f32 = nullptr, *d_b_f32 = nullptr, *d_c_f32 = nullptr;
  CheckCuda(cudaMalloc(&d_a_f32, a.size() * sizeof(float)), "malloc a f32");
  CheckCuda(cudaMalloc(&d_b_f32, b.size() * sizeof(float)), "malloc b f32");
  CheckCuda(cudaMalloc(&d_c_f32, static_cast<size_t>(rows) * n * sizeof(float)),
            "malloc c f32");
  CheckCuda(cudaMemcpy(d_a_f32, a.data(), a.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy a f32");
  CheckCuda(cudaMemcpy(d_b_f32, b.data(), b.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy b f32");

  for (int i = 0; i < 20; ++i) kiln::GemmBTCuda(d_a_f32, d_b_f32, d_c_f32, rows, k, n);
  CheckCuda(cudaDeviceSynchronize(), "fp32 warmup");
  float fp32_median_ms = TimeMedianMs(iterations, [&] {
    kiln::GemmBTCuda(d_a_f32, d_b_f32, d_c_f32, rows, k, n);
  });

  // INT8 path -- quantize once on the host (this cost is paid once per
  // weight at load time in a real deployment, and once per activation per
  // call; only the GEMM itself is what this benchmark times).
  auto quantize_per_row = [](const std::vector<float>& values, int64_t out_rows,
                             int64_t cols, std::vector<int8_t>* out_q,
                             std::vector<float>* out_scales) {
    out_q->resize(values.size());
    out_scales->resize(out_rows);
    for (int64_t r = 0; r < out_rows; ++r) {
      float max_abs = 0.0f;
      for (int64_t c = 0; c < cols; ++c) {
        max_abs = std::max(max_abs, std::abs(values[r * cols + c]));
      }
      float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
      (*out_scales)[r] = scale;
      for (int64_t c = 0; c < cols; ++c) {
        float steps = std::round(values[r * cols + c] / scale);
        steps = std::clamp(steps, -127.0f, 127.0f);
        (*out_q)[r * cols + c] = static_cast<int8_t>(steps);
      }
    }
  };
  std::vector<int8_t> qa, qb;
  std::vector<float> scales_a, scales_b;
  quantize_per_row(a, rows, k, &qa, &scales_a);
  quantize_per_row(b, n, k, &qb, &scales_b);

  int8_t *d_a_i8 = nullptr, *d_b_i8 = nullptr;
  float *d_scales_a = nullptr, *d_scales_b = nullptr, *d_c_i8 = nullptr;
  CheckCuda(cudaMalloc(&d_a_i8, qa.size() * sizeof(int8_t)), "malloc a i8");
  CheckCuda(cudaMalloc(&d_b_i8, qb.size() * sizeof(int8_t)), "malloc b i8");
  CheckCuda(cudaMalloc(&d_scales_a, scales_a.size() * sizeof(float)), "malloc scales a");
  CheckCuda(cudaMalloc(&d_scales_b, scales_b.size() * sizeof(float)), "malloc scales b");
  CheckCuda(cudaMalloc(&d_c_i8, static_cast<size_t>(rows) * n * sizeof(float)),
            "malloc c i8");
  CheckCuda(cudaMemcpy(d_a_i8, qa.data(), qa.size(), cudaMemcpyHostToDevice), "copy a i8");
  CheckCuda(cudaMemcpy(d_b_i8, qb.data(), qb.size(), cudaMemcpyHostToDevice), "copy b i8");
  CheckCuda(cudaMemcpy(d_scales_a, scales_a.data(), scales_a.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy scales a");
  CheckCuda(cudaMemcpy(d_scales_b, scales_b.data(), scales_b.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy scales b");

  for (int i = 0; i < 20; ++i) {
    kiln::Int8GemmBTCuda(d_a_i8, d_scales_a, d_b_i8, d_scales_b, d_c_i8, rows, k, n);
  }
  CheckCuda(cudaDeviceSynchronize(), "int8 warmup");
  float int8_median_ms = TimeMedianMs(iterations, [&] {
    kiln::Int8GemmBTCuda(d_a_i8, d_scales_a, d_b_i8, d_scales_b, d_c_i8, rows, k, n);
  });

  std::cout << std::fixed << std::setprecision(6)
            << "rows=" << rows << " k=" << k << " n=" << n
            << " fp32_median_ms=" << fp32_median_ms
            << " int8_median_ms=" << int8_median_ms
            << " speedup=" << (fp32_median_ms / int8_median_ms) << "\n";

  cudaFree(d_a_f32); cudaFree(d_b_f32); cudaFree(d_c_f32);
  cudaFree(d_a_i8); cudaFree(d_b_i8); cudaFree(d_scales_a); cudaFree(d_scales_b);
  cudaFree(d_c_i8);
}
