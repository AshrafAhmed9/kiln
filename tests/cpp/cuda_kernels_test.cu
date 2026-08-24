#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "executor/attention.h"
#include "executor/gemm.h"
#include "executor/rmsnorm.h"
#include "executor/rope.h"
#include "executor/sampler.h"
#include "kernels/cuda/kernels.h"

namespace kiln {
namespace {

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
              "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  T* data() { return data_; }
  void CopyFrom(const std::vector<T>& host) {
    ASSERT_EQ(host.size(), count_);
    CheckCuda(cudaMemcpy(data_, host.data(), count_ * sizeof(T),
                         cudaMemcpyHostToDevice), "copy to device");
  }
  std::vector<T> CopyToHost() {
    std::vector<T> host(count_);
    CheckCuda(cudaMemcpy(host.data(), data_, count_ * sizeof(T),
                         cudaMemcpyDeviceToHost), "copy from device");
    return host;
  }

 private:
  T* data_ = nullptr;
  size_t count_;
};

void SyncKernel(const char* kernel) {
  CheckCuda(cudaGetLastError(), kernel);
  CheckCuda(cudaDeviceSynchronize(), kernel);
}

TEST(CudaRmsNorm, MatchesCpuReference) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kDim = 64;
  std::vector<float> input(kRows * kDim);
  std::vector<float> weight(kDim);
  for (int64_t i = 0; i < kRows * kDim; ++i) input[i] = (i % 11 - 5) * 0.125f;
  for (int64_t i = 0; i < kDim; ++i) weight[i] = 1.0f + i * 0.01f;
  std::vector<float> expected(kRows * kDim);
  RmsNorm(input.data(), weight.data(), expected.data(), kRows, kDim, 1e-5f);

  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<float> device_weight(weight.size());
  DeviceBuffer<float> device_output(expected.size());
  device_input.CopyFrom(input);
  device_weight.CopyFrom(weight);
  RmsNormCuda(device_input.data(), device_weight.data(), device_output.data(),
              kRows, kDim, 1e-5f);
  SyncKernel("RmsNormCuda");
  std::vector<float> actual = device_output.CopyToHost();
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaGemm, CuBlasMatchesCpuReference) {
  constexpr int64_t kRows = 3;
  constexpr int64_t kInput = 4;
  constexpr int64_t kOutput = 5;
  std::vector<float> a(kRows * kInput);
  std::vector<float> b(kOutput * kInput);
  for (size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<int>(i) - 4) * 0.125f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = (static_cast<int>(i) - 7) * 0.1f;
  std::vector<float> expected(kRows * kOutput);
  GemmBT(a.data(), b.data(), expected.data(), kRows, kInput, kOutput);
  DeviceBuffer<float> device_a(a.size());
  DeviceBuffer<float> device_b(b.size());
  DeviceBuffer<float> device_out(expected.size());
  device_a.CopyFrom(a);
  device_b.CopyFrom(b);
  GemmBTCuda(device_a.data(), device_b.data(), device_out.data(), kRows, kInput,
             kOutput);
  SyncKernel("GemmBTCuda");
  std::vector<float> actual = device_out.CopyToHost();
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaElementwise, MatchesCpuReference) {
  std::vector<float> left = {-2.0f, -0.5f, 0.0f, 1.5f};
  std::vector<float> right = {0.25f, 2.0f, -3.0f, 4.0f};
  std::vector<float> expected_add = left;
  std::vector<float> expected_swiglu(left.size());
  for (size_t i = 0; i < left.size(); ++i) {
    expected_add[i] += right[i];
    expected_swiglu[i] = (left[i] / (1.0f + std::exp(-left[i]))) * right[i];
  }
  DeviceBuffer<float> device_left(left.size());
  DeviceBuffer<float> device_gate(left.size());
  DeviceBuffer<float> device_right(right.size());
  DeviceBuffer<float> device_out(left.size());
  device_left.CopyFrom(left);
  device_gate.CopyFrom(left);
  device_right.CopyFrom(right);
  AddCuda(device_left.data(), device_right.data(), left.size());
  SyncKernel("AddCuda");
  std::vector<float> actual_add = device_left.CopyToHost();
  for (size_t i = 0; i < left.size(); ++i) EXPECT_NEAR(actual_add[i], expected_add[i], 1e-6f);
  SwiGluActivateCuda(device_gate.data(), device_right.data(), device_out.data(), left.size());
  SyncKernel("SwiGluActivateCuda");
  std::vector<float> actual_swiglu = device_out.CopyToHost();
  for (size_t i = 0; i < left.size(); ++i) EXPECT_NEAR(actual_swiglu[i], expected_swiglu[i], 1e-6f);
}

TEST(CudaSampler, MatchesCpuGreedyArgmax) {
  std::vector<float> logits = {-2.0f, 0.1f, 5.0f, 4.0f, 3.0f};
  SamplerConfig config;
  config.temperature = 0.0f;
  std::mt19937 rng(1);
  int32_t expected = Sample(logits.data(), logits.size(), config, {}, rng);
  DeviceBuffer<float> device_logits(logits.size());
  DeviceBuffer<int32_t> device_output(1);
  device_logits.CopyFrom(logits);
  ArgmaxCuda(device_logits.data(), logits.size(), device_output.data());
  SyncKernel("ArgmaxCuda");
  EXPECT_EQ(device_output.CopyToHost()[0], expected);
}

TEST(CudaRope, MatchesCpuReference) {
  constexpr int64_t kTokens = 2;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kHeadDim = 8;
  std::vector<float> expected(kTokens * kHeads * kHeadDim);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = i * 0.1f;
  std::vector<float> actual = expected;
  std::vector<int64_t> positions = {0, 3};
  ApplyRope(expected.data(), positions.data(), kTokens, kHeads, kHeadDim, 10000.0f);
  DeviceBuffer<float> device_x(actual.size());
  DeviceBuffer<int64_t> device_positions(positions.size());
  device_x.CopyFrom(actual);
  device_positions.CopyFrom(positions);
  ApplyRopeCuda(device_x.data(), device_positions.data(), kTokens, kHeads,
                kHeadDim, 10000.0f);
  SyncKernel("ApplyRopeCuda");
  actual = device_x.CopyToHost();
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaAttention, MatchesCpuReference) {
  constexpr int64_t kSeqLen = 2;
  constexpr int64_t kKvLen = 3;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kHeadDim = 4;
  std::vector<float> q(kSeqLen * kHeads * kHeadDim, 0.1f);
  std::vector<float> k(kKvLen * kKvHeads * kHeadDim, 0.2f);
  std::vector<float> v(kKvLen * kKvHeads * kHeadDim);
  for (size_t i = 0; i < v.size(); ++i) v[i] = i * 0.05f;
  std::vector<float> expected(q.size());
  Attention(q.data(), k.data(), v.data(), expected.data(), kSeqLen, kKvLen,
            kHeads, kKvHeads, kHeadDim, /*query_start_pos=*/1);
  DeviceBuffer<float> device_q(q.size());
  DeviceBuffer<float> device_k(k.size());
  DeviceBuffer<float> device_v(v.size());
  DeviceBuffer<float> device_output(expected.size());
  device_q.CopyFrom(q);
  device_k.CopyFrom(k);
  device_v.CopyFrom(v);
  AttentionCuda(device_q.data(), device_k.data(), device_v.data(),
                device_output.data(), kSeqLen, kKvLen, kHeads, kKvHeads,
                kHeadDim, /*query_start_pos=*/1, /*valid_kv_len=*/-1);
  SyncKernel("AttentionCuda");
  std::vector<float> actual = device_output.CopyToHost();
  for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

}  // namespace
}  // namespace kiln
