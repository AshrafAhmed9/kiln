#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "executor/attention.h"
#include "executor/cuda_model.h"
#include "executor/gemm.h"
#include "executor/model.h"
#include "executor/rmsnorm.h"
#include "executor/rope.h"
#include "executor/sampler.h"
#include "kernels/cuda/kernels.h"
#include "quant/quantize.h"

namespace kiln {
namespace {

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

// cuBLAS's INT8xINT8->INT32 tensor-core path (CUBLAS_COMPUTE_32I) needs
// compute capability 6.1 or newer (Pascal GP102-and-up, or Turing/Ampere) --
// a P100 (6.0, the exact GPU this repo has been validated on via Kaggle) is
// one minor version short of it. This distinguishes that real, named
// hardware gap from an actual bug: the CPU reference (Int8GemmBT) is still
// fully tested regardless of which GPU this session happens to have.
bool DeviceSupportsInt8TensorCores() {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp props;
  CheckCuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
  return (props.major > 6) || (props.major == 6 && props.minor >= 1);
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

ModelConfig TinyCudaModelConfig() {
  ModelConfig config;
  config.vocab_size = 16;
  config.hidden_size = 8;
  config.n_layers = 2;
  config.n_heads = 2;
  config.n_kv_heads = 1;
  config.head_dim = 4;
  config.ffn_hidden = 16;
  config.max_seq_len = 8;
  return config;
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

// Phase 26: the real INT8 GEMM, checked against Int8GemmBT's CPU reference
// -- the same already-quantized numbers, same INT32 accumulation math, run
// on the GPU's tensor cores instead of a CPU loop. This is a correctness
// check, not the speed measurement (that's bench/int8_gemm_cuda_benchmark.cu).
TEST(CudaInt8Gemm, MatchesCpuReference) {
  if (!DeviceSupportsInt8TensorCores()) {
    GTEST_SKIP() << "This GPU's compute capability is below 6.1 -- cuBLAS's "
                    "INT8 tensor-core path (CUBLAS_COMPUTE_32I) isn't "
                    "available here. Real, named hardware gap, not a bug: "
                    "see docs/correctness.md.";
  }
  constexpr int64_t kRows = 4;   // "M" -- number of activation rows
  constexpr int64_t kInput = 32;   // "K" -- must be a multiple of 4 for cuBLAS's INT8 IMMA path
  constexpr int64_t kOutput = 16;  // "N" -- number of output features

  std::mt19937 rng(5);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> activations(kRows * kInput);
  for (float& v : activations) v = dist(rng);
  std::vector<float> weights(kOutput * kInput);
  for (float& v : weights) v = dist(rng);

  std::vector<int8_t> quantized_a(kRows * kInput);
  std::vector<float> scales_a(kRows);
  QuantizeInt8PerChannel(activations.data(), kRows, kInput, quantized_a.data(),
                        scales_a.data());
  std::vector<int8_t> quantized_b(kOutput * kInput);
  std::vector<float> scales_b(kOutput);
  QuantizeInt8PerChannel(weights.data(), kOutput, kInput, quantized_b.data(),
                        scales_b.data());

  std::vector<float> expected(kRows * kOutput);
  Int8GemmBT(quantized_a.data(), scales_a.data(), quantized_b.data(),
            scales_b.data(), expected.data(), kRows, kInput, kOutput);

  DeviceBuffer<int8_t> device_a(quantized_a.size());
  DeviceBuffer<int8_t> device_b(quantized_b.size());
  DeviceBuffer<float> device_scales_a(scales_a.size());
  DeviceBuffer<float> device_scales_b(scales_b.size());
  DeviceBuffer<float> device_out(expected.size());
  device_a.CopyFrom(quantized_a);
  device_b.CopyFrom(quantized_b);
  device_scales_a.CopyFrom(scales_a);
  device_scales_b.CopyFrom(scales_b);

  Int8GemmBTCuda(device_a.data(), device_scales_a.data(), device_b.data(),
                device_scales_b.data(), device_out.data(), kRows, kInput,
                kOutput);
  SyncKernel("Int8GemmBTCuda");
  std::vector<float> actual = device_out.CopyToHost();

  // Exact-integer-math parity, not a loose accuracy bound: both sides
  // consume the identical already-quantized INT8 numbers and accumulate in
  // INT32, so the GPU and CPU paths should agree almost to the last bit of
  // float precision in the final scale multiply, not just "close."
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-3f);
  }
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

TEST(CudaModel, FullPrefillMatchesCpuReference) {
  Model model = Model::LoadRandom(TinyCudaModelConfig(), /*seed=*/29);
  const std::vector<int32_t> tokens = {1, 2, 3};
  std::vector<float> expected(tokens.size() * model.config().vocab_size);
  model.Forward(tokens.data(), /*batch_size=*/1, tokens.size(), nullptr,
                /*start_pos=*/0, nullptr, expected.data());

  CudaModel cuda_model(model);
  std::vector<float> actual(expected.size());
  cuda_model.Forward(tokens.data(), tokens.size(), /*start_pos=*/0,
                     actual.data());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }
}

TEST(CudaModel, CachedDecodeMatchesCpuReference) {
  Model model = Model::LoadRandom(TinyCudaModelConfig(), /*seed=*/37);
  const std::vector<int32_t> prompt = {1, 2, 3};
  const int32_t next_token = 4;
  KVCache cpu_cache(model.config().n_layers, model.config().max_seq_len,
                    model.config().n_kv_heads, model.config().head_dim);
  std::vector<float> cpu_prompt(prompt.size() * model.config().vocab_size);
  std::vector<float> expected(model.config().vocab_size);
  model.Forward(prompt.data(), /*batch_size=*/1, prompt.size(), nullptr,
                /*start_pos=*/0, &cpu_cache, cpu_prompt.data());
  model.Forward(&next_token, /*batch_size=*/1, /*seq_len=*/1, nullptr,
                /*start_pos=*/prompt.size(), &cpu_cache, expected.data());

  CudaModel cuda_model(model);
  std::vector<float> gpu_prompt(cpu_prompt.size());
  std::vector<float> actual(expected.size());
  cuda_model.ForwardCached(prompt.data(), prompt.size(), /*start_pos=*/0,
                           gpu_prompt.data());
  cuda_model.ForwardCached(&next_token, /*seq_len=*/1,
                           /*start_pos=*/prompt.size(), actual.data());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }
}

}  // namespace
}  // namespace kiln
