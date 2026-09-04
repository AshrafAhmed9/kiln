#include "quant/quantize.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>

#include "executor/gemm.h"

namespace kiln {
namespace {

TEST(QuantizeInt8, RoundTripStaysCloseToOriginal) {
  int64_t rows = 3, cols = 8;
  std::vector<float> weights = {
      0.1f,  -0.2f, 0.3f,  -0.4f, 0.05f, -0.05f, 0.01f, -0.01f,
      1.0f,  -2.0f, 3.0f,  -4.0f, 0.5f,  -0.5f,  0.1f,  -0.1f,
      -1.5f, 1.5f,  -1.5f, 1.5f,  -1.5f, 1.5f,   -1.5f, 1.5f,
  };
  std::vector<int8_t> quantized(rows * cols);
  std::vector<float> scales(rows);
  QuantizeInt8PerChannel(weights.data(), rows, cols, quantized.data(),
                         scales.data());

  std::vector<float> dequantized(rows * cols);
  DequantizeInt8PerChannel(quantized.data(), scales.data(), rows, cols,
                           dequantized.data());

  // Every row's biggest weight defines its scale, so the biggest weight in
  // each row should come back almost exactly right; smaller weights in
  // that row can be off by up to about half a "step" -- that's the known,
  // expected error of round-to-nearest quantization, not a bug.
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      float original = weights[r * cols + c];
      float reconstructed = dequantized[r * cols + c];
      EXPECT_NEAR(original, reconstructed, scales[r] * 0.5f + 1e-6f);
    }
  }
}

TEST(QuantizeInt8, AllZeroRowDoesNotDivideByZero) {
  int64_t rows = 1, cols = 4;
  std::vector<float> weights = {0, 0, 0, 0};
  std::vector<int8_t> quantized(cols);
  std::vector<float> scales(rows);

  QuantizeInt8PerChannel(weights.data(), rows, cols, quantized.data(),
                         scales.data());

  for (int8_t q : quantized) EXPECT_EQ(q, 0);
  EXPECT_TRUE(std::isfinite(scales[0]));
}

TEST(QuantizeInt4, OddGroupSizeThrowsInsteadOfSilentlyMisaligningBytes) {
  std::vector<float> weights(8, 1.0f);
  std::vector<uint8_t> packed(4);
  std::vector<float> scales(4);
  EXPECT_THROW(QuantizeInt4GroupWise(weights.data(), 1, 8, /*group_size=*/3,
                                     packed.data(), scales.data()),
               std::invalid_argument);
}

TEST(QuantizeInt4, GroupSizeNotDividingColsThrows) {
  // 6 is even (so it isn't rejected for that reason), but it doesn't
  // divide 8 columns evenly -- exactly the silent-truncation case this
  // check exists to catch.
  std::vector<float> weights(8, 1.0f);
  std::vector<uint8_t> packed(4);
  std::vector<float> scales(4);
  EXPECT_THROW(QuantizeInt4GroupWise(weights.data(), 1, 8, /*group_size=*/6,
                                     packed.data(), scales.data()),
               std::invalid_argument);
}

TEST(QuantizeInt4, RoundTripStaysWithinOneGroupStep) {
  int64_t rows = 2, cols = 8;
  int64_t group_size = 4;
  std::vector<float> weights = {
      0.1f, -0.2f, 0.3f, -0.4f, 1.0f, -1.0f, 0.5f, -0.5f,
      2.0f, -2.0f, 1.0f, -1.0f, 0.1f, 0.2f,  0.3f, 0.4f,
  };
  std::vector<uint8_t> packed(rows * cols / 2);
  std::vector<float> scales(rows * (cols / group_size));

  QuantizeInt4GroupWise(weights.data(), rows, cols, group_size, packed.data(),
                        scales.data());

  std::vector<float> dequantized(rows * cols);
  DequantizeInt4GroupWise(packed.data(), scales.data(), rows, cols, group_size,
                          dequantized.data());

  int64_t groups_per_row = cols / group_size;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      int64_t group = c / group_size;
      float scale = scales[r * groups_per_row + group];
      float original = weights[r * cols + c];
      float reconstructed = dequantized[r * cols + c];
      // INT4 only has 15 representable steps (vs INT8's 255), so the
      // expected round-to-nearest error is proportionally larger -- still
      // bounded by half a step, just a bigger step.
      EXPECT_NEAR(original, reconstructed, scale * 0.5f + 1e-6f);
    }
  }
}

// This is the closest this offline session can get to the plan's real
// deliverable (a perplexity/KL table against WikiText-2, which needs a
// real trained model and a real dataset, neither available here): a
// synthetic proxy showing quantized weights, used in a real matmul,
// produce output close to the full-precision result on random data.
TEST(QuantizeInt8, QuantizedMatmulStaysCloseToFullPrecisionMatmul) {
  int64_t rows = 16, cols = 16, batch = 4;
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> weights(rows * cols);
  for (float& w : weights) w = dist(rng);
  std::vector<float> input(batch * cols);
  for (float& x : input) x = dist(rng);

  std::vector<float> full_precision_out(batch * rows);
  GemmBT(input.data(), weights.data(), full_precision_out.data(), batch, cols,
         rows);

  std::vector<int8_t> quantized(rows * cols);
  std::vector<float> scales(rows);
  QuantizeInt8PerChannel(weights.data(), rows, cols, quantized.data(),
                         scales.data());
  std::vector<float> dequantized_weights(rows * cols);
  DequantizeInt8PerChannel(quantized.data(), scales.data(), rows, cols,
                           dequantized_weights.data());

  std::vector<float> quantized_out(batch * rows);
  GemmBT(input.data(), dequantized_weights.data(), quantized_out.data(), batch,
         cols, rows);

  float mean_squared_error = 0.0f;
  for (size_t i = 0; i < full_precision_out.size(); ++i) {
    float diff = full_precision_out[i] - quantized_out[i];
    mean_squared_error += diff * diff;
  }
  mean_squared_error /= static_cast<float>(full_precision_out.size());

  // A loose bound -- the point of this test is to catch a broken
  // quantizer producing wildly wrong output, not to certify a specific
  // accuracy target (that's what the real, deferred perplexity/KL
  // measurement against a real model is for).
  EXPECT_LT(mean_squared_error, 1.0f);
}

// Phase 26's actual INT8xINT8 GEMM (accumulate in INT32, scale once at the
// end) versus this project's existing quantize-then-dequantize-then-FP32-
// matmul path (Phase 9): both quantize the same weights the same way, so
// their outputs should be close to each other (both are approximating the
// same full-precision answer, from the same quantized weight numbers) --
// checking that directly is a much tighter, more specific claim than only
// checking either one against full precision separately, and it's the
// claim that actually matters: swapping which GEMM strategy computes the
// quantized answer must not change what "quantized" means.
TEST(QuantizeInt8, RealInt8GemmMatchesDequantizeThenFp32MatmulPath) {
  int64_t rows = 16, cols = 16, batch = 4;
  std::mt19937 rng(9);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> weights(rows * cols);
  for (float& w : weights) w = dist(rng);
  std::vector<float> input(batch * cols);
  for (float& x : input) x = dist(rng);

  std::vector<int8_t> quantized_weights(rows * cols);
  std::vector<float> weight_scales(rows);
  QuantizeInt8PerChannel(weights.data(), rows, cols, quantized_weights.data(),
                         weight_scales.data());

  std::vector<int8_t> quantized_input(batch * cols);
  std::vector<float> input_scales(batch);
  QuantizeInt8PerChannel(input.data(), batch, cols, quantized_input.data(),
                         input_scales.data());

  // The existing Phase 9 path: dequantize the weights back to float, then
  // an ordinary FP32 matmul (the input here is used at full precision,
  // unquantized, since Phase 9 only ever quantized weights).
  std::vector<float> dequantized_weights(rows * cols);
  DequantizeInt8PerChannel(quantized_weights.data(), weight_scales.data(), rows,
                           cols, dequantized_weights.data());
  std::vector<float> dequant_then_fp32_out(batch * rows);
  GemmBT(input.data(), dequantized_weights.data(), dequant_then_fp32_out.data(),
         batch, cols, rows);

  // The new path: both sides already INT8, accumulated in INT32, scaled
  // once at the end.
  std::vector<float> real_int8_out(batch * rows);
  Int8GemmBT(quantized_input.data(), input_scales.data(),
             quantized_weights.data(), weight_scales.data(),
             real_int8_out.data(), batch, cols, rows);

  float mean_squared_error = 0.0f;
  for (size_t i = 0; i < dequant_then_fp32_out.size(); ++i) {
    float diff = dequant_then_fp32_out[i] - real_int8_out[i];
    mean_squared_error += diff * diff;
  }
  mean_squared_error /= static_cast<float>(dequant_then_fp32_out.size());

  // Loose for the same reason as the test above -- this quantizes the
  // input as well as the weights, which the Phase 9 comparison point
  // doesn't, so a real (if small) gap between the two is expected, not
  // just quantizer noise.
  EXPECT_LT(mean_squared_error, 2.0f);
}

}  // namespace
}  // namespace kiln
