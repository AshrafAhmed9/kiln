#include "quant/quantize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kiln {

void QuantizeInt8PerChannel(const float* weights, int64_t rows, int64_t cols,
                            int8_t* out_quantized, float* out_scales) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = weights + r * cols;
    float max_abs = 0.0f;
    for (int64_t c = 0; c < cols; ++c)
      max_abs = std::max(max_abs, std::abs(row[c]));

    // A row of all zeros gets scale 1 so the division below is safe.
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    out_scales[r] = scale;

    int8_t* out_row = out_quantized + r * cols;
    for (int64_t c = 0; c < cols; ++c) {
      float steps = std::round(row[c] / scale);
      steps = std::clamp(steps, -127.0f, 127.0f);
      out_row[c] = static_cast<int8_t>(steps);
    }
  }
}

void DequantizeInt8PerChannel(const int8_t* quantized, const float* scales,
                              int64_t rows, int64_t cols, float* out_weights) {
  for (int64_t r = 0; r < rows; ++r) {
    float scale = scales[r];
    const int8_t* row = quantized + r * cols;
    float* out_row = out_weights + r * cols;
    for (int64_t c = 0; c < cols; ++c) out_row[c] = row[c] * scale;
  }
}

namespace {

// Shifts a value from [-8, 7] up by 8 so it fits in 4 bits as 0 to 15.
uint8_t ToNibble(int value) { return static_cast<uint8_t>(value + 8); }
int FromNibble(uint8_t nibble) { return static_cast<int>(nibble) - 8; }

// Both INT4 functions need rows to split into whole groups, and groups to
// pack into whole bytes. Otherwise the results would be silently wrong, so
// this check fails loudly instead.
void CheckInt4Preconditions(int64_t cols, int64_t group_size) {
  if (group_size % 2 != 0) {
    throw std::invalid_argument(
        "QuantizeInt4GroupWise: group_size must be even, so every group "
        "packs into whole bytes");
  }
  if (cols % group_size != 0) {
    throw std::invalid_argument(
        "QuantizeInt4GroupWise: group_size must evenly divide cols");
  }
}

}  // namespace

void QuantizeInt4GroupWise(const float* weights, int64_t rows, int64_t cols,
                           int64_t group_size, uint8_t* out_packed,
                           float* out_scales) {
  CheckInt4Preconditions(cols, group_size);
  int64_t groups_per_row = cols / group_size;
  int64_t bytes_per_row = cols / 2;

  for (int64_t r = 0; r < rows; ++r) {
    const float* row = weights + r * cols;

    for (int64_t g = 0; g < groups_per_row; ++g) {
      const float* group = row + g * group_size;
      float max_abs = 0.0f;
      for (int64_t i = 0; i < group_size; ++i) {
        max_abs = std::max(max_abs, std::abs(group[i]));
      }
      float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
      out_scales[r * groups_per_row + g] = scale;

      uint8_t* packed_row = out_packed + r * bytes_per_row;
      for (int64_t i = 0; i < group_size; i += 2) {
        int64_t col = g * group_size + i;
        float low_steps = std::clamp(std::round(group[i] / scale), -7.0f, 7.0f);
        float high_steps =
            std::clamp(std::round(group[i + 1] / scale), -7.0f, 7.0f);
        uint8_t low = ToNibble(static_cast<int>(low_steps));
        uint8_t high = ToNibble(static_cast<int>(high_steps));
        packed_row[col / 2] = static_cast<uint8_t>(low | (high << 4));
      }
    }
  }
}

void DequantizeInt4GroupWise(const uint8_t* packed, const float* scales,
                             int64_t rows, int64_t cols, int64_t group_size,
                             float* out_weights) {
  CheckInt4Preconditions(cols, group_size);
  int64_t groups_per_row = cols / group_size;
  int64_t bytes_per_row = cols / 2;

  for (int64_t r = 0; r < rows; ++r) {
    const uint8_t* packed_row = packed + r * bytes_per_row;
    float* out_row = out_weights + r * cols;

    for (int64_t g = 0; g < groups_per_row; ++g) {
      float scale = scales[r * groups_per_row + g];
      for (int64_t i = 0; i < group_size; i += 2) {
        int64_t col = g * group_size + i;
        uint8_t byte = packed_row[col / 2];
        int low = FromNibble(byte & 0x0F);
        int high = FromNibble((byte >> 4) & 0x0F);
        out_row[col] = low * scale;
        out_row[col + 1] = high * scale;
      }
    }
  }
}

void Int8GemmBT(const int8_t* a_quantized, const float* a_scales,
                const int8_t* b_quantized_transposed, const float* b_scales,
                float* c, int64_t M, int64_t K, int64_t N) {
  for (int64_t m = 0; m < M; ++m) {
    const int8_t* a_row = a_quantized + m * K;
    float a_scale = a_scales[m];

    for (int64_t n = 0; n < N; ++n) {
      const int8_t* b_row = b_quantized_transposed + n * K;

      // Add up the products as whole numbers. Each product is at most
      // 127 * 127, so a 32-bit total cannot overflow for any real model.
      int32_t acc = 0;
      for (int64_t k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_row[k]);
      }

      // Convert to a decimal once per output, after the loop, so the inner
      // loop never touches a float.
      c[m * N + n] = static_cast<float>(acc) * a_scale * b_scales[n];
    }
  }
}

}  // namespace kiln
