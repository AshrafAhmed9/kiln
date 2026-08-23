#include "quant/quantize.h"

#include <algorithm>
#include <cmath>

namespace kiln {

void QuantizeInt8PerChannel(const float* weights, int64_t rows, int64_t cols,
                            int8_t* out_quantized, float* out_scales) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = weights + r * cols;
    float max_abs = 0.0f;
    for (int64_t c = 0; c < cols; ++c) max_abs = std::max(max_abs, std::abs(row[c]));

    // If every weight in this row happens to be zero, any nonzero scale
    // works (0 * anything is still 0) -- 1.0 avoids a division by zero
    // below without changing the (all-zero) result.
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

// Stores a value from [-8, 7] as a small non-negative number in [0, 15] by
// shifting it up by 8 -- this is what makes it fit cleanly in 4 bits
// without needing any sign-handling logic when packing or unpacking.
uint8_t ToNibble(int value) { return static_cast<uint8_t>(value + 8); }
int FromNibble(uint8_t nibble) { return static_cast<int>(nibble) - 8; }

}  // namespace

void QuantizeInt4GroupWise(const float* weights, int64_t rows, int64_t cols,
                          int64_t group_size, uint8_t* out_packed,
                          float* out_scales) {
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

}  // namespace kiln
