#pragma once
#include <cstdint>
#include <cstring>

#include "loader/safetensors.h"

namespace kiln {

// Checkpoints store weights as 16-bit numbers (BF16 or FP16), but the CPU
// model computes in 32-bit FP32. These convert one number at a time. That is
// slow-ish but simple, and it only runs once when a model is loaded.

inline float BF16ToF32(uint16_t bits) {
  // BF16 is the top 16 bits of an FP32, so converting is a shift.
  uint32_t as_f32_bits = static_cast<uint32_t>(bits) << 16;
  float result;
  std::memcpy(&result, &as_f32_bits, sizeof(result));
  return result;
}

inline float F16ToF32(uint16_t bits) {
  uint32_t sign = (bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x3FFu;
  uint32_t result_bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      result_bits = sign;  // signed zero
    } else {
      // A tiny FP16 number (subnormal): shift it into FP32's normal range.
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3FFu;
      uint32_t f32_exponent = 127 - 15 - shift;
      result_bits = sign | (f32_exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    result_bits = sign | 0x7F800000u | (mantissa << 13);  // inf/NaN
  } else {
    uint32_t f32_exponent = exponent - 15 + 127;
    result_bits = sign | (f32_exponent << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &result_bits, sizeof(result));
  return result;
}

// Converts one tensor's raw bytes to FP32. `out` must have room for
// `element_count` floats.
inline void ConvertToF32(const TensorView& view, size_t element_count,
                         float* out) {
  if (view.dtype == DType::kF32) {
    std::memcpy(out, view.data, element_count * sizeof(float));
    return;
  }
  const auto* raw = reinterpret_cast<const uint16_t*>(view.data);
  for (size_t i = 0; i < element_count; ++i) {
    out[i] =
        (view.dtype == DType::kBF16) ? BF16ToF32(raw[i]) : F16ToF32(raw[i]);
  }
}

}  // namespace kiln
