#include "loader/safetensors.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace kiln {
namespace {

// Writes a minimal valid .safetensors file with one F32 tensor of the given
// values, so loader tests don't depend on any external fixture.
std::string WriteTestFile(const std::vector<float>& values,
                           const std::string& shape_json) {
  std::string path = "/tmp/kiln_test_" + std::to_string(rand()) + ".safetensors";
  std::string header = "{\"x\":{\"dtype\":\"F32\",\"shape\":" + shape_json +
                        ",\"data_offsets\":[0," +
                        std::to_string(values.size() * sizeof(float)) +
                        "]}}";
  uint64_t header_len = header.size();

  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), header.size());
  out.write(reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(float));
  out.close();
  return path;
}

TEST(Safetensors, RoundTripsOneTensor) {
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::string path = WriteTestFile(values, "[2,2]");

  SafetensorsFile file = SafetensorsFile::Load(path);
  EXPECT_TRUE(file.HasTensor("x"));
  EXPECT_FALSE(file.HasTensor("missing"));
  const TensorView& view = file.Tensor("x");

  EXPECT_EQ(view.dtype, DType::kF32);
  EXPECT_EQ(view.shape, (std::vector<int64_t>{2, 2}));
  const float* data = reinterpret_cast<const float*>(view.data);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], values[i]);
  }
  std::remove(path.c_str());
}

TEST(Safetensors, RejectsTruncatedHeader) {
  std::string path = "/tmp/kiln_test_truncated.safetensors";
  std::ofstream out(path, std::ios::binary);
  uint64_t huge_len = 1'000'000;
  out.write(reinterpret_cast<const char*>(&huge_len), sizeof(huge_len));
  out.write("short", 5);
  out.close();

  EXPECT_THROW(SafetensorsFile::Load(path), std::runtime_error);
  std::remove(path.c_str());
}

TEST(Safetensors, RejectsOffsetsPastDataRegion) {
  std::string path = "/tmp/kiln_test_badoffset.safetensors";
  std::string header =
      "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,999]}}";
  uint64_t header_len = header.size();

  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), header.size());
  float value = 1.0f;
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  out.close();

  EXPECT_THROW(SafetensorsFile::Load(path), std::runtime_error);
  std::remove(path.c_str());
}

}  // namespace
}  // namespace kiln
