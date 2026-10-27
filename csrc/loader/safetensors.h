#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kiln {

enum class DType { kF32, kF16, kBF16 };

// One tensor inside the file. `data` points straight into the memory-mapped
// file (nothing is copied), so it is only valid while the SafetensorsFile
// that produced it is alive.
struct TensorView {
  const std::byte* data;
  DType dtype;
  std::vector<int64_t> shape;
};

// Reads a .safetensors file: 8 bytes giving the header length, a JSON header
// describing each tensor (type, shape, where its bytes are), then the raw
// bytes. The file is memory-mapped instead of read into memory.
class SafetensorsFile {
 public:
  // Throws with a specific reason if the file is damaged (cut-off header,
  // bad JSON, or a tensor that runs past the end of the data).
  static SafetensorsFile Load(const std::string& path);

  ~SafetensorsFile();
  SafetensorsFile(SafetensorsFile&&) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&&) noexcept;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  const TensorView& Tensor(const std::string& name) const;
  bool HasTensor(const std::string& name) const;
  const std::unordered_map<std::string, TensorView>& tensors() const {
    return tensors_;
  }

 private:
  SafetensorsFile() = default;

  void* mapped_ = nullptr;  // where the file is mapped in memory
  size_t mapped_size_ = 0;
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace kiln
