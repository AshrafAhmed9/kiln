#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kiln {

enum class DType { kF32, kF16, kBF16 };

// A zero-copy view into one tensor's bytes inside the mmap'd file. `data`
// points directly into the mapping -- no allocation, no copy. Valid only as
// long as the owning SafetensorsFile is alive.
struct TensorView {
  const std::byte* data;
  DType dtype;
  std::vector<int64_t> shape;
};

// Parses a .safetensors file: an 8-byte little-endian header length, that
// many bytes of JSON describing each tensor's dtype/shape/byte-offsets, then
// the raw tensor bytes. Loads via mmap so TensorView::data is a direct
// pointer into the file -- see docs/learning/phase-01.md for why this
// format is a zero-copy fit.
class SafetensorsFile {
 public:
  // Throws std::runtime_error with a specific reason on any corruption:
  // truncated header, unparseable JSON, or a tensor whose offsets run past
  // the data region. Fails loudly at load time rather than handing back a
  // dangling view.
  static SafetensorsFile Load(const std::string& path);

  ~SafetensorsFile();
  SafetensorsFile(SafetensorsFile&&) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&&) noexcept;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  const TensorView& Tensor(const std::string& name) const;
  const std::unordered_map<std::string, TensorView>& tensors() const {
    return tensors_;
  }

 private:
  SafetensorsFile() = default;

  void* mapped_ = nullptr;   // mmap base address
  size_t mapped_size_ = 0;
  std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace kiln
