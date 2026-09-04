#include "loader/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace kiln {

namespace {

DType ParseDType(const std::string& s) {
  if (s == "F32") return DType::kF32;
  if (s == "F16") return DType::kF16;
  if (s == "BF16") return DType::kBF16;
  throw std::runtime_error("safetensors: unsupported dtype '" + s + "'");
}

}  // namespace

SafetensorsFile SafetensorsFile::Load(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path);

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("safetensors: fstat failed for " + path);
  }
  size_t file_size = static_cast<size_t>(st.st_size);

  if (file_size < 8) {
    close(fd);
    throw std::runtime_error("safetensors: file too small for a header: " +
                             path);
  }

  void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);  // the mapping keeps the file open; the fd itself is not needed
  if (mapped == MAP_FAILED) {
    throw std::runtime_error("safetensors: mmap failed for " + path);
  }

  const auto* bytes = static_cast<const unsigned char*>(mapped);
  uint64_t header_len;
  std::memcpy(&header_len, bytes, sizeof(header_len));

  if (header_len > file_size - 8) {
    munmap(mapped, file_size);
    throw std::runtime_error(
        "safetensors: declared header length runs past end of file: " + path);
  }

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(bytes + 8, bytes + 8 + header_len);
  } catch (const nlohmann::json::exception& e) {
    munmap(mapped, file_size);
    throw std::runtime_error("safetensors: header is not valid JSON in " +
                             path + ": " + e.what());
  }

  size_t data_region_start = 8 + header_len;
  size_t data_region_size = file_size - data_region_start;

  SafetensorsFile result;
  result.mapped_ = mapped;
  result.mapped_size_ = file_size;

  for (auto& [name, meta] : header.items()) {
    if (name == "__metadata__") continue;

    if (!meta.contains("data_offsets") || !meta.contains("shape") ||
        !meta.contains("dtype")) {
      munmap(mapped, file_size);
      throw std::runtime_error("safetensors: tensor '" + name +
                               "' is missing a required field in " + path);
    }

    uint64_t start = meta["data_offsets"][0].get<uint64_t>();
    uint64_t end = meta["data_offsets"][1].get<uint64_t>();
    if (end < start || end > data_region_size) {
      munmap(mapped, file_size);
      throw std::runtime_error("safetensors: tensor '" + name +
                               "' offsets run past the data region in " + path);
    }

    TensorView view;
    view.data =
        reinterpret_cast<const std::byte*>(bytes) + data_region_start + start;
    view.dtype = ParseDType(meta["dtype"].get<std::string>());
    for (const auto& dim : meta["shape"]) {
      view.shape.push_back(dim.get<int64_t>());
    }
    result.tensors_.emplace(name, std::move(view));
  }

  return result;
}

SafetensorsFile::~SafetensorsFile() {
  if (mapped_) munmap(mapped_, mapped_size_);
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : mapped_(other.mapped_),
      mapped_size_(other.mapped_size_),
      tensors_(std::move(other.tensors_)) {
  other.mapped_ = nullptr;
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
  if (this != &other) {
    if (mapped_) munmap(mapped_, mapped_size_);
    mapped_ = other.mapped_;
    mapped_size_ = other.mapped_size_;
    tensors_ = std::move(other.tensors_);
    other.mapped_ = nullptr;
  }
  return *this;
}

const TensorView& SafetensorsFile::Tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("safetensors: no tensor named '" + name + "'");
  }
  return it->second;
}

bool SafetensorsFile::HasTensor(const std::string& name) const {
  return tensors_.find(name) != tensors_.end();
}

}  // namespace kiln
