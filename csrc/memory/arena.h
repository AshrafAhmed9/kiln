#pragma once
#include <cstddef>
#include <vector>

namespace kiln {

// A bump allocator: hands out sequential slices of one preallocated block.
// Nothing is freed individually -- the whole arena is reset or destroyed at
// once. This is the shape every later allocator in Kiln builds on (the
// paged KV cache is a block-table variant of the same idea), so it starts
// simple and stays legible.
class Arena {
 public:
  explicit Arena(size_t capacity_bytes);

  // Returns a pointer to `size` bytes, or a null pointer if there isn't
  // enough room left. The caller has to check for that null pointer --
  // this function never silently writes past the end of its block.
  void* Allocate(size_t size);

  size_t used() const { return offset_; }
  size_t capacity() const { return storage_.size(); }

  // Rewinds the arena so all prior allocations become invalid. Callers must
  // not touch pointers returned before Reset() after calling it.
  void Reset();

 private:
  std::vector<std::byte> storage_;
  size_t offset_ = 0;
};

}  // namespace kiln
