#include "memory/arena.h"

namespace kiln {

Arena::Arena(size_t capacity_bytes) : storage_(capacity_bytes) {}

void* Arena::Allocate(size_t size) {
  // If there isn't enough room left, hand back nothing rather than writing
  // past the end of our block. The caller is expected to check for this
  // and either reuse space (via Reset) or fail the request -- the same way
  // any allocator says "no" instead of quietly corrupting memory.
  if (offset_ + size > storage_.size()) return nullptr;

  void* ptr = storage_.data() + offset_;
  offset_ += size;
  return ptr;
}

void Arena::Reset() { offset_ = 0; }

}  // namespace kiln
