#include "memory/arena.h"

#include <cassert>

namespace kiln {

Arena::Arena(size_t capacity_bytes) : storage_(capacity_bytes) {}

void* Arena::Allocate(size_t size) {
  assert(offset_ + size <= storage_.size() &&
         "Arena exhausted -- every Kiln allocation must go through an "
         "arena, so a caller hitting this must raise the arena's capacity "
         "rather than reach for a bare `new`/`cudaMalloc`.");
  if (offset_ + size > storage_.size()) return nullptr;

  void* ptr = storage_.data() + offset_;
  offset_ += size;
  return ptr;
}

void Arena::Reset() { offset_ = 0; }

}  // namespace kiln
