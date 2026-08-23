#include "kv/paged_kv_cache.h"

#include <cstring>
#include <stdexcept>

namespace kiln {

PagedKVCache::PagedKVCache(int64_t n_layers, int64_t num_blocks,
                            int64_t block_size, int64_t n_kv_heads,
                            int64_t head_dim)
    : n_layers_(n_layers),
      block_size_(block_size),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim) {
  ref_counts_.assign(num_blocks, 0);
  free_blocks_.reserve(num_blocks);
  for (int64_t i = num_blocks - 1; i >= 0; --i) free_blocks_.push_back(i);

  size_t floats_per_block =
      static_cast<size_t>(block_size * n_kv_heads * head_dim);
  k_.assign(n_layers, std::vector<float>(num_blocks * floats_per_block));
  v_.assign(n_layers, std::vector<float>(num_blocks * floats_per_block));
}

int64_t PagedKVCache::AllocateBlock() {
  if (free_blocks_.empty()) return -1;
  int64_t block_id = free_blocks_.back();
  free_blocks_.pop_back();
  ref_counts_[block_id] = 1;
  return block_id;
}

void PagedKVCache::IncRef(int64_t block_id) { ++ref_counts_[block_id]; }

void PagedKVCache::DecRef(int64_t block_id) {
  // This mirrors a mistake already made and fixed once in this project
  // (Arena::Allocate, Phase 0 -- see docs/correctness.md): a debug-only
  // assert here would mean debug builds crash loudly on a double-free
  // while release builds silently let the reference count go negative
  // and never give the block back to the free pool. One rule, in every
  // build: a double-free is a real caller bug, so it's reported the same
  // way every time, not just when assertions happen to be compiled in.
  if (ref_counts_[block_id] <= 0) {
    throw std::runtime_error(
        "PagedKVCache::DecRef called on a block nobody currently owns -- "
        "this would double-free a block");
  }
  --ref_counts_[block_id];
  if (ref_counts_[block_id] == 0) free_blocks_.push_back(block_id);
}

float* PagedKVCache::K(int64_t layer, int64_t block_id) {
  size_t floats_per_block =
      static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_);
  return k_[layer].data() + block_id * floats_per_block;
}

float* PagedKVCache::V(int64_t layer, int64_t block_id) {
  size_t floats_per_block =
      static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_);
  return v_[layer].data() + block_id * floats_per_block;
}

const float* PagedKVCache::K(int64_t layer, int64_t block_id) const {
  size_t floats_per_block =
      static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_);
  return k_[layer].data() + block_id * floats_per_block;
}

const float* PagedKVCache::V(int64_t layer, int64_t block_id) const {
  size_t floats_per_block =
      static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_);
  return v_[layer].data() + block_id * floats_per_block;
}

void PagedKVCache::CopyBlockContents(int64_t from_block, int64_t to_block) {
  size_t bytes_per_block =
      static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_) *
      sizeof(float);
  for (int64_t layer = 0; layer < n_layers_; ++layer) {
    std::memcpy(K(layer, to_block), K(layer, from_block), bytes_per_block);
    std::memcpy(V(layer, to_block), V(layer, from_block), bytes_per_block);
  }
}

PagedSequence::PagedSequence(PagedKVCache* cache) : cache_(cache) {}

PagedSequence PagedSequence::Fresh(PagedKVCache* cache) {
  return PagedSequence(cache);
}

PagedSequence PagedSequence::Fork(const PagedSequence& parent) {
  PagedSequence child(parent.cache_);
  child.block_table_ = parent.block_table_;
  child.length_ = parent.length_;
  for (int64_t block_id : child.block_table_) {
    child.cache_->IncRef(block_id);
  }
  return child;
}

std::pair<int64_t, int64_t> PagedSequence::PrepareWriteSlot() {
  int64_t slot_in_block = length_ % cache_->block_size();
  bool needs_new_block = (slot_in_block == 0);

  if (needs_new_block) {
    int64_t new_block = cache_->AllocateBlock();
    if (new_block < 0) {
      throw std::runtime_error(
          "PagedKVCache: out of physical blocks -- the caller must not "
          "have admitted more sequences than the cache has room for");
    }
    block_table_.push_back(new_block);
    return {new_block, 0};
  }

  int64_t current_block = block_table_.back();
  if (cache_->RefCount(current_block) > 1) {
    // Someone else still shares this block -- copying it now, before
    // writing, is the entire copy-on-write mechanism: this sequence gets
    // its own private copy, the other sequence keeps using the original
    // untouched, and no data either of them already wrote is disturbed.
    int64_t private_block = cache_->AllocateBlock();
    if (private_block < 0) {
      throw std::runtime_error(
          "PagedKVCache: out of physical blocks during copy-on-write");
    }
    cache_->CopyBlockContents(current_block, private_block);
    cache_->DecRef(current_block);
    block_table_.back() = private_block;
    return {private_block, slot_in_block};
  }

  return {current_block, slot_in_block};
}

void PagedSequence::Release() {
  for (int64_t block_id : block_table_) cache_->DecRef(block_id);
  block_table_.clear();
  length_ = 0;
}

}  // namespace kiln
