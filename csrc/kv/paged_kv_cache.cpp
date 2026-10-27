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

  k_.assign(n_layers, std::vector<float>(num_blocks * FloatsPerBlock()));
  v_.assign(n_layers, std::vector<float>(num_blocks * FloatsPerBlock()));
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
  // Freeing a block nobody owns is a bug in the caller. Throw in every
  // build, instead of using assert(), which release builds skip. Skipping it
  // would let the count go negative and lose the block.
  if (ref_counts_[block_id] <= 0) {
    throw std::runtime_error(
        "PagedKVCache::DecRef: block is not owned by anyone (double free)");
  }
  --ref_counts_[block_id];
  if (ref_counts_[block_id] == 0) free_blocks_.push_back(block_id);
}

float* PagedKVCache::K(int64_t layer, int64_t block_id) {
  return k_[layer].data() + block_id * FloatsPerBlock();
}

float* PagedKVCache::V(int64_t layer, int64_t block_id) {
  return v_[layer].data() + block_id * FloatsPerBlock();
}

const float* PagedKVCache::K(int64_t layer, int64_t block_id) const {
  return k_[layer].data() + block_id * FloatsPerBlock();
}

const float* PagedKVCache::V(int64_t layer, int64_t block_id) const {
  return v_[layer].data() + block_id * FloatsPerBlock();
}

void PagedKVCache::CopyBlockContents(int64_t from_block, int64_t to_block) {
  size_t bytes_per_block = FloatsPerBlock() * sizeof(float);
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
      throw std::runtime_error("PagedKVCache: out of blocks");
    }
    block_table_.push_back(new_block);
    return {new_block, 0};
  }

  int64_t current_block = block_table_.back();
  if (cache_->RefCount(current_block) > 1) {
    // Another text shares this block. Copy it before writing: this text
    // gets a private copy, and the other keeps the original untouched.
    // That is copy-on-write.
    int64_t private_block = cache_->AllocateBlock();
    if (private_block < 0) {
      throw std::runtime_error(
          "PagedKVCache: out of blocks during copy-on-write");
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
