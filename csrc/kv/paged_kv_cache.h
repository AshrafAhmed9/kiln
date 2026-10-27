#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// A KV cache built from small fixed-size blocks, like pages in an operating
// system. All memory is cut into blocks up front, and each text takes
// blocks from a shared pool only as it grows. Nothing is reserved for the
// longest possible text.
//
// The big benefit: texts that start with the same prompt can point at the
// same blocks instead of each keeping a copy. When one of them needs to
// change a shared block, it first makes its own private copy
// (copy-on-write). A block id means the same block in every layer, so one
// block table covers all layers.
class PagedKVCache {
 public:
  PagedKVCache(int64_t n_layers, int64_t num_blocks, int64_t block_size,
               int64_t n_kv_heads, int64_t head_dim);

  int64_t block_size() const { return block_size_; }
  int64_t num_free_blocks() const { return free_blocks_.size(); }

  // Takes one block from the pool, owned by one text so far. Returns -1 if
  // the pool is empty, so callers must check for that.
  int64_t AllocateBlock();

  // Records that one more text is sharing this block.
  void IncRef(int64_t block_id);

  // Records that one text stopped using this block. When nobody uses it
  // any more, it goes back into the pool.
  void DecRef(int64_t block_id);

  int64_t RefCount(int64_t block_id) const { return ref_counts_[block_id]; }

  float* K(int64_t layer, int64_t block_id);
  float* V(int64_t layer, int64_t block_id);
  const float* K(int64_t layer, int64_t block_id) const;
  const float* V(int64_t layer, int64_t block_id) const;

  // Copies one block's keys and values (all layers) into another block.
  // This is the "copy" in copy-on-write.
  void CopyBlockContents(int64_t from_block, int64_t to_block);

 private:
  int64_t n_layers_;
  int64_t block_size_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  std::vector<int64_t> ref_counts_;
  std::vector<int64_t> free_blocks_;
  // k_[layer] holds every block's keys for that layer, one block after
  // another. v_ is the same for values.

  // How many floats one block holds in one layer.
  size_t FloatsPerBlock() const {
    return static_cast<size_t>(block_size_ * n_kv_heads_ * head_dim_);
  }

  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
};

// One text's view into the paged cache: the list of blocks it uses, in
// order, and how many words it holds.
class PagedSequence {
 public:
  PagedSequence(PagedKVCache* cache);

  // Starts an empty text that shares nothing.
  static PagedSequence Fresh(PagedKVCache* cache);

  // Starts a new text that shares every block `parent` has. This is how
  // texts with the same prompt share memory. Nothing is copied yet; a copy
  // only happens when one of them writes into a shared block.
  static PagedSequence Fork(const PagedSequence& parent);

  int64_t length() const { return length_; }

  // Finds a private slot for the next word's keys and values. If the last
  // block is shared with another text, it copies that block first. Returns
  // (block_id, slot_within_block) to write into, for every layer.
  std::pair<int64_t, int64_t> PrepareWriteSlot();

  // Call once after writing the new word into every layer at the slot
  // PrepareWriteSlot() returned.
  void CommitToken() { ++length_; }

  // Gives back every block this text uses, so others can reuse them.
  void Release();

  const std::vector<int64_t>& block_table() const { return block_table_; }

 private:
  PagedKVCache* cache_;
  std::vector<int64_t> block_table_;
  int64_t length_ = 0;
};

}  // namespace kiln
