#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// A paged KV cache: all of the cache's memory is carved into fixed-size
// blocks up front, and sequences pick up new blocks from a shared pool as
// they grow, instead of each sequence getting one giant pre-reserved
// chunk (the way the simpler Phase 3 KVCache works). See
// docs/learning/phase-08.md for why this is the right generalization and
// what it makes possible (no wasted worst-case reservation, and sharing
// an identical prompt's blocks between sequences instead of duplicating
// them).
//
// A block index means the same physical block across every layer -- one
// block table, shared by all layers, which is what keeps this simple.
class PagedKVCache {
 public:
  PagedKVCache(int64_t n_layers, int64_t num_blocks, int64_t block_size,
               int64_t n_kv_heads, int64_t head_dim);

  int64_t block_size() const { return block_size_; }
  int64_t num_free_blocks() const { return free_blocks_.size(); }

  // Hands out one block from the shared pool, marking it owned by exactly
  // one sequence so far. Returns -1 if the pool is empty -- callers must
  // check for this rather than assume a block is always available (the
  // same fail-closed discipline as every other allocator in this project).
  int64_t AllocateBlock();

  // Marks one more sequence as sharing this block (used when a new
  // sequence forks from an existing one and starts out pointing at the
  // same blocks).
  void IncRef(int64_t block_id);

  // Marks one sequence as no longer using this block. Once nobody is
  // using it any more, the block goes back into the shared pool so a
  // future AllocateBlock() call can reuse it.
  void DecRef(int64_t block_id);

  int64_t RefCount(int64_t block_id) const { return ref_counts_[block_id]; }

  float* K(int64_t layer, int64_t block_id);
  float* V(int64_t layer, int64_t block_id);
  const float* K(int64_t layer, int64_t block_id) const;
  const float* V(int64_t layer, int64_t block_id) const;

  // Copies one block's K and V numbers, at every layer, from `from_block`
  // into `to_block`. This is the actual data-moving half of copy-on-write
  // -- allocating a new block only makes room for a private copy; this is
  // what actually puts the old block's contents into it.
  void CopyBlockContents(int64_t from_block, int64_t to_block);

 private:
  int64_t n_layers_;
  int64_t block_size_;
  int64_t n_kv_heads_;
  int64_t head_dim_;
  std::vector<int64_t> ref_counts_;
  std::vector<int64_t> free_blocks_;
  // k_[layer] holds every block's K numbers for that layer, laid out one
  // block after another; same for v_.
  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
};

// One sequence's view into the paged cache: which physical blocks belong
// to it, in order, and how many of the current last block's slots are
// already filled.
class PagedSequence {
 public:
  PagedSequence(PagedKVCache* cache);

  // Starts a brand new sequence sharing no blocks with anyone.
  static PagedSequence Fresh(PagedKVCache* cache);

  // Starts a new sequence that begins by sharing every block `parent`
  // currently has -- the prefix-sharing case from docs/learning/phase-08.md.
  // Nothing is copied yet; sharing only becomes an actual copy the moment
  // either sequence needs to write somewhere it doesn't own alone.
  static PagedSequence Fork(const PagedSequence& parent);

  int64_t length() const { return length_; }

  // Ensures there's a private, writable slot for the next token, doing a
  // copy-on-write first if the current last block is shared with anyone
  // else. Returns (block_id, slot_within_block) for the caller to write
  // this token's K/V numbers into, for every layer.
  std::pair<int64_t, int64_t> PrepareWriteSlot();

  // Call once, after writing this token's K/V numbers into every layer at
  // the slot PrepareWriteSlot() returned.
  void CommitToken() { ++length_; }

  // Releases every block this sequence still owns -- call when the
  // sequence is done and its blocks (or its share of them) should become
  // reusable.
  void Release();

  const std::vector<int64_t>& block_table() const { return block_table_; }

 private:
  PagedKVCache* cache_;
  std::vector<int64_t> block_table_;
  int64_t length_ = 0;
};

}  // namespace kiln
