#include "kv/paged_kv_cache.h"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <stdexcept>

namespace kiln {
namespace {

TEST(PagedKVCache, AllocatingAndFreeingReturnsToTheSamePoolSize) {
  PagedKVCache cache(/*n_layers=*/2, /*num_blocks=*/4, /*block_size=*/2,
                     /*n_kv_heads=*/1, /*head_dim=*/2);
  ASSERT_EQ(cache.num_free_blocks(), 4);

  int64_t a = cache.AllocateBlock();
  int64_t b = cache.AllocateBlock();
  EXPECT_EQ(cache.num_free_blocks(), 2);
  EXPECT_NE(a,
            b);  // no double-allocation: two calls never return the same block

  cache.DecRef(a);
  cache.DecRef(b);
  EXPECT_EQ(cache.num_free_blocks(),
            4);  // no leak: every freed block comes back
}

TEST(PagedKVCache,
     DoubleFreeingABlockThrowsInsteadOfSilentlyCorruptingRefCounts) {
  PagedKVCache cache(1, 4, 2, 1, 2);
  int64_t block = cache.AllocateBlock();
  cache.DecRef(block);  // returns it to the pool -- ref count is now 0

  // Freeing it again is a real caller bug (a double-free): this must be
  // reported the same way in every build, not just crash an assert in
  // debug builds while silently corrupting the free pool in release ones
  // (the exact mistake already made and fixed once in Arena -- see
  // docs/correctness.md).
  EXPECT_THROW(cache.DecRef(block), std::runtime_error);
}

TEST(PagedKVCache, RunningOutOfBlocksFailsClosedInsteadOfCrashing) {
  PagedKVCache cache(1, /*num_blocks=*/1, 2, 1, 2);
  EXPECT_NE(cache.AllocateBlock(), -1);
  EXPECT_EQ(cache.AllocateBlock(),
            -1);  // the pool is empty -- reported, not crashed into
}

// A randomized property test in the same spirit as the scheduler's: throw
// a long, seeded sequence of allocate/free operations at the cache and
// check, after every single one, that the books still balance -- every
// block is either free or owned by exactly the sequences that hold it,
// and the pool never "loses" or "duplicates" a block.
TEST(PagedKVCache, BookkeepingStaysConsistentUnderRandomOperations) {
  const int64_t num_blocks = 20;
  PagedKVCache cache(1, num_blocks, 4, 1, 2);
  std::mt19937 rng(99);
  std::vector<int64_t> held_blocks;

  for (int step = 0; step < 500; ++step) {
    bool should_allocate = held_blocks.empty() || (rng() % 2 == 0);
    if (should_allocate) {
      int64_t block = cache.AllocateBlock();
      if (block != -1) {
        EXPECT_EQ(cache.RefCount(block), 1);
        held_blocks.push_back(block);
      }
    } else {
      size_t index = rng() % held_blocks.size();
      cache.DecRef(held_blocks[index]);
      held_blocks.erase(held_blocks.begin() + index);
    }
    EXPECT_EQ(
        cache.num_free_blocks() + static_cast<int64_t>(held_blocks.size()),
        num_blocks);
  }

  for (int64_t block : held_blocks) cache.DecRef(block);
  EXPECT_EQ(cache.num_free_blocks(), num_blocks);
}

TEST(PagedSequence, ForkedSequenceSharesParentsBlocksWithoutCopying) {
  PagedKVCache cache(1, 4, 2, 1, 2);
  PagedSequence parent = PagedSequence::Fresh(&cache);

  float k[2] = {1.0f, 2.0f};
  float v[2] = {3.0f, 4.0f};
  auto [block, slot] = parent.PrepareWriteSlot();
  std::memcpy(cache.K(0, block) + slot * 2, k, sizeof(k));
  std::memcpy(cache.V(0, block) + slot * 2, v, sizeof(v));
  parent.CommitToken();

  PagedSequence child = PagedSequence::Fork(parent);

  // Forking shouldn't have allocated any new blocks -- it should just be
  // sharing the parent's existing one.
  EXPECT_EQ(child.block_table(), parent.block_table());
  EXPECT_EQ(cache.RefCount(parent.block_table()[0]), 2);

  parent.Release();
  child.Release();
}

// The real correctness property of copy-on-write: after two sequences
// fork from a shared prefix and then each write their own new token, each
// one must see only its own new data -- and critically, the ORIGINAL
// shared prefix data must still be exactly what it was, undisturbed by
// either descendant's write.
TEST(PagedSequence, DivergingAfterForkNeverCorruptsTheOtherSequence) {
  PagedKVCache cache(
      1, 8, 4, 1,
      2);  // block_size 4, so there's room to grow past the shared prefix
  PagedSequence parent = PagedSequence::Fresh(&cache);

  float shared_k[2] = {10.0f, 20.0f};
  float shared_v[2] = {30.0f, 40.0f};
  auto [block, slot] = parent.PrepareWriteSlot();
  std::memcpy(cache.K(0, block) + slot * 2, shared_k, sizeof(shared_k));
  std::memcpy(cache.V(0, block) + slot * 2, shared_v, sizeof(shared_v));
  parent.CommitToken();

  PagedSequence child = PagedSequence::Fork(parent);

  float parent_k[2] = {100.0f, 200.0f};
  auto [p_block, p_slot] = parent.PrepareWriteSlot();
  std::memcpy(cache.K(0, p_block) + p_slot * 2, parent_k, sizeof(parent_k));
  parent.CommitToken();

  float child_k[2] = {-1.0f, -2.0f};
  auto [c_block, c_slot] = child.PrepareWriteSlot();
  std::memcpy(cache.K(0, c_block) + c_slot * 2, child_k, sizeof(child_k));
  child.CommitToken();

  // Each sequence's own new token must be exactly what it wrote.
  const float* parent_second_token = cache.K(0, p_block) + p_slot * 2;
  EXPECT_FLOAT_EQ(parent_second_token[0], 100.0f);
  const float* child_second_token = cache.K(0, c_block) + c_slot * 2;
  EXPECT_FLOAT_EQ(child_second_token[0], -1.0f);

  // Both sequences must still see the ORIGINAL shared first token,
  // unaffected by either one's second write -- this is the entire promise
  // copy-on-write exists to keep.
  int64_t parent_first_block = parent.block_table()[0];
  int64_t child_first_block = child.block_table()[0];
  const float* parent_first_token = cache.K(0, parent_first_block);
  const float* child_first_token = cache.K(0, child_first_block);
  EXPECT_FLOAT_EQ(parent_first_token[0], 10.0f);
  EXPECT_FLOAT_EQ(child_first_token[0], 10.0f);

  parent.Release();
  child.Release();
}

}  // namespace
}  // namespace kiln
