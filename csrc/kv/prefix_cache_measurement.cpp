#include "kv/prefix_cache_measurement.h"

#include "kv/paged_kv_cache.h"

#include <array>
#include <memory>
#include <random>
#include <stdexcept>

namespace kiln {
namespace {

void AppendTokens(PagedSequence* sequence, int64_t count) {
  for (int64_t token = 0; token < count; ++token) {
    sequence->PrepareWriteSlot();
    sequence->CommitToken();
  }
}

}  // namespace

PrefixCacheMeasurement MeasureSyntheticPrefixCacheWorkload(
    uint32_t seed, int64_t conversations) {
  if (conversations < 0) {
    throw std::invalid_argument("conversation count must not be negative");
  }

  constexpr int64_t kBlockSize = 8;
  constexpr int64_t kVariantCount = 4;
  // The workload has five long-lived prefix blocks plus at most one private
  // copy-on-write block for the currently simulated conversation.
  PagedKVCache cache(/*n_layers=*/1, /*num_blocks=*/16, kBlockSize,
                     /*n_kv_heads=*/1, /*head_dim=*/1);
  PagedSequence shared_system = PagedSequence::Fresh(&cache);
  AppendTokens(&shared_system, /*count=*/16);

  std::array<std::unique_ptr<PagedSequence>, kVariantCount> variants;
  PrefixCacheMeasurement result;
  std::mt19937 rng(seed);

  for (int64_t conversation = 0; conversation < conversations; ++conversation) {
    int64_t variant = static_cast<int64_t>(rng() % kVariantCount);
    if (!variants[variant]) {
      // Creating a variant reuses the two system-prompt blocks, then makes one
      // partially filled variant block that later conversations can share.
      result.prefix_block_lookups += shared_system.block_table().size();
      result.prefix_block_hits += shared_system.block_table().size();
      variants[variant] = std::make_unique<PagedSequence>(
          PagedSequence::Fork(shared_system));
      AppendTokens(variants[variant].get(), /*count=*/4);
      ++result.prefix_block_lookups;
    }

    PagedSequence request = PagedSequence::Fork(*variants[variant]);
    result.prefix_block_lookups += request.block_table().size();
    result.prefix_block_hits += request.block_table().size();

    // The variant's final block contains four of eight slots, so the first
    // write must copy it. Later writes stay private. The random length makes
    // the workload varied while the seed makes its result reproducible.
    int64_t shared_block = request.block_table().back();
    int64_t divergent_tokens = 1 + static_cast<int64_t>(rng() % 4);
    AppendTokens(&request, divergent_tokens);
    if (request.block_table().back() != shared_block) ++result.copy_on_write_events;
    request.Release();
  }

  for (auto& variant : variants) {
    if (variant) variant->Release();
  }
  shared_system.Release();
  return result;
}

}  // namespace kiln
