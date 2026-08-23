#pragma once

#include <cstdint>

namespace kiln {

// The result of one reproducible, synthetic prefix-cache workload. A lookup is
// one logical prefix block a new sequence needs; it is a hit when that block
// already exists in the paged cache and the sequence can share it.
struct PrefixCacheMeasurement {
  int64_t prefix_block_lookups = 0;
  int64_t prefix_block_hits = 0;
  int64_t copy_on_write_events = 0;

  double hit_rate() const {
    return prefix_block_lookups == 0
               ? 0.0
               : static_cast<double>(prefix_block_hits) / prefix_block_lookups;
  }
};

// Simulates conversations that share a 16-token system prompt and one of four
// 20-token prompt variants. Each conversation then writes a seeded number of
// divergent tokens, so the partially filled final prefix block takes the real
// copy-on-write path. This is deliberately synthetic: it measures the cache
// mechanism without implying that Kiln has received production traffic.
PrefixCacheMeasurement MeasureSyntheticPrefixCacheWorkload(
    uint32_t seed, int64_t conversations);

}  // namespace kiln
