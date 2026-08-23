#include "kv/prefix_cache_measurement.h"

#include <iomanip>
#include <iostream>

int main() {
  constexpr uint32_t kSeed = 20260824;
  constexpr int64_t kConversations = 80;
  kiln::PrefixCacheMeasurement result =
      kiln::MeasureSyntheticPrefixCacheWorkload(kSeed, kConversations);

  std::cout << "Synthetic prefix-cache workload\n"
            << "  seed: " << kSeed << "\n"
            << "  conversations: " << kConversations << "\n"
            << "  prefix block lookups: " << result.prefix_block_lookups << "\n"
            << "  prefix block hits: " << result.prefix_block_hits << "\n"
            << "  copy-on-write events: " << result.copy_on_write_events << "\n"
            << "  synthetic prefix-block hit rate: " << std::fixed
            << std::setprecision(2) << result.hit_rate() * 100.0 << "%\n";
}
