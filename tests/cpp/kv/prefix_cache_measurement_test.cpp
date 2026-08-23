#include "kv/prefix_cache_measurement.h"

#include <gtest/gtest.h>

namespace kiln {
namespace {

TEST(PrefixCacheMeasurement, SeededSyntheticWorkloadReportsActualSharedBlocks) {
  PrefixCacheMeasurement result =
      MeasureSyntheticPrefixCacheWorkload(/*seed=*/20260824,
                                          /*conversations=*/80);

  // Four variants each introduce one new partial block. Every request then
  // shares its variant's three prefix blocks before diverging, which makes the
  // expected accounting simple enough to catch regressions in the workload.
  EXPECT_EQ(result.prefix_block_lookups, 252);
  EXPECT_EQ(result.prefix_block_hits, 248);
  EXPECT_EQ(result.copy_on_write_events, 80);
  EXPECT_DOUBLE_EQ(result.hit_rate(), 248.0 / 252.0);
}

TEST(PrefixCacheMeasurement, EmptyWorkloadHasNoInventedRate) {
  PrefixCacheMeasurement result =
      MeasureSyntheticPrefixCacheWorkload(/*seed=*/1, /*conversations=*/0);

  EXPECT_EQ(result.prefix_block_lookups, 0);
  EXPECT_EQ(result.prefix_block_hits, 0);
  EXPECT_EQ(result.copy_on_write_events, 0);
  EXPECT_DOUBLE_EQ(result.hit_rate(), 0.0);
}

}  // namespace
}  // namespace kiln
