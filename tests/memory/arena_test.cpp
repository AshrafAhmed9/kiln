#include "memory/arena.h"

#include <gtest/gtest.h>

namespace kiln {

TEST(Arena, AllocatesSequentially) {
  Arena arena(64);
  void* a = arena.Allocate(16);
  void* b = arena.Allocate(16);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(static_cast<std::byte*>(b) - static_cast<std::byte*>(a), 16);
  EXPECT_EQ(arena.used(), 32u);
}

TEST(Arena, ReturnsNullWhenExhausted) {
  Arena arena(16);
  EXPECT_NE(arena.Allocate(16), nullptr);
  EXPECT_EQ(arena.Allocate(1), nullptr);
}

TEST(Arena, ResetReclaimsSpace) {
  Arena arena(16);
  arena.Allocate(16);
  arena.Reset();
  EXPECT_EQ(arena.used(), 0u);
  EXPECT_NE(arena.Allocate(16), nullptr);
}

}  // namespace kiln
