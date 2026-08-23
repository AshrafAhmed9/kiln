#include "tokenizer/bpe.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

namespace kiln {
namespace {

// A tiny hand-built tokenizer.json: vocab covers every byte-level symbol
// for 'a','b','c',' ' plus two merged tokens, so the merge loop actually
// exercises multi-step merging. This is NOT a HuggingFace conformance test
// (that needs the real fixture -- see docs/learning/phase-01.md); it tests
// that our own merge algorithm is internally consistent: encode then decode
// returns the original string, and a known merge sequence produces the
// expected token count.
std::string WriteTestTokenizer() {
  std::string path = "/tmp/kiln_test_tokenizer.json";
  // Merges: "a"+"b" -> "ab" (rank 0), "ab"+"c" -> "abc" (rank 1).
  std::string json = R"({
    "model": {
      "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
      "merges": ["a b", "ab c"]
    }
  })";
  std::ofstream out(path);
  out << json;
  return path;
}

TEST(BpeTokenizer, MergesInRankOrder) {
  std::string path = WriteTestTokenizer();
  BpeTokenizer tok = BpeTokenizer::Load(path);

  std::vector<int32_t> ids = tok.Encode("abc");
  // "a","b","c" should fully merge down to the single token "abc" (id 4),
  // since both merges apply in rank order within one pre-token chunk.
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 4);

  std::remove(path.c_str());
}

TEST(BpeTokenizer, EncodeDecodeRoundTrips) {
  std::string path = WriteTestTokenizer();
  BpeTokenizer tok = BpeTokenizer::Load(path);

  std::string original = "abc";
  std::vector<int32_t> ids = tok.Encode(original);
  std::string decoded = tok.Decode(ids);
  EXPECT_EQ(decoded, original);

  std::remove(path.c_str());
}

TEST(BpeTokenizer, NoMergeAvailableLeavesSymbolsSeparate) {
  std::string path = WriteTestTokenizer();
  BpeTokenizer tok = BpeTokenizer::Load(path);

  // "c" alone has no merge partner -- should stay one token, id 2.
  std::vector<int32_t> ids = tok.Encode("c");
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], 2);

  std::remove(path.c_str());
}

}  // namespace
}  // namespace kiln
