#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kiln {

// Byte-level BPE, loaded from a HuggingFace tokenizer.json (vocab + an
// ordered merge list). See docs/learning/phase-01.md for why merges are
// applied in rank order, not sequence order, and why the byte-level
// remapping makes the tokenizer total over all possible byte strings.
class BpeTokenizer {
 public:
  static BpeTokenizer Load(const std::string& tokenizer_json_path);

  std::vector<int32_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int32_t>& ids) const;

 private:
  BpeTokenizer() = default;

  // Applies the merge loop to one pre-token (a run of byte-level symbols
  // with no BPE applied yet), returning the final list of vocab IDs.
  std::vector<int32_t> BpeMergeWord(
      const std::vector<std::string>& symbols) const;

  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  // merge_rank_[{a, b}] = position in the training merge list (lower =
  // higher priority). Looked up per adjacent pair during encoding.
  std::unordered_map<std::string, int32_t> merge_rank_;  // key: "a\x00b"

  // The 256-entry byte <-> printable-unicode-character table (GPT-2 style):
  // every raw byte maps to some visible character, so any byte string has a
  // representation and round-trips exactly through Decode(Encode(x)).
  std::unordered_map<uint8_t, std::string> byte_to_unicode_;
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
};

}  // namespace kiln
