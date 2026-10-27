#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kiln {

// Byte-pair-encoding (BPE) tokenizer. It turns text into token ids and back.
//
// It loads a Hugging Face tokenizer.json, which holds a vocabulary and a
// ranked list of merges. Encoding starts with single bytes and repeatedly
// merges the best-ranked adjacent pair until no merge applies. Working on
// bytes means any text, in any language, can be encoded and decoded exactly.
class BpeTokenizer {
 public:
  static BpeTokenizer Load(const std::string& tokenizer_json_path);

  std::vector<int32_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int32_t>& ids) const;

 private:
  BpeTokenizer() = default;

  // Runs the merge loop on one chunk of text and returns its token ids.
  std::vector<int32_t> BpeMergeWord(
      const std::vector<std::string>& symbols) const;

  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<int32_t, std::string> id_to_token_;
  // Rank of each mergeable pair; a lower number is merged first.
  std::unordered_map<std::string, int32_t> merge_rank_;  // key: "a\x00b"

  // Each of the 256 byte values gets a visible character to stand for it
  // (GPT-2 style), so every byte string can be written down and recovered.
  std::unordered_map<uint8_t, std::string> byte_to_unicode_;
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
};

}  // namespace kiln
