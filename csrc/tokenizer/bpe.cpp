#include "tokenizer/bpe.h"

#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace kiln {

namespace {

std::string Utf8Encode(uint32_t codepoint) {
  std::string out;
  if (codepoint < 0x80) {
    out += static_cast<char>(codepoint);
  } else if (codepoint < 0x800) {
    out += static_cast<char>(0xC0 | (codepoint >> 6));
    out += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (codepoint >> 12));
    out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (codepoint & 0x3F));
  }
  return out;
}

// Splits a UTF-8 string into a vector of single-codepoint substrings. Used
// to invert the byte<->unicode mapping during Decode.
std::vector<std::string> Utf8Codepoints(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = s[i];
    size_t len = (c < 0x80) ? 1 : (c >> 5 == 0x6) ? 2 : (c >> 4 == 0xE) ? 3 : 4;
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

// GPT-2/ByteLevel pre-tokenization uses Unicode categories. ICU gives the
// C++ tokenizer the same \p{L}, \p{N}, and whitespace semantics that the
// Hugging Face tokenizer uses, instead of treating every non-ASCII byte as a
// letter and splitting punctuation/number runs differently.
std::vector<std::string> PreTokenize(const std::string& text) {
  static const icu::RegexPattern* pattern = []() {
    UErrorCode status = U_ZERO_ERROR;
    const auto expression = icu::UnicodeString::fromUTF8(
        "(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\\p{L}+| ?\\p{N}+| "
        "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    const icu::RegexPattern* compiled =
        icu::RegexPattern::compile(expression, 0, status);
    if (U_FAILURE(status)) {
      throw std::runtime_error("bpe: failed to compile Unicode pre-tokenizer");
    }
    return compiled;
  }();

  UErrorCode status = U_ZERO_ERROR;
  const icu::UnicodeString input = icu::UnicodeString::fromUTF8(text);
  std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(input, status));
  if (U_FAILURE(status)) {
    throw std::runtime_error("bpe: failed to create Unicode pre-tokenizer");
  }
  std::vector<std::string> chunks;
  while (matcher->find(status)) {
    icu::UnicodeString match = matcher->group(status);
    std::string utf8;
    match.toUTF8String(utf8);
    chunks.push_back(std::move(utf8));
  }
  if (U_FAILURE(status)) {
    throw std::runtime_error("bpe: Unicode pre-tokenizer failed");
  }
  return chunks;
}

}  // namespace

BpeTokenizer BpeTokenizer::Load(const std::string& tokenizer_json_path) {
  std::ifstream file(tokenizer_json_path);
  if (!file) {
    throw std::runtime_error("bpe: cannot open " + tokenizer_json_path);
  }
  nlohmann::json j;
  file >> j;

  BpeTokenizer tok;

  // The byte<->unicode table (GPT-2 algorithm): printable Latin-1 ranges
  // map to themselves; every other byte value (0-255) gets assigned the
  // next unused codepoint above 255. This makes every byte value map to
  // *some* visible, single-codepoint symbol -- see docs/learning/phase-01.md
  // for why this is what makes byte-level BPE total over all byte strings.
  std::vector<int> printable;
  for (int b = '!'; b <= '~'; ++b) printable.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) printable.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) printable.push_back(b);

  std::vector<bool> is_printable(256, false);
  for (int b : printable) is_printable[b] = true;

  int next_extra = 256;
  for (int b = 0; b < 256; ++b) {
    uint32_t codepoint = is_printable[b] ? b : next_extra++;
    std::string ch = Utf8Encode(codepoint);
    tok.byte_to_unicode_[static_cast<uint8_t>(b)] = ch;
    tok.unicode_to_byte_[ch] = static_cast<uint8_t>(b);
  }

  const auto& model = j.contains("model") ? j["model"] : j;
  for (auto& [token, id] : model["vocab"].items()) {
    tok.vocab_[token] = id.get<int32_t>();
    tok.id_to_token_[id.get<int32_t>()] = token;
  }

  int32_t rank = 0;
  for (const auto& merge : model["merges"]) {
    std::string a, b;
    if (merge.is_string()) {
      std::string s = merge.get<std::string>();
      size_t space = s.find(' ');
      a = s.substr(0, space);
      b = s.substr(space + 1);
    } else {
      a = merge[0].get<std::string>();
      b = merge[1].get<std::string>();
    }
    tok.merge_rank_[a + '\0' + b] = rank++;
  }

  return tok;
}

std::vector<int32_t> BpeTokenizer::BpeMergeWord(
    const std::vector<std::string>& symbols) const {
  std::vector<std::string> current = symbols;

  while (current.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_index = 0;
    bool found = false;

    for (size_t i = 0; i + 1 < current.size(); ++i) {
      auto it = merge_rank_.find(current[i] + '\0' + current[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
        found = true;
      }
    }
    if (!found) break;

    current[best_index] += current[best_index + 1];
    current.erase(current.begin() + best_index + 1);
  }

  std::vector<int32_t> ids;
  ids.reserve(current.size());
  for (const auto& symbol : current) {
    auto it = vocab_.find(symbol);
    if (it == vocab_.end()) {
      throw std::runtime_error("bpe: symbol not in vocab after merging: " +
                               symbol);
    }
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int32_t> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<int32_t> ids;
  for (const std::string& chunk : PreTokenize(text)) {
    std::vector<std::string> symbols;
    symbols.reserve(chunk.size());
    for (unsigned char byte : chunk) {
      symbols.push_back(byte_to_unicode_.at(byte));
    }
    std::vector<int32_t> chunk_ids = BpeMergeWord(symbols);
    ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
  }
  return ids;
}

std::string BpeTokenizer::Decode(const std::vector<int32_t>& ids) const {
  std::string mapped;
  for (int32_t id : ids) {
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) {
      throw std::runtime_error("bpe: unknown token id " + std::to_string(id));
    }
    mapped += it->second;
  }

  std::string bytes;
  for (const std::string& codepoint : Utf8Codepoints(mapped)) {
    auto it = unicode_to_byte_.find(codepoint);
    if (it == unicode_to_byte_.end()) {
      throw std::runtime_error("bpe: decode symbol not in byte table");
    }
    bytes += static_cast<char>(it->second);
  }
  return bytes;
}

}  // namespace kiln
