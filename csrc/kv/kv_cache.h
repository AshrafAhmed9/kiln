#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// The KV cache for one text. While generating, every layer needs the keys
// (K) and values (V) of all earlier words. Saving them here means each new
// word only has to compute its own, instead of redoing the whole text.
//
// Each layer gets one buffer, sized up front for the longest allowed text,
// so nothing is reallocated while generating. length() is how many words
// are stored so far; all layers share it, because each Forward call adds
// the same number of words to every layer.
//
// See PagedKVCache for a version that splits memory into blocks.
class KVCache {
 public:
  KVCache(int64_t n_layers, int64_t max_seq_len, int64_t n_kv_heads,
          int64_t head_dim);

  // Stores the keys and values of `n_new` new words for one layer. Call
  // once per layer per Forward call. It does not change length(); call
  // Advance() once after every layer has been appended.
  void Append(int64_t layer, const float* new_k, const float* new_v,
              int64_t n_new);
  void Advance(int64_t n_new) { length_ += n_new; }

  const float* K(int64_t layer) const { return k_[layer].data(); }
  const float* V(int64_t layer) const { return v_[layer].data(); }
  int64_t length() const { return length_; }

 private:
  int64_t n_kv_heads_;
  int64_t head_dim_;
  int64_t length_ = 0;
  // k_[layer] is [max_seq_len, n_kv_heads * head_dim]
  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
};

}  // namespace kiln
