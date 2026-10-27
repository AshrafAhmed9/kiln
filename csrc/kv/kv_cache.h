#pragma once
#include <cstdint>
#include <vector>

namespace kiln {

// Contiguous, single-sequence KV cache (simple version; paged upgrades
// this to a block/paged allocator for multiple concurrent sequences --
// contiguous is the simplest starting point, and paging generalizes it).
//
// One buffer per layer, sized for the whole max sequence up front (arena
// discipline: no per-token reallocation). `length()` is
// how many positions are already filled; every layer shares the same
// length, since one Model::Forward call advances every layer by the same
// number of new tokens.
class KVCache {
 public:
  KVCache(int64_t n_layers, int64_t max_seq_len, int64_t n_kv_heads,
          int64_t head_dim);

  // Writes `n_new` new K/V rows for `layer` starting at the current
  // length -- call once per layer, per Forward() call, in layer order.
  // Does not advance length(); call Advance() once after all layers.
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
  std::vector<std::vector<float>>
      k_;  // k_[layer] is [max_seq_len, n_kv_heads*head_dim]
  std::vector<std::vector<float>> v_;
};

}  // namespace kiln
