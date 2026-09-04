#include "kv/kv_cache.h"

#include <cassert>
#include <cstring>

namespace kiln {

KVCache::KVCache(int64_t n_layers, int64_t max_seq_len, int64_t n_kv_heads,
                 int64_t head_dim)
    : n_kv_heads_(n_kv_heads), head_dim_(head_dim) {
  size_t per_layer = static_cast<size_t>(max_seq_len * n_kv_heads * head_dim);
  k_.assign(n_layers, std::vector<float>(per_layer));
  v_.assign(n_layers, std::vector<float>(per_layer));
}

void KVCache::Append(int64_t layer, const float* new_k, const float* new_v,
                     int64_t n_new) {
  int64_t row_size = n_kv_heads_ * head_dim_;
  size_t offset = static_cast<size_t>(length_ * row_size);
  size_t count = static_cast<size_t>(n_new * row_size);
  assert(offset + count <= k_[layer].size() &&
         "KVCache exhausted -- Forward() was called with more tokens than "
         "max_seq_len allows for");
  std::memcpy(k_[layer].data() + offset, new_k, count * sizeof(float));
  std::memcpy(v_[layer].data() + offset, new_v, count * sizeof(float));
}

}  // namespace kiln
