#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "executor/config.h"
#include "kv/kv_cache.h"

namespace kiln {

// One decoder layer's weights, HF layout throughout: every matrix is
// [out_features, in_features] so GemmBT (C = A @ B^T) can use it directly
// without a transpose pass at load time.
struct LayerWeights {
  std::vector<float> attn_norm;  // [dim]
  std::vector<float> wq;         // [n_heads*head_dim, dim]
  std::vector<float> wk;         // [n_kv_heads*head_dim, dim]
  std::vector<float> wv;         // [n_kv_heads*head_dim, dim]
  std::vector<float> wo;         // [dim, n_heads*head_dim]
  std::vector<float> ffn_norm;   // [dim]
  std::vector<float> w_gate;     // [ffn_hidden, dim]
  std::vector<float> w_up;       // [ffn_hidden, dim]
  std::vector<float> w_down;     // [dim, ffn_hidden]
};

// The full Llama-architecture model: embedding -> N decoder layers -> final
// norm -> LM head. This is Kiln's Phase 2 (forward pass) + Phase 3 (KV
// cache) + Phase 4 (padded batching) combined into one entry point, since
// they're one code path with optional arguments rather than three separate
// ones -- see docs/defense.md for why that's a deliberate simplicity
// choice, not scope-cutting.
class Model {
 public:
  // Deterministic small random weights -- used by every test in this repo,
  // since no real checkpoint has been run through tools/oracle.py in this
  // environment yet (that step needs a real HF install; see ADR-009).
  static Model LoadRandom(const ModelConfig& config, uint32_t seed);

  // Loads real weights from a safetensors file using the standard
  // Llama/HF key naming ("model.layers.{i}.self_attn.q_proj.weight", etc).
  // Written to spec; not yet exercised against a real checkpoint in this
  // session (see docs/defense.md phase-02 entry).
  static Model LoadFromSafetensors(const ModelConfig& config,
                                   const std::string& path);

  // tokens is [batch_size * seq_len], batch-major (item 0's seq_len tokens,
  // then item 1's, ...). valid_lengths (may be null => everything valid) is
  // per-item real length, for padded batches (Phase 4). start_pos is the
  // absolute position of each item's first token -- same value for every
  // item in this call (multi-item KV-cached decode at different start
  // positions per item is Phase 5+ scheduler territory, not this
  // function's job). cache is only meaningful when batch_size == 1
  // (Phase 3's contiguous single-sequence cache).
  // out_logits, sized [batch_size * seq_len, vocab_size], is filled for
  // every row including padding -- the caller decides which rows matter.
  void Forward(const int32_t* tokens, int64_t batch_size, int64_t seq_len,
               const int64_t* valid_lengths, int64_t start_pos,
               KVCache* cache, float* out_logits) const;

  const ModelConfig& config() const { return config_; }

 private:
  ModelConfig config_;
  std::vector<float> tok_embeddings;  // [vocab_size, dim]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;  // [dim]
  std::vector<float> lm_head;     // [vocab_size, dim]
};

}  // namespace kiln
