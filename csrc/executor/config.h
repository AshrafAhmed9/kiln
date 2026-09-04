#pragma once
#include <cstdint>

namespace kiln {

// Llama-architecture config: RMSNorm, RoPE, grouped-query attention, SwiGLU.
// Real checkpoints set these from the model's config.json; tests use small
// hand-picked values so the math stays checkable by hand.
struct ModelConfig {
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;  // d
  int64_t n_layers = 0;
  int64_t n_heads = 0;     // query heads
  int64_t n_kv_heads = 0;  // key/value heads (GQA: n_kv_heads <= n_heads)
  int64_t head_dim = 0;    // hidden_size / n_heads
  int64_t ffn_hidden = 0;  // SwiGLU intermediate size
  int64_t max_seq_len = 0;
  float rms_eps = 1e-5f;
  float rope_theta = 10000.0f;
};

}  // namespace kiln
