#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "executor/config.h"
#include "kv/kv_cache.h"

namespace kiln {

class CudaModel;

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

// The full Llama-style model: word embeddings -> N decoder layers -> final
// norm -> output layer that scores every word in the vocabulary.
class Model {
 public:
  // Small random weights from a seed, so tests don't need a real checkpoint.
  // The same seed always gives the same weights.
  static Model LoadRandom(const ModelConfig& config, uint32_t seed);

  // Loads real trained weights from a .safetensors file, using the standard
  // Llama / Hugging Face tensor names. Checked against real SmolLM2
  // checkpoints by tools/hf_parity.py.
  static Model LoadFromSafetensors(const ModelConfig& config,
                                   const std::string& path);

  // Runs the model on `seq_len` words (token ids) and writes one row of
  // scores per word into out_logits, shape [seq_len, vocab_size]. The last
  // row says how likely each possible next word is.
  //
  // start_pos is where these words sit in the whole text: 0 for a fresh
  // prompt, or the number of words already cached when adding one more.
  // cache is optional. With a cache, the keys and values of these words are
  // saved for later calls; without one, everything is recomputed each time.
  // layer_outputs is optional and collects each layer's output for testing.
  void Forward(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
               KVCache* cache, float* out_logits,
               std::vector<std::vector<float>>* layer_outputs = nullptr) const;

  const ModelConfig& config() const { return config_; }

  // Folds a trained LoRA adapter into one weight matrix of one layer, in
  // place. `which` selects the matrix:
  // "wq", "wk", "wv", "wo", "w_gate", "w_up", or "w_down". Throws on an
  // unrecognized name rather than silently doing nothing.
  void MergeLoraIntoLayer(int64_t layer_idx, const std::string& which,
                          const float* lora_a, const float* lora_b,
                          int64_t rank, float scale);

 private:
  friend class CudaModel;

  ModelConfig config_;
  std::vector<float> tok_embeddings;  // [vocab_size, dim]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;  // [dim]
  std::vector<float> lm_head;     // [vocab_size, dim]
};

}  // namespace kiln
