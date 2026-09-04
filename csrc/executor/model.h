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
               const int64_t* valid_lengths, int64_t start_pos, KVCache* cache,
               float* out_logits,
               std::vector<std::vector<float>>* layer_outputs = nullptr) const;

  // Runs one cached decode token for every sequence in a batch. The matrix
  // work is shared across the batch, but each sequence keeps a separate KV
  // cache and absolute position, which is the minimum contract real
  // continuous batching needs. `tokens`, `start_positions`, and `caches`
  // each have batch_size entries; out_logits is [batch_size, vocab_size].
  void ForwardDecodeBatch(const int32_t* tokens, int64_t batch_size,
                          const int64_t* start_positions,
                          KVCache* const* caches, float* out_logits) const;

  // Prefills several sequences of *different* lengths in one call, with no
  // padding at all -- this is what Phase 23's note calls "ragged prefill":
  // `tokens` is every sequence's real tokens concatenated back-to-back
  // (no filler), `seq_lengths` gives each sequence's real length, and
  // `caches` gives each sequence's own (already-existing, possibly
  // non-empty for a continued prefill) KV cache. Every matmul in every
  // layer runs once over the whole concatenated batch -- exactly the
  // total real work, never padding's wasted rows -- and only attention
  // treats each sequence as its own separate lane, by calling the same
  // tested Attention() function once per sequence on its own slice, the
  // same pattern the padded Forward() path above already uses.
  // out_logits is [total_tokens, vocab_size], where total_tokens is the
  // sum of seq_lengths -- the caller picks out whichever row(s) it wants
  // (usually just the last row of each sequence).
  void ForwardPrefillBatch(const int32_t* tokens, int64_t num_sequences,
                           const int64_t* seq_lengths, KVCache* const* caches,
                           float* out_logits) const;

  // Tensor-parallel forward pass, simulated across `world_size` ranks in
  // this one process -- the Megatron-style scheme ADR-012/Phase 12 already
  // proved exact on synthetic matrices in
  // kiln_py/runtime/tensor_parallel_sim.py, run here for real against this
  // model's own weights. Attention is column-parallel (each rank owns a
  // contiguous slice of the heads and computes attention for just those heads
  // -- no communication needed, since one head's attention never depends on
  // another head's numbers) and the output/down projections are row-parallel
  // (each rank computes its own partial contribution to the full output; the
  // ranks' partial sums are added together, standing in for the one real
  // network all-reduce a multi-GPU run would need per block). Requires n_heads,
  // n_kv_heads, and ffn_hidden to all divide evenly by world_size.
  void ForwardTensorParallelSimulated(const int32_t* tokens, int64_t seq_len,
                                      int64_t world_size,
                                      float* out_logits) const;

  const ModelConfig& config() const { return config_; }

  // Folds a trained LoRA adapter into one weight matrix of one layer, in
  // place -- see docs/learning/phase-14.md. `which` selects the matrix:
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
