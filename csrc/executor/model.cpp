#include "executor/model.h"

#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "executor/attention.h"
#include "executor/gemm.h"
#include "executor/lora.h"
#include "executor/rmsnorm.h"
#include "executor/rope.h"
#include "executor/swiglu.h"
#include "loader/dtype.h"
#include "loader/safetensors.h"

namespace kiln {

namespace {

// Fills a buffer with small random numbers. Real models load trained
// numbers from a checkpoint file; for testing our own code we don't need
// numbers that mean anything, we just need numbers that are always the
// same for a given seed, so a test can run twice and get the same answer.
void FillRandom(std::vector<float>& buf, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  for (float& v : buf) v = dist(rng);
}

}  // namespace

Model Model::LoadRandom(const ModelConfig& config, uint32_t seed) {
  Model model;
  model.config_ = config;
  std::mt19937 rng(seed);

  int64_t d = config.hidden_size;
  int64_t q_dim = config.n_heads * config.head_dim;
  int64_t kv_dim = config.n_kv_heads * config.head_dim;

  model.tok_embeddings.assign(config.vocab_size * d, 0.0f);
  FillRandom(model.tok_embeddings, rng);

  model.layers.resize(config.n_layers);
  for (auto& layer : model.layers) {
    layer.attn_norm.assign(d, 1.0f);  // start norm weights at 1 (no-op scale)
    layer.wq.assign(q_dim * d, 0.0f);
    layer.wk.assign(kv_dim * d, 0.0f);
    layer.wv.assign(kv_dim * d, 0.0f);
    layer.wo.assign(d * q_dim, 0.0f);
    layer.ffn_norm.assign(d, 1.0f);
    layer.w_gate.assign(config.ffn_hidden * d, 0.0f);
    layer.w_up.assign(config.ffn_hidden * d, 0.0f);
    layer.w_down.assign(d * config.ffn_hidden, 0.0f);

    FillRandom(layer.wq, rng);
    FillRandom(layer.wk, rng);
    FillRandom(layer.wv, rng);
    FillRandom(layer.wo, rng);
    FillRandom(layer.w_gate, rng);
    FillRandom(layer.w_up, rng);
    FillRandom(layer.w_down, rng);
  }

  model.final_norm.assign(d, 1.0f);
  model.lm_head.assign(config.vocab_size * d, 0.0f);
  FillRandom(model.lm_head, rng);

  return model;
}

Model Model::LoadFromSafetensors(const ModelConfig& config,
                                 const std::string& path) {
  // This follows the standard Llama/HuggingFace naming for weight files
  // (for example "model.layers.3.self_attn.q_proj.weight"). It has been
  // written carefully to match that naming, but it has not yet been run
  // against a real downloaded checkpoint in this environment -- doing
  // that needs a real HuggingFace install, which this offline session
  // doesn't have (see the project's zero-budget notes, ADR-009). The
  // honest status of this function is recorded in docs/defense.md.
  SafetensorsFile file = SafetensorsFile::Load(path);
  Model model;
  model.config_ = config;

  auto load_tensor = [&](const std::string& name, std::vector<float>& out) {
    const TensorView& view = file.Tensor(name);
    size_t count = 1;
    for (int64_t dim : view.shape) count *= static_cast<size_t>(dim);
    out.resize(count);
    ConvertToF32(view, count, out.data());
  };

  load_tensor("model.embed_tokens.weight", model.tok_embeddings);

  model.layers.resize(config.n_layers);
  for (int64_t i = 0; i < config.n_layers; ++i) {
    std::string prefix = "model.layers." + std::to_string(i) + ".";
    auto& layer = model.layers[i];
    load_tensor(prefix + "input_layernorm.weight", layer.attn_norm);
    load_tensor(prefix + "self_attn.q_proj.weight", layer.wq);
    load_tensor(prefix + "self_attn.k_proj.weight", layer.wk);
    load_tensor(prefix + "self_attn.v_proj.weight", layer.wv);
    load_tensor(prefix + "self_attn.o_proj.weight", layer.wo);
    load_tensor(prefix + "post_attention_layernorm.weight", layer.ffn_norm);
    load_tensor(prefix + "mlp.gate_proj.weight", layer.w_gate);
    load_tensor(prefix + "mlp.up_proj.weight", layer.w_up);
    load_tensor(prefix + "mlp.down_proj.weight", layer.w_down);
  }

  load_tensor("model.norm.weight", model.final_norm);
  if (file.HasTensor("lm_head.weight")) {
    load_tensor("lm_head.weight", model.lm_head);
  } else {
    // Llama-family checkpoints often tie the output head to the embedding
    // table, storing one copy under model.embed_tokens.weight. Reusing that
    // already-converted vector preserves the checkpoint's intended weights
    // without requiring a duplicate tensor to exist on disk.
    model.lm_head = model.tok_embeddings;
  }

  return model;
}

void Model::Forward(const int32_t* tokens, int64_t batch_size, int64_t seq_len,
                    const int64_t* valid_lengths, int64_t start_pos,
                    KVCache* cache, float* out_logits,
                    std::vector<std::vector<float>>* layer_outputs) const {
  // A cache only makes sense for one sentence at a time in this project's
  // current design (Phase 3's cache is built for a single sequence; batching
  // several sentences together with no cache at all is Phase 4's separate
  // case). Passing both together isn't a case anyone should hit on purpose,
  // so it's rejected loudly here rather than silently doing the wrong thing
  // (quietly ignoring the cache would be far worse than an error, since the
  // caller would get an answer back and have no way to know it was computed
  // without the caching they asked for).
  if (cache != nullptr && batch_size > 1) {
    throw std::invalid_argument(
        "Model::Forward: a KV cache can only be used with batch_size == 1 "
        "in this project's current design -- see docs/defense.md phase 4/8 "
        "entries for why batching and caching don't yet combine.");
  }

  int64_t d = config_.hidden_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t kv_dim = config_.n_kv_heads * config_.head_dim;
  int64_t n_rows = batch_size * seq_len;

  // Step 1: turn each input word (an integer ID) into its list of numbers
  // (its "embedding") by copying that row out of the embedding table.
  std::vector<float> x(n_rows * d);
  for (int64_t r = 0; r < n_rows; ++r) {
    int32_t token = tokens[r];
    std::memcpy(x.data() + r * d, tok_embeddings.data() + token * d,
                d * sizeof(float));
  }

  // Every token in this call gets an absolute position in its own sentence
  // -- item 0's tokens start at start_pos, item 1's tokens also start at
  // start_pos (each batch item is its own independent sentence, all lined
  // up to begin at the same offset for this simple padded-batch case).
  std::vector<int64_t> positions(n_rows);
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t i = 0; i < seq_len; ++i) {
      positions[b * seq_len + i] = start_pos + i;
    }
  }

  std::vector<float> normed(n_rows * d);
  std::vector<float> q(n_rows * q_dim);
  std::vector<float> k(n_rows * kv_dim);
  std::vector<float> v(n_rows * kv_dim);
  std::vector<float> attn_out(n_rows * q_dim);
  std::vector<float> proj(n_rows * d);
  std::vector<float> mlp_out(n_rows * d);

  for (int64_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    const LayerWeights& layer = layers[layer_idx];

    // "Read the sentence so far, decide what matters" step: attention.
    RmsNorm(x.data(), layer.attn_norm.data(), normed.data(), n_rows, d,
            config_.rms_eps);
    GemmBT(normed.data(), layer.wq.data(), q.data(), n_rows, d, q_dim);
    GemmBT(normed.data(), layer.wk.data(), k.data(), n_rows, d, kv_dim);
    GemmBT(normed.data(), layer.wv.data(), v.data(), n_rows, d, kv_dim);

    // Rotary position embedding stamps each token's position into its
    // query/key numbers, so attention naturally knows how far apart two
    // tokens are without needing a separate "position" input.
    ApplyRope(q.data(), positions.data(), n_rows, config_.n_heads,
              config_.head_dim, config_.rope_theta);
    ApplyRope(k.data(), positions.data(), n_rows, config_.n_kv_heads,
              config_.head_dim, config_.rope_theta);

    if (batch_size == 1) {
      const float* k_for_attn = k.data();
      const float* v_for_attn = v.data();
      int64_t kv_len = seq_len;

      if (cache != nullptr) {
        cache->Append(layer_idx, k.data(), v.data(), seq_len);
        k_for_attn = cache->K(layer_idx);
        v_for_attn = cache->V(layer_idx);
        kv_len = cache->length() + seq_len;
      }

      Attention(q.data(), k_for_attn, v_for_attn, attn_out.data(), seq_len,
                kv_len, config_.n_heads, config_.n_kv_heads, config_.head_dim,
                start_pos);
    } else {
      // Padded batch (Phase 4): every sentence in the batch keeps to its
      // own lane and never attends into another sentence's tokens or
      // padding -- we just run the same attention function once per
      // sentence, pointed at that sentence's own slice of the batch.
      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t offset = b * seq_len;
        int64_t valid = valid_lengths ? valid_lengths[b] : seq_len;
        Attention(q.data() + offset * q_dim, k.data() + offset * kv_dim,
                  v.data() + offset * kv_dim, attn_out.data() + offset * q_dim,
                  seq_len, seq_len, config_.n_heads, config_.n_kv_heads,
                  config_.head_dim, start_pos, valid);
      }
    }

    GemmBT(attn_out.data(), layer.wo.data(), proj.data(), n_rows, q_dim, d);
    for (int64_t i = 0; i < n_rows * d; ++i) x[i] += proj[i];  // residual add

    // "Think about what was just read" step: the feed-forward network.
    RmsNorm(x.data(), layer.ffn_norm.data(), normed.data(), n_rows, d,
            config_.rms_eps);
    SwiGlu(normed.data(), layer.w_gate.data(), layer.w_up.data(),
           layer.w_down.data(), mlp_out.data(), n_rows, d, config_.ffn_hidden);
    for (int64_t i = 0; i < n_rows * d; ++i)
      x[i] += mlp_out[i];  // residual add
    if (layer_outputs != nullptr) layer_outputs->push_back(x);
  }

  if (cache != nullptr) cache->Advance(seq_len);

  RmsNorm(x.data(), final_norm.data(), normed.data(), n_rows, d,
          config_.rms_eps);
  GemmBT(normed.data(), lm_head.data(), out_logits, n_rows, d,
         config_.vocab_size);
}

void Model::ForwardDecodeBatch(const int32_t* tokens, int64_t batch_size,
                               const int64_t* start_positions,
                               KVCache* const* caches,
                               float* out_logits) const {
  if (batch_size <= 0) return;
  for (int64_t b = 0; b < batch_size; ++b) {
    if (caches[b] == nullptr) {
      throw std::invalid_argument(
          "Model::ForwardDecodeBatch requires one KV cache per sequence");
    }
    if (caches[b]->length() != start_positions[b]) {
      throw std::invalid_argument(
          "Model::ForwardDecodeBatch cache length must match start position");
    }
  }

  int64_t d = config_.hidden_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t kv_dim = config_.n_kv_heads * config_.head_dim;

  std::vector<float> x(batch_size * d);
  for (int64_t b = 0; b < batch_size; ++b) {
    std::memcpy(x.data() + b * d, tok_embeddings.data() + tokens[b] * d,
                d * sizeof(float));
  }

  std::vector<float> normed(batch_size * d);
  std::vector<float> q(batch_size * q_dim);
  std::vector<float> k(batch_size * kv_dim);
  std::vector<float> v(batch_size * kv_dim);
  std::vector<float> attn_out(batch_size * q_dim);
  std::vector<float> proj(batch_size * d);
  std::vector<float> mlp_out(batch_size * d);

  for (int64_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    const LayerWeights& layer = layers[layer_idx];
    RmsNorm(x.data(), layer.attn_norm.data(), normed.data(), batch_size, d,
            config_.rms_eps);
    GemmBT(normed.data(), layer.wq.data(), q.data(), batch_size, d, q_dim);
    GemmBT(normed.data(), layer.wk.data(), k.data(), batch_size, d, kv_dim);
    GemmBT(normed.data(), layer.wv.data(), v.data(), batch_size, d, kv_dim);
    ApplyRope(q.data(), start_positions, batch_size, config_.n_heads,
              config_.head_dim, config_.rope_theta);
    ApplyRope(k.data(), start_positions, batch_size, config_.n_kv_heads,
              config_.head_dim, config_.rope_theta);

    for (int64_t b = 0; b < batch_size; ++b) {
      caches[b]->Append(layer_idx, k.data() + b * kv_dim, v.data() + b * kv_dim,
                        /*n_new=*/1);
      int64_t kv_len = caches[b]->length() + 1;
      Attention(q.data() + b * q_dim, caches[b]->K(layer_idx),
                caches[b]->V(layer_idx), attn_out.data() + b * q_dim,
                /*seq_len=*/1, kv_len, config_.n_heads, config_.n_kv_heads,
                config_.head_dim, start_positions[b]);
    }

    GemmBT(attn_out.data(), layer.wo.data(), proj.data(), batch_size, q_dim, d);
    for (int64_t i = 0; i < batch_size * d; ++i) x[i] += proj[i];
    RmsNorm(x.data(), layer.ffn_norm.data(), normed.data(), batch_size, d,
            config_.rms_eps);
    SwiGlu(normed.data(), layer.w_gate.data(), layer.w_up.data(),
           layer.w_down.data(), mlp_out.data(), batch_size, d,
           config_.ffn_hidden);
    for (int64_t i = 0; i < batch_size * d; ++i) x[i] += mlp_out[i];
  }

  for (int64_t b = 0; b < batch_size; ++b) caches[b]->Advance(/*n_new=*/1);
  RmsNorm(x.data(), final_norm.data(), normed.data(), batch_size, d,
          config_.rms_eps);
  GemmBT(normed.data(), lm_head.data(), out_logits, batch_size, d,
         config_.vocab_size);
}

void Model::ForwardPrefillBatch(const int32_t* tokens, int64_t num_sequences,
                                const int64_t* seq_lengths,
                                KVCache* const* caches,
                                float* out_logits) const {
  if (num_sequences <= 0) return;
  for (int64_t b = 0; b < num_sequences; ++b) {
    if (caches[b] == nullptr) {
      throw std::invalid_argument(
          "Model::ForwardPrefillBatch requires one KV cache per sequence");
    }
  }

  // Where sequence b's tokens start within the concatenated `tokens`
  // array -- the ragged equivalent of "b * seq_len" in the padded path.
  std::vector<int64_t> offsets(num_sequences);
  int64_t total_tokens = 0;
  for (int64_t b = 0; b < num_sequences; ++b) {
    offsets[b] = total_tokens;
    total_tokens += seq_lengths[b];
  }

  int64_t d = config_.hidden_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t kv_dim = config_.n_kv_heads * config_.head_dim;
  int64_t n_rows = total_tokens;

  std::vector<float> x(n_rows * d);
  for (int64_t r = 0; r < n_rows; ++r) {
    int32_t token = tokens[r];
    std::memcpy(x.data() + r * d, tok_embeddings.data() + token * d,
                d * sizeof(float));
  }

  // Each sequence's tokens are positioned relative to whatever's already
  // in its own cache (0 for a fresh prefill, or partway through for a
  // prefill that's continuing an existing sequence) -- not relative to
  // this batch's own row index, which is what makes this genuinely
  // "ragged" rather than just "unpadded but otherwise like Forward()."
  std::vector<int64_t> positions(n_rows);
  for (int64_t b = 0; b < num_sequences; ++b) {
    int64_t start_pos = caches[b]->length();
    for (int64_t i = 0; i < seq_lengths[b]; ++i) {
      positions[offsets[b] + i] = start_pos + i;
    }
  }

  std::vector<float> normed(n_rows * d);
  std::vector<float> q(n_rows * q_dim);
  std::vector<float> k(n_rows * kv_dim);
  std::vector<float> v(n_rows * kv_dim);
  std::vector<float> attn_out(n_rows * q_dim);
  std::vector<float> proj(n_rows * d);
  std::vector<float> mlp_out(n_rows * d);

  for (int64_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    const LayerWeights& layer = layers[layer_idx];

    // Every matmul below runs once over all n_rows -- the entire point of
    // ragged batching: total_tokens is exactly the real work, with none of
    // a padded batch's wasted rows.
    RmsNorm(x.data(), layer.attn_norm.data(), normed.data(), n_rows, d,
            config_.rms_eps);
    GemmBT(normed.data(), layer.wq.data(), q.data(), n_rows, d, q_dim);
    GemmBT(normed.data(), layer.wk.data(), k.data(), n_rows, d, kv_dim);
    GemmBT(normed.data(), layer.wv.data(), v.data(), n_rows, d, kv_dim);
    ApplyRope(q.data(), positions.data(), n_rows, config_.n_heads,
              config_.head_dim, config_.rope_theta);
    ApplyRope(k.data(), positions.data(), n_rows, config_.n_kv_heads,
              config_.head_dim, config_.rope_theta);

    // Attention is the one step that can't be one big matmul across
    // sequences -- a token must never attend into a different sequence's
    // tokens. So this loops per sequence, on its own slice of q/k/v,
    // calling the exact same Attention() function the single-sequence and
    // padded-batch paths above already use and already have parity tests
    // for -- no new attention code to get wrong here.
    for (int64_t b = 0; b < num_sequences; ++b) {
      int64_t offset = offsets[b];
      int64_t len = seq_lengths[b];
      caches[b]->Append(layer_idx, k.data() + offset * kv_dim,
                        v.data() + offset * kv_dim, len);
      int64_t kv_len = caches[b]->length() + len;
      Attention(q.data() + offset * q_dim, caches[b]->K(layer_idx),
                caches[b]->V(layer_idx), attn_out.data() + offset * q_dim, len,
                kv_len, config_.n_heads, config_.n_kv_heads, config_.head_dim,
                caches[b]->length());
    }

    GemmBT(attn_out.data(), layer.wo.data(), proj.data(), n_rows, q_dim, d);
    for (int64_t i = 0; i < n_rows * d; ++i) x[i] += proj[i];
    RmsNorm(x.data(), layer.ffn_norm.data(), normed.data(), n_rows, d,
            config_.rms_eps);
    SwiGlu(normed.data(), layer.w_gate.data(), layer.w_up.data(),
           layer.w_down.data(), mlp_out.data(), n_rows, d, config_.ffn_hidden);
    for (int64_t i = 0; i < n_rows * d; ++i) x[i] += mlp_out[i];
  }

  for (int64_t b = 0; b < num_sequences; ++b) {
    caches[b]->Advance(seq_lengths[b]);
  }

  RmsNorm(x.data(), final_norm.data(), normed.data(), n_rows, d,
          config_.rms_eps);
  GemmBT(normed.data(), lm_head.data(), out_logits, n_rows, d,
         config_.vocab_size);
}

namespace {

// wo and w_down are stored [out_features, in_features] row-major, so a
// *column* slice (the in_features range one rank owns) is not a
// contiguous block the way a row slice is -- this gathers that column
// slice into its own small contiguous buffer once per rank, per layer.
// This is data movement, not new numerical code: every number it copies
// is untouched, and GemmBT is then called exactly the way it already is
// everywhere else in this file.
void GatherColumnSlice(const float* full, int64_t out_features,
                       int64_t full_in_features, int64_t slice_start,
                       int64_t slice_width, float* out_slice) {
  for (int64_t row = 0; row < out_features; ++row) {
    std::memcpy(out_slice + row * slice_width,
                full + row * full_in_features + slice_start,
                slice_width * sizeof(float));
  }
}

}  // namespace

void Model::ForwardTensorParallelSimulated(const int32_t* tokens,
                                           int64_t seq_len, int64_t world_size,
                                           float* out_logits) const {
  if (config_.n_heads % world_size != 0 ||
      config_.n_kv_heads % world_size != 0 ||
      config_.ffn_hidden % world_size != 0) {
    throw std::invalid_argument(
        "ForwardTensorParallelSimulated: n_heads, n_kv_heads, and ffn_hidden "
        "must all divide evenly by world_size");
  }

  int64_t d = config_.hidden_size;
  int64_t heads_per_rank = config_.n_heads / world_size;
  int64_t kv_heads_per_rank = config_.n_kv_heads / world_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t q_dim_per_rank = heads_per_rank * config_.head_dim;
  int64_t kv_dim_per_rank = kv_heads_per_rank * config_.head_dim;
  int64_t ffn_per_rank = config_.ffn_hidden / world_size;

  std::vector<float> x(seq_len * d);
  for (int64_t i = 0; i < seq_len; ++i) {
    std::memcpy(x.data() + i * d, tok_embeddings.data() + tokens[i] * d,
                d * sizeof(float));
  }
  std::vector<int64_t> positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i) positions[i] = i;

  std::vector<float> normed(seq_len * d);
  std::vector<float> attn_combined(seq_len * d);
  std::vector<float> mlp_combined(seq_len * d);
  std::vector<float> wo_shard(d * q_dim_per_rank);
  std::vector<float> w_down_shard(d * ffn_per_rank);
  std::vector<float> q_shard(seq_len * q_dim_per_rank);
  std::vector<float> k_shard(seq_len * kv_dim_per_rank);
  std::vector<float> v_shard(seq_len * kv_dim_per_rank);
  std::vector<float> attn_out_shard(seq_len * q_dim_per_rank);
  std::vector<float> partial(seq_len * d);

  for (int64_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    const LayerWeights& layer = layers[layer_idx];

    // The two norms below run once, in full, on every rank in a real
    // multi-GPU deployment (they're cheap and every rank needs the same
    // answer) -- so there's exactly one call here too, not one per rank.
    RmsNorm(x.data(), layer.attn_norm.data(), normed.data(), seq_len, d,
            config_.rms_eps);
    std::fill(attn_combined.begin(), attn_combined.end(), 0.0f);

    for (int64_t rank = 0; rank < world_size; ++rank) {
      const float* wq_shard = layer.wq.data() + rank * q_dim_per_rank * d;
      const float* wk_shard = layer.wk.data() + rank * kv_dim_per_rank * d;
      const float* wv_shard = layer.wv.data() + rank * kv_dim_per_rank * d;

      GemmBT(normed.data(), wq_shard, q_shard.data(), seq_len, d,
             q_dim_per_rank);
      GemmBT(normed.data(), wk_shard, k_shard.data(), seq_len, d,
             kv_dim_per_rank);
      GemmBT(normed.data(), wv_shard, v_shard.data(), seq_len, d,
             kv_dim_per_rank);

      // RoPE's rotation angle depends only on a token's position and which
      // pair-of-numbers within a head it is -- never on which head, or how
      // many heads there are in total -- so applying it to this rank's
      // subset of heads is identical to applying it to all heads and
      // keeping only this subset's numbers afterward.
      ApplyRope(q_shard.data(), positions.data(), seq_len, heads_per_rank,
                config_.head_dim, config_.rope_theta);
      ApplyRope(k_shard.data(), positions.data(), seq_len, kv_heads_per_rank,
                config_.head_dim, config_.rope_theta);

      // Column-parallel attention: this rank's heads are independent of
      // every other rank's heads, so the exact same Attention() function
      // used everywhere else in this file runs unmodified on just this
      // rank's slice -- no communication needed for attention itself.
      Attention(q_shard.data(), k_shard.data(), v_shard.data(),
                attn_out_shard.data(), seq_len, seq_len, heads_per_rank,
                kv_heads_per_rank, config_.head_dim, /*query_start_pos=*/0);

      GatherColumnSlice(layer.wo.data(), d, q_dim, rank * q_dim_per_rank,
                        q_dim_per_rank, wo_shard.data());
      GemmBT(attn_out_shard.data(), wo_shard.data(), partial.data(), seq_len,
             q_dim_per_rank, d);
      // The one combine step per block (constitution: the real point of
      // pairing column-parallel with row-parallel) -- summing every rank's
      // partial output stands in for the single real all-reduce a genuine
      // multi-GPU run would need here, the same simulated substitution
      // tensor_parallel_sim.py already uses and names honestly.
      for (int64_t i = 0; i < seq_len * d; ++i) attn_combined[i] += partial[i];
    }
    for (int64_t i = 0; i < seq_len * d; ++i) x[i] += attn_combined[i];

    RmsNorm(x.data(), layer.ffn_norm.data(), normed.data(), seq_len, d,
            config_.rms_eps);
    std::fill(mlp_combined.begin(), mlp_combined.end(), 0.0f);

    for (int64_t rank = 0; rank < world_size; ++rank) {
      const float* w_gate_shard = layer.w_gate.data() + rank * ffn_per_rank * d;
      const float* w_up_shard = layer.w_up.data() + rank * ffn_per_rank * d;
      GatherColumnSlice(layer.w_down.data(), d, config_.ffn_hidden,
                        rank * ffn_per_rank, ffn_per_rank, w_down_shard.data());

      // Each rank's SwiGlu call already produces a complete partial
      // contribution to the full `dim`-sized output -- restricting
      // ffn_hidden to this rank's slice just restricts which hidden units
      // that sum runs over, so summing every rank's result gives exactly
      // the same answer the unsharded call would.
      SwiGlu(normed.data(), w_gate_shard, w_up_shard, w_down_shard.data(),
             partial.data(), seq_len, d, ffn_per_rank);
      for (int64_t i = 0; i < seq_len * d; ++i) mlp_combined[i] += partial[i];
    }
    for (int64_t i = 0; i < seq_len * d; ++i) x[i] += mlp_combined[i];
  }

  // The embedding table and LM head are kept full/replicated here rather
  // than sharded -- the plan's stated Phase 12 scope is attention and MLP
  // sharding specifically (see docs/defense.md); vocab-parallel embeddings
  // are a real, separate technique this doesn't claim to cover.
  RmsNorm(x.data(), final_norm.data(), normed.data(), seq_len, d,
          config_.rms_eps);
  GemmBT(normed.data(), lm_head.data(), out_logits, seq_len, d,
         config_.vocab_size);
}

void Model::MergeLoraIntoLayer(int64_t layer_idx, const std::string& which,
                               const float* lora_a, const float* lora_b,
                               int64_t rank, float scale) {
  LayerWeights& layer = layers[layer_idx];
  int64_t d = config_.hidden_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t kv_dim = config_.n_kv_heads * config_.head_dim;

  // Every weight matrix in this project is stored [out_features,
  // in_features] (see LayerWeights' comment in model.h), so merging a
  // LoRA adapter into any of them is the same operation with different
  // dimensions -- this table is just "which matrix, and what shape is it."
  if (which == "wq") {
    MergeLoraAdapter(layer.wq.data(), lora_a, lora_b, q_dim, d, rank, scale);
  } else if (which == "wk") {
    MergeLoraAdapter(layer.wk.data(), lora_a, lora_b, kv_dim, d, rank, scale);
  } else if (which == "wv") {
    MergeLoraAdapter(layer.wv.data(), lora_a, lora_b, kv_dim, d, rank, scale);
  } else if (which == "wo") {
    MergeLoraAdapter(layer.wo.data(), lora_a, lora_b, d, q_dim, rank, scale);
  } else if (which == "w_gate") {
    MergeLoraAdapter(layer.w_gate.data(), lora_a, lora_b, config_.ffn_hidden, d,
                     rank, scale);
  } else if (which == "w_up") {
    MergeLoraAdapter(layer.w_up.data(), lora_a, lora_b, config_.ffn_hidden, d,
                     rank, scale);
  } else if (which == "w_down") {
    MergeLoraAdapter(layer.w_down.data(), lora_a, lora_b, d, config_.ffn_hidden,
                     rank, scale);
  } else {
    throw std::invalid_argument(
        "Model::MergeLoraIntoLayer: unknown matrix "
        "name '" +
        which + "'");
  }
}

}  // namespace kiln
