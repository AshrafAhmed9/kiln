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
  load_tensor("lm_head.weight", model.lm_head);

  return model;
}

void Model::Forward(const int32_t* tokens, int64_t batch_size,
                     int64_t seq_len, const int64_t* valid_lengths,
                     int64_t start_pos, KVCache* cache,
                     float* out_logits) const {
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
    for (int64_t i = 0; i < n_rows * d; ++i) x[i] += mlp_out[i];  // residual add
  }

  if (cache != nullptr) cache->Advance(seq_len);

  RmsNorm(x.data(), final_norm.data(), normed.data(), n_rows, d,
          config_.rms_eps);
  GemmBT(normed.data(), lm_head.data(), out_logits, n_rows, d,
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
    MergeLoraAdapter(layer.w_gate.data(), lora_a, lora_b, config_.ffn_hidden,
                     d, rank, scale);
  } else if (which == "w_up") {
    MergeLoraAdapter(layer.w_up.data(), lora_a, lora_b, config_.ffn_hidden, d,
                     rank, scale);
  } else if (which == "w_down") {
    MergeLoraAdapter(layer.w_down.data(), lora_a, lora_b, d,
                     config_.ffn_hidden, rank, scale);
  } else {
    throw std::invalid_argument("Model::MergeLoraIntoLayer: unknown matrix "
                                "name '" + which + "'");
  }
}

}  // namespace kiln
