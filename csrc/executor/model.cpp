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
  // Follows the standard Llama/Hugging Face weight naming (for example
  // "model.layers.3.self_attn.q_proj.weight"). Checked against real
  // SmolLM2 checkpoints by tools/hf_parity.py.
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

void Model::Forward(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
                    KVCache* cache, float* out_logits,
                    std::vector<std::vector<float>>* layer_outputs) const {
  int64_t d = config_.hidden_size;
  int64_t q_dim = config_.n_heads * config_.head_dim;
  int64_t kv_dim = config_.n_kv_heads * config_.head_dim;

  // Step 1: turn each word (an integer id) into its list of numbers (its
  // "embedding") by copying that row out of the embedding table.
  std::vector<float> x(seq_len * d);
  for (int64_t r = 0; r < seq_len; ++r) {
    std::memcpy(x.data() + r * d, tok_embeddings.data() + tokens[r] * d,
                d * sizeof(float));
  }

  // Where each word sits in the whole text.
  std::vector<int64_t> positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i) positions[i] = start_pos + i;

  // Scratch space reused by every layer.
  std::vector<float> normed(seq_len * d);
  std::vector<float> q(seq_len * q_dim);
  std::vector<float> k(seq_len * kv_dim);
  std::vector<float> v(seq_len * kv_dim);
  std::vector<float> attn_out(seq_len * q_dim);
  std::vector<float> proj(seq_len * d);
  std::vector<float> mlp_out(seq_len * d);

  for (int64_t layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx) {
    const LayerWeights& layer = layers[layer_idx];

    // Step 2, attention: each word looks back at the earlier words.
    RmsNorm(x.data(), layer.attn_norm.data(), normed.data(), seq_len, d,
            config_.rms_eps);
    GemmBT(normed.data(), layer.wq.data(), q.data(), seq_len, d, q_dim);
    GemmBT(normed.data(), layer.wk.data(), k.data(), seq_len, d, kv_dim);
    GemmBT(normed.data(), layer.wv.data(), v.data(), seq_len, d, kv_dim);

    // RoPE stamps each word's position into its query and key numbers, so
    // attention knows how far apart two words are.
    ApplyRope(q.data(), positions.data(), seq_len, config_.n_heads,
              config_.head_dim, config_.rope_theta);
    ApplyRope(k.data(), positions.data(), seq_len, config_.n_kv_heads,
              config_.head_dim, config_.rope_theta);

    // With a cache, save this call's keys and values and attend over
    // everything saved so far. Without one, attend over just these words.
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

    GemmBT(attn_out.data(), layer.wo.data(), proj.data(), seq_len, q_dim, d);
    for (int64_t i = 0; i < seq_len * d; ++i) x[i] += proj[i];  // residual add

    // Step 3, feed-forward network: process what attention gathered.
    RmsNorm(x.data(), layer.ffn_norm.data(), normed.data(), seq_len, d,
            config_.rms_eps);
    SwiGlu(normed.data(), layer.w_gate.data(), layer.w_up.data(),
           layer.w_down.data(), mlp_out.data(), seq_len, d, config_.ffn_hidden);
    for (int64_t i = 0; i < seq_len * d; ++i) x[i] += mlp_out[i];  // residual
    if (layer_outputs != nullptr) layer_outputs->push_back(x);
  }

  if (cache != nullptr) cache->Advance(seq_len);

  // Step 4: score every word in the vocabulary for each position.
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

  // Every weight matrix is stored [out_features, in_features], so merging
  // is the same operation for each one. Only the shape differs.
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
