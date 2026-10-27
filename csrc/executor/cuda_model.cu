// The model running on the GPU. At construction, all weights are copied to
// GPU memory once. Each Forward call then copies in the token ids, runs
// every layer on the GPU, and copies the scores back. This follows the same
// steps as Model::Forward in model.cpp.
#include "executor/cuda_model.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels/cuda/kernels.h"

namespace kiln {
namespace {

// Turn CUDA / cuBLAS error codes into C++ exceptions with a readable message.
void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void CheckCuBlas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

// An array in GPU memory. It frees itself when it goes out of scope, so
// nothing leaks, and it can be moved but not copied.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { Allocate(count); }
  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void Allocate(size_t count) {
    if (count == count_) return;
    CheckCuda(cudaFree(data_), "cudaFree");
    data_ = nullptr;
    count_ = count;
    if (count_ != 0) {
      CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                "cudaMalloc");
    }
  }

  T* data() const { return data_; }
  size_t size() const { return count_; }

  void CopyFromHost(const T* host, size_t count) {
    if (count > count_) throw std::invalid_argument("device copy exceeds buffer capacity");
    CheckCuda(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice),
              "copy to device");
  }
  void CopyToHost(T* host, size_t count) const {
    if (count > count_) throw std::invalid_argument("device copy exceeds buffer capacity");
    CheckCuda(cudaMemcpy(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
              "copy from device");
  }
  void CopyFromDevice(const T* source, size_t count, size_t destination_offset) {
    if (destination_offset > count_ || count > count_ - destination_offset) {
      throw std::invalid_argument("device copy exceeds buffer capacity");
    }
    CheckCuda(cudaMemcpy(data_ + destination_offset, source, count * sizeof(T),
                         cudaMemcpyDeviceToDevice), "copy within device");
  }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

// Looks up each word's embedding (its list of numbers) from the table.
// One GPU block handles one word.
__global__ void GatherEmbeddingsKernel(const int32_t* tokens,
                                       const float* embeddings, float* out,
                                       int64_t seq_len, int64_t dim) {
  int64_t row = blockIdx.x;
  int64_t column = threadIdx.x;
  if (row < seq_len && column < dim) {
    out[row * dim + column] = embeddings[tokens[row] * dim + column];
  }
}

void GatherEmbeddingsCuda(const int32_t* tokens, const float* embeddings,
                          float* out, int64_t seq_len, int64_t dim) {
  GatherEmbeddingsKernel<<<seq_len, static_cast<unsigned int>(dim)>>>(
      tokens, embeddings, out, seq_len, dim);
}

// Matrix multiply through cuBLAS: C[m,n] = A[m,k] * B^T, with B stored [n,k].
void GemmBTWithHandle(cublasHandle_t handle, const float* a,
                      const float* b_transposed, float* c, int64_t m,
                      int64_t k, int64_t n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // cuBLAS is column-major. C_row=A*B^T becomes C^T=B*A^T when viewed by
  // cuBLAS, which swaps operands and reverses the output dimensions.
  CheckCuBlas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           static_cast<int>(n), static_cast<int>(m),
                           static_cast<int>(k), &alpha, b_transposed,
                           static_cast<int>(k), a, static_cast<int>(k), &beta,
                           c, static_cast<int>(n)),
              "cublasSgemm");
}

}  // namespace

// Everything that lives in GPU memory.
struct CudaModel::Impl {
  // One decoder layer's weights, copied to the GPU.
  struct Layer {
    DeviceBuffer<float> attn_norm;
    DeviceBuffer<float> wq;
    DeviceBuffer<float> wk;
    DeviceBuffer<float> wv;
    DeviceBuffer<float> wo;
    DeviceBuffer<float> ffn_norm;
    DeviceBuffer<float> w_gate;
    DeviceBuffer<float> w_up;
    DeviceBuffer<float> w_down;

    explicit Layer(const LayerWeights& host)
        : attn_norm(host.attn_norm.size()), wq(host.wq.size()), wk(host.wk.size()),
          wv(host.wv.size()), wo(host.wo.size()), ffn_norm(host.ffn_norm.size()),
          w_gate(host.w_gate.size()), w_up(host.w_up.size()), w_down(host.w_down.size()) {
      attn_norm.CopyFromHost(host.attn_norm.data(), host.attn_norm.size());
      wq.CopyFromHost(host.wq.data(), host.wq.size());
      wk.CopyFromHost(host.wk.data(), host.wk.size());
      wv.CopyFromHost(host.wv.data(), host.wv.size());
      wo.CopyFromHost(host.wo.data(), host.wo.size());
      ffn_norm.CopyFromHost(host.ffn_norm.data(), host.ffn_norm.size());
      w_gate.CopyFromHost(host.w_gate.data(), host.w_gate.size());
      w_up.CopyFromHost(host.w_up.data(), host.w_up.size());
      w_down.CopyFromHost(host.w_down.data(), host.w_down.size());
    }
  };

  // The saved keys and values for one layer (the GPU's KV cache).
  struct CacheLayer {
    explicit CacheLayer(size_t count) : k(count), v(count) {}
    DeviceBuffer<float> k;
    DeviceBuffer<float> v;
  };

  explicit Impl(const Model& model)
      : config(model.config_), embeddings(model.tok_embeddings.size()),
        final_norm(model.final_norm.size()), lm_head(model.lm_head.size()),
        tokens(config.max_seq_len), positions(config.max_seq_len),
        x(config.max_seq_len * config.hidden_size),
        normed(config.max_seq_len * config.hidden_size),
        q(config.max_seq_len * config.n_heads * config.head_dim),
        k(config.max_seq_len * config.n_kv_heads * config.head_dim),
        v(config.max_seq_len * config.n_kv_heads * config.head_dim),
        attn_out(config.max_seq_len * config.n_heads * config.head_dim),
        projection(config.max_seq_len * config.hidden_size),
        gate(config.max_seq_len * config.ffn_hidden),
        up(config.max_seq_len * config.ffn_hidden),
        hidden(config.max_seq_len * config.ffn_hidden),
        mlp_out(config.max_seq_len * config.hidden_size),
        logits(config.max_seq_len * config.vocab_size) {
    if (config.max_seq_len <= 0 || config.hidden_size <= 0 ||
        config.head_dim <= 0 || config.head_dim > 1024) {
      throw std::invalid_argument("CudaModel requires valid dimensions and head_dim <= 1024");
    }
    embeddings.CopyFromHost(model.tok_embeddings.data(), model.tok_embeddings.size());
    final_norm.CopyFromHost(model.final_norm.data(), model.final_norm.size());
    lm_head.CopyFromHost(model.lm_head.data(), model.lm_head.size());
    layers.reserve(model.layers.size());
    for (const LayerWeights& layer : model.layers) layers.emplace_back(layer);
    cache_layers.reserve(model.layers.size());
    const size_t cache_size = static_cast<size_t>(config.max_seq_len) *
                              config.n_kv_heads * config.head_dim;
    for (int64_t i = 0; i < config.n_layers; ++i) {
      cache_layers.emplace_back(cache_size);
    }
    CheckCuBlas(cublasCreate(&handle), "cublasCreate");
  }

  ~Impl() { cublasDestroy(handle); }

  // The weights, the KV cache, and scratch buffers sized for the longest
  // allowed text, so no memory is allocated while generating.
  ModelConfig config;
  cublasHandle_t handle = nullptr;
  DeviceBuffer<float> embeddings;
  DeviceBuffer<float> final_norm;
  DeviceBuffer<float> lm_head;
  std::vector<Layer> layers;
  std::vector<CacheLayer> cache_layers;
  int64_t cache_length = 0;
  DeviceBuffer<int32_t> tokens;
  DeviceBuffer<int64_t> positions;
  DeviceBuffer<float> x;
  DeviceBuffer<float> normed;
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> attn_out;
  DeviceBuffer<float> projection;
  DeviceBuffer<float> gate;
  DeviceBuffer<float> up;
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> mlp_out;
  DeviceBuffer<float> logits;
};

CudaModel::CudaModel(const Model& model) : impl_(std::make_unique<Impl>(model)) {}
CudaModel::~CudaModel() = default;
CudaModel::CudaModel(CudaModel&&) noexcept = default;
CudaModel& CudaModel::operator=(CudaModel&&) noexcept = default;

// Without the cache: every call works on the whole text it is given.
void CudaModel::Forward(const int32_t* host_tokens, int64_t seq_len,
                        int64_t start_pos, float* out_logits) const {
  ForwardImpl(host_tokens, seq_len, start_pos, out_logits, /*use_cache=*/false);
}

// With the cache: keys and values are saved on the GPU, so later calls only
// need the newest word.
void CudaModel::ForwardCached(const int32_t* host_tokens, int64_t seq_len,
                              int64_t start_pos, float* out_logits) const {
  if (start_pos != impl_->cache_length) {
    throw std::invalid_argument(
        "CudaModel::ForwardCached start position must equal cache length");
  }
  ForwardImpl(host_tokens, seq_len, start_pos, out_logits, /*use_cache=*/true);
}

void CudaModel::ResetCache() const { impl_->cache_length = 0; }

const ModelConfig& CudaModel::config() const { return impl_->config; }

// The shared forward pass. The steps match Model::Forward on the CPU.
void CudaModel::ForwardImpl(const int32_t* host_tokens, int64_t seq_len,
                            int64_t start_pos, float* out_logits,
                            bool use_cache) const {
  if (seq_len <= 0 || seq_len > impl_->config.max_seq_len) {
    throw std::invalid_argument("CudaModel::Forward sequence length is out of range");
  }
  if (use_cache && seq_len > impl_->config.max_seq_len - impl_->cache_length) {
    throw std::invalid_argument("CudaModel::ForwardCached KV cache is full");
  }
  for (int64_t i = 0; i < seq_len; ++i) {
    if (host_tokens[i] < 0 || host_tokens[i] >= impl_->config.vocab_size) {
      throw std::invalid_argument("CudaModel::Forward token is out of range");
    }
  }
  // Check the inputs, then copy the token ids and positions to the GPU and
  // look up each word's embedding.
  std::vector<int64_t> host_positions(seq_len);
  for (int64_t i = 0; i < seq_len; ++i) host_positions[i] = start_pos + i;
  impl_->tokens.CopyFromHost(host_tokens, seq_len);
  impl_->positions.CopyFromHost(host_positions.data(), seq_len);
  GatherEmbeddingsCuda(impl_->tokens.data(), impl_->embeddings.data(), impl_->x.data(),
                       seq_len, impl_->config.hidden_size);

  const int64_t d = impl_->config.hidden_size;
  const int64_t q_dim = impl_->config.n_heads * impl_->config.head_dim;
  const int64_t kv_dim = impl_->config.n_kv_heads * impl_->config.head_dim;
  for (const Impl::Layer& layer : impl_->layers) {
    // Attention: normalize, compute queries/keys/values, stamp in positions.
    RmsNormCuda(impl_->x.data(), layer.attn_norm.data(), impl_->normed.data(),
                seq_len, d, impl_->config.rms_eps);
    GemmBTWithHandle(impl_->handle, impl_->normed.data(), layer.wq.data(),
                     impl_->q.data(), seq_len, d, q_dim);
    GemmBTWithHandle(impl_->handle, impl_->normed.data(), layer.wk.data(),
                     impl_->k.data(), seq_len, d, kv_dim);
    GemmBTWithHandle(impl_->handle, impl_->normed.data(), layer.wv.data(),
                     impl_->v.data(), seq_len, d, kv_dim);
    ApplyRopeCuda(impl_->q.data(), impl_->positions.data(), seq_len,
                  impl_->config.n_heads, impl_->config.head_dim,
                  impl_->config.rope_theta);
    ApplyRopeCuda(impl_->k.data(), impl_->positions.data(), seq_len,
                  impl_->config.n_kv_heads, impl_->config.head_dim,
                  impl_->config.rope_theta);
    // With the cache on, save this call's keys and values and attend over
    // everything saved so far.
    const float* k_for_attention = impl_->k.data();
    const float* v_for_attention = impl_->v.data();
    int64_t kv_len = seq_len;
    if (use_cache) {
      const size_t offset = static_cast<size_t>(impl_->cache_length * kv_dim);
      const size_t count = static_cast<size_t>(seq_len * kv_dim);
      const size_t layer_index = static_cast<size_t>(&layer - impl_->layers.data());
      impl_->cache_layers[layer_index].k.CopyFromDevice(impl_->k.data(), count, offset);
      impl_->cache_layers[layer_index].v.CopyFromDevice(impl_->v.data(), count, offset);
      k_for_attention = impl_->cache_layers[layer_index].k.data();
      v_for_attention = impl_->cache_layers[layer_index].v.data();
      kv_len = impl_->cache_length + seq_len;
    }
    AttentionCuda(impl_->q.data(), k_for_attention, v_for_attention,
                  impl_->attn_out.data(), seq_len, kv_len,
                  impl_->config.n_heads, impl_->config.n_kv_heads,
                  impl_->config.head_dim, start_pos);
    GemmBTWithHandle(impl_->handle, impl_->attn_out.data(), layer.wo.data(),
                     impl_->projection.data(), seq_len, q_dim, d);
    AddCuda(impl_->x.data(), impl_->projection.data(), seq_len * d);  // residual

    // Feed-forward network (SwiGLU).
    RmsNormCuda(impl_->x.data(), layer.ffn_norm.data(), impl_->normed.data(),
                seq_len, d, impl_->config.rms_eps);
    GemmBTWithHandle(impl_->handle, impl_->normed.data(), layer.w_gate.data(),
                     impl_->gate.data(), seq_len, d, impl_->config.ffn_hidden);
    GemmBTWithHandle(impl_->handle, impl_->normed.data(), layer.w_up.data(),
                     impl_->up.data(), seq_len, d, impl_->config.ffn_hidden);
    SwiGluActivateCuda(impl_->gate.data(), impl_->up.data(), impl_->hidden.data(),
                       seq_len * impl_->config.ffn_hidden);
    GemmBTWithHandle(impl_->handle, impl_->hidden.data(), layer.w_down.data(),
                     impl_->mlp_out.data(), seq_len, impl_->config.ffn_hidden, d);
    AddCuda(impl_->x.data(), impl_->mlp_out.data(), seq_len * d);
  }
  // Final norm, then score every word in the vocabulary and copy the scores
  // back to the CPU.
  RmsNormCuda(impl_->x.data(), impl_->final_norm.data(), impl_->normed.data(),
              seq_len, d, impl_->config.rms_eps);
  GemmBTWithHandle(impl_->handle, impl_->normed.data(), impl_->lm_head.data(),
                   impl_->logits.data(), seq_len, d, impl_->config.vocab_size);
  CheckCuda(cudaGetLastError(), "CudaModel kernels");
  CheckCuda(cudaDeviceSynchronize(), "CudaModel kernels");
  CheckCuda(cudaMemcpy(out_logits, impl_->logits.data(),
                       static_cast<size_t>(seq_len * impl_->config.vocab_size) * sizeof(float),
                       cudaMemcpyDeviceToHost), "copy logits from device");
  if (use_cache) impl_->cache_length += seq_len;
}

}  // namespace kiln
