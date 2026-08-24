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

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

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

struct CudaModel::Impl {
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
    CheckCuBlas(cublasCreate(&handle), "cublasCreate");
  }

  ~Impl() { cublasDestroy(handle); }

  ModelConfig config;
  cublasHandle_t handle = nullptr;
  DeviceBuffer<float> embeddings;
  DeviceBuffer<float> final_norm;
  DeviceBuffer<float> lm_head;
  std::vector<Layer> layers;
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

void CudaModel::Forward(const int32_t* host_tokens, int64_t seq_len,
                        int64_t start_pos, float* out_logits) const {
  if (seq_len <= 0 || seq_len > impl_->config.max_seq_len) {
    throw std::invalid_argument("CudaModel::Forward sequence length is out of range");
  }
  for (int64_t i = 0; i < seq_len; ++i) {
    if (host_tokens[i] < 0 || host_tokens[i] >= impl_->config.vocab_size) {
      throw std::invalid_argument("CudaModel::Forward token is out of range");
    }
  }
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
    AttentionCuda(impl_->q.data(), impl_->k.data(), impl_->v.data(),
                  impl_->attn_out.data(), seq_len, seq_len,
                  impl_->config.n_heads, impl_->config.n_kv_heads,
                  impl_->config.head_dim, start_pos, -1);
    GemmBTWithHandle(impl_->handle, impl_->attn_out.data(), layer.wo.data(),
                     impl_->projection.data(), seq_len, q_dim, d);
    AddCuda(impl_->x.data(), impl_->projection.data(), seq_len * d);

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
  RmsNormCuda(impl_->x.data(), impl_->final_norm.data(), impl_->normed.data(),
              seq_len, d, impl_->config.rms_eps);
  GemmBTWithHandle(impl_->handle, impl_->normed.data(), impl_->lm_head.data(),
                   impl_->logits.data(), seq_len, d, impl_->config.vocab_size);
  CheckCuda(cudaGetLastError(), "CudaModel kernels");
  CheckCuda(cudaDeviceSynchronize(), "CudaModel kernels");
  CheckCuda(cudaMemcpy(out_logits, impl_->logits.data(),
                       static_cast<size_t>(seq_len * impl_->config.vocab_size) * sizeof(float),
                       cudaMemcpyDeviceToHost), "copy logits from device");
}

}  // namespace kiln
