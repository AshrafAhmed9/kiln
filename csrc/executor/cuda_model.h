#pragma once

#include <cstdint>
#include <memory>

#include "executor/model.h"

namespace kiln {

// The model running on the GPU, for one text at a time. It keeps its own KV
// cache in GPU memory, one contiguous buffer like KVCache. (The paged cache
// is CPU-only.)
class CudaModel {
 public:
  explicit CudaModel(const Model& model);
  ~CudaModel();

  CudaModel(const CudaModel&) = delete;
  CudaModel& operator=(const CudaModel&) = delete;
  CudaModel(CudaModel&&) noexcept;
  CudaModel& operator=(CudaModel&&) noexcept;

  // Runs a whole text with no cache and copies the scores, shape
  // [seq_len, vocab_size], back to the CPU. start_pos means the same as in
  // Model::Forward.
  void Forward(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
               float* out_logits) const;

  // Like Forward, but saves keys and values in the GPU's own cache.
  // start_pos must equal the number of words already cached, to catch
  // mixing up two different texts.
  void ForwardCached(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
                     float* out_logits) const;
  void ResetCache() const;
  const ModelConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void ForwardImpl(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
                   float* out_logits, bool use_cache) const;
};

}  // namespace kiln
