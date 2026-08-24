#pragma once

#include <cstdint>
#include <memory>

#include "executor/model.h"

namespace kiln {

// Device-resident forward pass for one sequence. Its contiguous GPU KV cache
// intentionally matches KVCache's single-sequence contract; batching and
// paging need a different device layout and are not hidden behind a CPU
// fallback.
class CudaModel {
 public:
  explicit CudaModel(const Model& model);
  ~CudaModel();

  CudaModel(const CudaModel&) = delete;
  CudaModel& operator=(const CudaModel&) = delete;
  CudaModel(CudaModel&&) noexcept;
  CudaModel& operator=(CudaModel&&) noexcept;

  // Runs one complete prefill sequence and copies [seq_len, vocab_size]
  // logits back to host memory. `start_pos` has the same meaning as
  // Model::Forward, but no KV cache is accepted by this initial executor.
  void Forward(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
               float* out_logits) const;

  // Appends tokens to the executor-owned GPU cache. start_pos must equal the
  // current cache length, which prevents a caller from accidentally using
  // keys from one sequence with positions from another.
  void ForwardCached(const int32_t* tokens, int64_t seq_len,
                     int64_t start_pos, float* out_logits) const;
  void ResetCache() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void ForwardImpl(const int32_t* tokens, int64_t seq_len, int64_t start_pos,
                   float* out_logits, bool use_cache) const;
};

}  // namespace kiln
