#pragma once

#include <cstdint>
#include <memory>

#include "executor/model.h"

namespace kiln {

// Device-resident, uncached forward pass for one sequence. It deliberately
// has a smaller contract than Model: GPU KV-cache ownership and batched
// scheduling need their own device data layouts, so treating either as a
// hidden fallback to the CPU path would make the executor's behavior unclear.
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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kiln
