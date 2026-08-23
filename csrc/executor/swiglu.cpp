#include "executor/swiglu.h"

#include <cmath>
#include <vector>

#include "executor/gemm.h"

namespace kiln {

void SwiGlu(const float* x, const float* w_gate, const float* w_up,
            const float* w_down, float* out, int64_t n_rows, int64_t dim,
            int64_t ffn_hidden) {
  std::vector<float> gate(n_rows * ffn_hidden);
  std::vector<float> up(n_rows * ffn_hidden);
  GemmBT(x, w_gate, gate.data(), n_rows, dim, ffn_hidden);
  GemmBT(x, w_up, up.data(), n_rows, dim, ffn_hidden);

  std::vector<float> hidden(n_rows * ffn_hidden);
  for (int64_t i = 0; i < n_rows * ffn_hidden; ++i) {
    float g = gate[i];
    float silu = g / (1.0f + std::exp(-g));  // silu(g) = g * sigmoid(g)
    hidden[i] = silu * up[i];
  }

  GemmBT(hidden.data(), w_down, out, n_rows, ffn_hidden, dim);
}

}  // namespace kiln
