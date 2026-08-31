// The pybind11 boundary (constitution §6). Kept thin and audited: this file
// is the ONLY place allowed to know about both Python and the C++ compute
// layer. It doesn't do any math itself -- it just wraps the real C++
// classes (Model, KVCache, the sampler) so Python can call them.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>

#include "executor/model.h"
#ifdef KILN_BUILD_CUDA
#include "executor/cuda_model.h"
#endif
#include "executor/sampler.h"
#include "kv/kv_cache.h"
#include "quant/quantize.h"
#include "tokenizer/bpe.h"

namespace py = pybind11;

namespace kiln {

std::string Ping() { return "pong"; }

// A thin Python-friendly wrapper around Model::Forward. Instead of asking
// the Python caller to hand us raw memory pointers (which pybind11 could
// do, but which would be an easy way to accidentally corrupt memory from
// the Python side), this takes and returns ordinary numpy arrays and does
// the pointer plumbing itself, once, in this one file.
// Every numpy array accepted across the boundary below is declared
// c_style|forcecast: this makes pybind11 copy the data into a plain,
// contiguous, correctly-typed buffer first if the caller ever passes
// something else (a transposed view, a non-contiguous slice, a different
// dtype). Without this, the code below reads the array's raw pointer as
// if it were a flat contiguous buffer regardless of what it actually is
// -- silently reading the wrong numbers instead of failing.
using ContiguousFloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ContiguousInt32Array =
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

py::array_t<float> ModelForward(const Model& model,
                                 ContiguousInt32Array tokens,
                                 int64_t batch_size, int64_t seq_len,
                                 py::object valid_lengths, int64_t start_pos,
                                 KVCache* cache) {
  auto tokens_buf = tokens.request();
  const int64_t* valid_ptr = nullptr;
  std::vector<int64_t> valid_storage;
  if (!valid_lengths.is_none()) {
    valid_storage = valid_lengths.cast<std::vector<int64_t>>();
    valid_ptr = valid_storage.data();
  }

  py::array_t<float> out({batch_size * seq_len, model.config().vocab_size});
  model.Forward(static_cast<const int32_t*>(tokens_buf.ptr), batch_size,
                seq_len, valid_ptr, start_pos, cache,
                static_cast<float*>(out.request().ptr));
  return out;
}

py::list ModelForwardHiddenStates(const Model& model, ContiguousInt32Array tokens) {
  auto input = tokens.request();
  if (input.ndim != 1) throw std::invalid_argument("tokens must be 1D");
  std::vector<std::vector<float>> states;
  std::vector<float> logits(input.shape[0] * model.config().vocab_size);
  model.Forward(static_cast<const int32_t*>(input.ptr), 1, input.shape[0], nullptr,
                0, nullptr, logits.data(), &states);
  py::list out;
  for (const auto& state : states) {
    py::array_t<float> array(
        std::vector<py::ssize_t>{input.shape[0], model.config().hidden_size});
    std::memcpy(array.mutable_data(), state.data(), state.size() * sizeof(float));
    out.append(std::move(array));
  }
  return out;
}

py::array_t<float> ModelForwardDecodeBatch(const Model& model,
                                            ContiguousInt32Array tokens,
                                            std::vector<int64_t> positions,
                                            py::list caches) {
  auto tokens_buf = tokens.request();
  int64_t batch_size = tokens_buf.shape[0];
  if (tokens_buf.ndim != 1 || static_cast<int64_t>(positions.size()) != batch_size ||
      static_cast<int64_t>(py::len(caches)) != batch_size) {
    throw std::invalid_argument(
        "forward_decode_batch expects equally sized 1D tokens, positions, and caches");
  }
  std::vector<KVCache*> cache_ptrs;
  cache_ptrs.reserve(batch_size);
  for (py::handle cache : caches) cache_ptrs.push_back(cache.cast<KVCache*>());

  py::array_t<float> out({batch_size, model.config().vocab_size});
  model.ForwardDecodeBatch(static_cast<const int32_t*>(tokens_buf.ptr),
                           batch_size, positions.data(), cache_ptrs.data(),
                           static_cast<float*>(out.request().ptr));
  return out;
}

// Ragged prefill: `tokens` is every sequence's real tokens concatenated
// back-to-back (no padding), `seq_lengths` gives each one's real length,
// and `caches` gives each one's own KV cache. Mirrors
// ModelForwardDecodeBatch's shape above -- the Python-side change per
// call, C++ side does the real work through the exact same binding
// pattern.
py::array_t<float> ModelForwardPrefillBatch(const Model& model,
                                             ContiguousInt32Array tokens,
                                             std::vector<int64_t> seq_lengths,
                                             py::list caches) {
  auto tokens_buf = tokens.request();
  int64_t num_sequences = static_cast<int64_t>(seq_lengths.size());
  if (tokens_buf.ndim != 1 || static_cast<int64_t>(py::len(caches)) != num_sequences) {
    throw std::invalid_argument(
        "forward_prefill_batch expects one seq_length and one cache per sequence");
  }
  int64_t total_tokens = 0;
  for (int64_t len : seq_lengths) total_tokens += len;
  if (total_tokens != tokens_buf.shape[0]) {
    throw std::invalid_argument(
        "forward_prefill_batch: seq_lengths must sum to len(tokens)");
  }

  std::vector<KVCache*> cache_ptrs;
  cache_ptrs.reserve(num_sequences);
  for (py::handle cache : caches) cache_ptrs.push_back(cache.cast<KVCache*>());

  py::array_t<float> out({total_tokens, model.config().vocab_size});
  model.ForwardPrefillBatch(static_cast<const int32_t*>(tokens_buf.ptr),
                            num_sequences, seq_lengths.data(),
                            cache_ptrs.data(),
                            static_cast<float*>(out.request().ptr));
  return out;
}

void ModelMergeLoraIntoLayer(Model& model, int64_t layer_idx,
                             const std::string& which,
                             ContiguousFloatArray lora_a,
                             ContiguousFloatArray lora_b, float scale) {
  auto a = lora_a.request();
  auto b = lora_b.request();
  if (a.ndim != 2 || b.ndim != 2 || a.shape[0] != b.shape[1]) {
    throw std::invalid_argument("LoRA arrays must be A[rank, in] and B[out, rank]");
  }
  model.MergeLoraIntoLayer(layer_idx, which,
                            static_cast<const float*>(a.ptr),
                            static_cast<const float*>(b.ptr),
                            a.shape[0], scale);
}

#ifdef KILN_BUILD_CUDA
py::array_t<float> CudaModelForward(const CudaModel& model,
                                    ContiguousInt32Array tokens,
                                    int64_t start_pos, bool use_cache) {
  auto input = tokens.request();
  if (input.ndim != 1) throw std::invalid_argument("tokens must be 1D");
  const int64_t seq_len = input.shape[0];
  // The CUDA executor validates token IDs and cache position on the C++ side;
  // this wrapper only owns safe NumPy pointer and output-shape handling.
  py::array_t<float> out({seq_len, model.config().vocab_size});
  if (use_cache) {
    model.ForwardCached(static_cast<const int32_t*>(input.ptr), seq_len,
                        start_pos, static_cast<float*>(out.request().ptr));
  } else {
    model.Forward(static_cast<const int32_t*>(input.ptr), seq_len, start_pos,
                  static_cast<float*>(out.request().ptr));
  }
  return out;
}
#endif

// A byte-level tokenizer's decoded output is raw bytes, not necessarily
// valid UTF-8 text on its own -- especially one word (token) at a time, since
// a single character in a real alphabet can be spread across more than one
// token, and an untrained model can output any byte value at all. pybind11
// would otherwise convert a C++ std::string straight into a Python string by
// assuming it's valid UTF-8, which crashes the moment that assumption is
// wrong. Returning raw Python bytes instead hands that decision to the
// Python side, which knows how to handle it gracefully (see
// kiln_py/runtime/generate.py).
py::bytes DecodeToBytes(const BpeTokenizer& tokenizer,
                        const std::vector<int32_t>& ids) {
  return py::bytes(tokenizer.Decode(ids));
}

int32_t SampleFromLogits(ContiguousFloatArray logits,
                          const SamplerConfig& config,
                          const std::vector<int32_t>& previous_tokens,
                          uint32_t seed) {
  auto buf = logits.request();
  std::mt19937 rng(seed);
  return Sample(static_cast<const float*>(buf.ptr), buf.shape[0], config,
                previous_tokens, rng);
}

// Wraps QuantizeInt8PerChannel for Python: takes a 2D numpy array, returns
// (quantized int8 array, per-row scales) -- used to cross-check this
// implementation against the independent Python reference in
// tools/quantize_ref.py (see tests/py/test_quantize.py).
py::tuple QuantizeInt8PerChannelPy(ContiguousFloatArray weights) {
  auto buf = weights.request();
  int64_t rows = buf.shape[0];
  int64_t cols = buf.shape[1];

  py::array_t<int8_t> quantized({rows, cols});
  py::array_t<float> scales(rows);
  QuantizeInt8PerChannel(static_cast<const float*>(buf.ptr), rows, cols,
                         static_cast<int8_t*>(quantized.request().ptr),
                         static_cast<float*>(scales.request().ptr));
  return py::make_tuple(quantized, scales);
}

}  // namespace kiln

PYBIND11_MODULE(_C, m) {
  m.doc() = "Kiln compute extension -- the C++/CUDA side of the §6 boundary";
  m.def("ping", &kiln::Ping);

  py::class_<kiln::ModelConfig>(m, "ModelConfig")
      .def(py::init<>())
      .def_readwrite("vocab_size", &kiln::ModelConfig::vocab_size)
      .def_readwrite("hidden_size", &kiln::ModelConfig::hidden_size)
      .def_readwrite("n_layers", &kiln::ModelConfig::n_layers)
      .def_readwrite("n_heads", &kiln::ModelConfig::n_heads)
      .def_readwrite("n_kv_heads", &kiln::ModelConfig::n_kv_heads)
      .def_readwrite("head_dim", &kiln::ModelConfig::head_dim)
      .def_readwrite("ffn_hidden", &kiln::ModelConfig::ffn_hidden)
      .def_readwrite("max_seq_len", &kiln::ModelConfig::max_seq_len)
      .def_readwrite("rms_eps", &kiln::ModelConfig::rms_eps)
      .def_readwrite("rope_theta", &kiln::ModelConfig::rope_theta);

  py::class_<kiln::Model>(m, "Model")
      .def_static("load_random", &kiln::Model::LoadRandom)
      .def_static("load_from_safetensors", &kiln::Model::LoadFromSafetensors)
      .def("forward", &kiln::ModelForward, py::arg("tokens"),
           py::arg("batch_size"), py::arg("seq_len"),
           py::arg("valid_lengths") = py::none(), py::arg("start_pos") = 0,
           py::arg("cache") = nullptr)
      .def("forward_hidden_states", &kiln::ModelForwardHiddenStates,
           py::arg("tokens"))
      .def("forward_decode_batch", &kiln::ModelForwardDecodeBatch,
           py::arg("tokens"), py::arg("positions"), py::arg("caches"))
      .def("forward_prefill_batch", &kiln::ModelForwardPrefillBatch,
           py::arg("tokens"), py::arg("seq_lengths"), py::arg("caches"))
      .def("merge_lora_into_layer", &kiln::ModelMergeLoraIntoLayer,
           py::arg("layer_idx"), py::arg("which"), py::arg("lora_a"),
           py::arg("lora_b"), py::arg("scale"))
      .def_property_readonly("config", &kiln::Model::config);

#ifdef KILN_BUILD_CUDA
  py::class_<kiln::CudaModel>(m, "CudaModel")
      .def(py::init<const kiln::Model&>())
      .def("forward", [](const kiln::CudaModel& model,
                          kiln::ContiguousInt32Array tokens,
                          int64_t start_pos) {
        return kiln::CudaModelForward(model, std::move(tokens), start_pos,
                                      /*use_cache=*/false);
      }, py::arg("tokens"), py::arg("start_pos") = 0)
      .def("forward_cached", [](const kiln::CudaModel& model,
                                 kiln::ContiguousInt32Array tokens,
                                 int64_t start_pos) {
        return kiln::CudaModelForward(model, std::move(tokens), start_pos,
                                      /*use_cache=*/true);
      }, py::arg("tokens"), py::arg("start_pos"))
      .def("reset_cache", &kiln::CudaModel::ResetCache)
      .def_property_readonly("config", &kiln::CudaModel::config);
#endif

  py::class_<kiln::KVCache>(m, "KVCache")
      .def(py::init<int64_t, int64_t, int64_t, int64_t>(), py::arg("n_layers"),
           py::arg("max_seq_len"), py::arg("n_kv_heads"), py::arg("head_dim"))
      .def_property_readonly("length", &kiln::KVCache::length);

  py::class_<kiln::SamplerConfig>(m, "SamplerConfig")
      .def(py::init<>())
      .def_readwrite("temperature", &kiln::SamplerConfig::temperature)
      .def_readwrite("top_k", &kiln::SamplerConfig::top_k)
      .def_readwrite("top_p", &kiln::SamplerConfig::top_p)
      .def_readwrite("repetition_penalty",
                      &kiln::SamplerConfig::repetition_penalty);

  m.def("sample", &kiln::SampleFromLogits, py::arg("logits"),
        py::arg("config"), py::arg("previous_tokens"), py::arg("seed"));

  py::class_<kiln::BpeTokenizer>(m, "BpeTokenizer")
      .def_static("load", &kiln::BpeTokenizer::Load)
      .def("encode", &kiln::BpeTokenizer::Encode)
      .def("decode", &kiln::DecodeToBytes);

  m.def("quantize_int8_per_channel", &kiln::QuantizeInt8PerChannelPy,
        py::arg("weights"));
}
