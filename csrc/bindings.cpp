// The bridge between Python and C++ (built with pybind11). This is the only
// file that knows about both sides. It does no math; it wraps the C++
// classes (Model, KVCache, the sampler, the tokenizer) so Python can call
// them as `kiln_py._C`.
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

// Python passes numpy arrays, and the C++ code reads them as plain flat
// memory. These two types make pybind11 first copy any array that is not
// already flat and the right type (a transposed view, a slice, the wrong
// dtype) into one that is. Without that, C++ would read the wrong numbers
// without any error.
using ContiguousFloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ContiguousInt32Array =
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

py::array_t<float> ModelForward(const Model& model, ContiguousInt32Array tokens,
                                int64_t start_pos, KVCache* cache) {
  auto input = tokens.request();
  if (input.ndim != 1) throw std::invalid_argument("tokens must be 1D");
  int64_t seq_len = input.shape[0];
  py::array_t<float> out({seq_len, model.config().vocab_size});
  model.Forward(static_cast<const int32_t*>(input.ptr), seq_len, start_pos,
                cache, static_cast<float*>(out.request().ptr));
  return out;
}

py::list ModelForwardHiddenStates(const Model& model,
                                  ContiguousInt32Array tokens) {
  auto input = tokens.request();
  if (input.ndim != 1) throw std::invalid_argument("tokens must be 1D");
  std::vector<std::vector<float>> states;
  std::vector<float> logits(input.shape[0] * model.config().vocab_size);
  model.Forward(static_cast<const int32_t*>(input.ptr), input.shape[0], 0,
                nullptr, logits.data(), &states);
  py::list out;
  for (const auto& state : states) {
    py::array_t<float> array(
        std::vector<py::ssize_t>{input.shape[0], model.config().hidden_size});
    std::memcpy(array.mutable_data(), state.data(),
                state.size() * sizeof(float));
    out.append(std::move(array));
  }
  return out;
}

void ModelMergeLoraIntoLayer(Model& model, int64_t layer_idx,
                             const std::string& which,
                             ContiguousFloatArray lora_a,
                             ContiguousFloatArray lora_b, float scale) {
  auto a = lora_a.request();
  auto b = lora_b.request();
  if (a.ndim != 2 || b.ndim != 2 || a.shape[0] != b.shape[1]) {
    throw std::invalid_argument(
        "LoRA arrays must be A[rank, in] and B[out, rank]");
  }
  model.MergeLoraIntoLayer(layer_idx, which, static_cast<const float*>(a.ptr),
                           static_cast<const float*>(b.ptr), a.shape[0], scale);
}

#ifdef KILN_BUILD_CUDA
py::array_t<float> CudaModelForward(const CudaModel& model,
                                    ContiguousInt32Array tokens,
                                    int64_t start_pos, bool use_cache) {
  auto input = tokens.request();
  if (input.ndim != 1) throw std::invalid_argument("tokens must be 1D");
  const int64_t seq_len = input.shape[0];
  // The CUDA model checks token ids and cache position itself; this wrapper
  // only converts between numpy arrays and pointers.
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

// Decoded tokens are raw bytes, and are not always valid text. One character
// can be split across several tokens, and a random-weight model can output
// any byte. pybind11 would turn a std::string into a Python str and crash on
// invalid text, so this returns Python `bytes` instead and lets the Python
// side decide (see kiln_py/runtime/generate.py).
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

// Wraps QuantizeInt8PerChannel for Python: takes a 2D numpy array and
// returns (int8 values, per-row scales). tests/py/test_quantize.py compares
// it with the separate Python version in tools/quantize_ref.py.
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
  m.doc() =
      "Kiln compute extension -- the C++/CUDA side of the pybind11 boundary";
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
           py::arg("start_pos") = 0, py::arg("cache") = nullptr)
      .def("forward_hidden_states", &kiln::ModelForwardHiddenStates,
           py::arg("tokens"))
      .def("merge_lora_into_layer", &kiln::ModelMergeLoraIntoLayer,
           py::arg("layer_idx"), py::arg("which"), py::arg("lora_a"),
           py::arg("lora_b"), py::arg("scale"))
      .def_property_readonly("config", &kiln::Model::config);

#ifdef KILN_BUILD_CUDA
  py::class_<kiln::CudaModel>(m, "CudaModel")
      .def(py::init<const kiln::Model&>())
      .def(
          "forward",
          [](const kiln::CudaModel& model, kiln::ContiguousInt32Array tokens,
             int64_t start_pos) {
            return kiln::CudaModelForward(model, std::move(tokens), start_pos,
                                          /*use_cache=*/false);
          },
          py::arg("tokens"), py::arg("start_pos") = 0)
      .def(
          "forward_cached",
          [](const kiln::CudaModel& model, kiln::ContiguousInt32Array tokens,
             int64_t start_pos) {
            return kiln::CudaModelForward(model, std::move(tokens), start_pos,
                                          /*use_cache=*/true);
          },
          py::arg("tokens"), py::arg("start_pos"))
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

  m.def("sample", &kiln::SampleFromLogits, py::arg("logits"), py::arg("config"),
        py::arg("previous_tokens"), py::arg("seed"));

  py::class_<kiln::BpeTokenizer>(m, "BpeTokenizer")
      .def_static("load", &kiln::BpeTokenizer::Load)
      .def("encode", &kiln::BpeTokenizer::Encode)
      .def("decode", &kiln::DecodeToBytes);

  m.def("quantize_int8_per_channel", &kiln::QuantizeInt8PerChannelPy,
        py::arg("weights"));
}
