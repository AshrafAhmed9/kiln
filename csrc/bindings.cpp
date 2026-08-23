// The pybind11 boundary (constitution §6). Kept thin and audited: this file
// is the ONLY place allowed to know about both Python and the C++ compute
// layer. It doesn't do any math itself -- it just wraps the real C++
// classes (Model, KVCache, the sampler) so Python can call them.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>

#include "executor/model.h"
#include "executor/sampler.h"
#include "kv/kv_cache.h"
#include "tokenizer/bpe.h"

namespace py = pybind11;

namespace kiln {

std::string Ping() { return "pong"; }

// A thin Python-friendly wrapper around Model::Forward. Instead of asking
// the Python caller to hand us raw memory pointers (which pybind11 could
// do, but which would be an easy way to accidentally corrupt memory from
// the Python side), this takes and returns ordinary numpy arrays and does
// the pointer plumbing itself, once, in this one file.
py::array_t<float> ModelForward(const Model& model,
                                 py::array_t<int32_t> tokens,
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

int32_t SampleFromLogits(py::array_t<float> logits,
                          const SamplerConfig& config,
                          const std::vector<int32_t>& previous_tokens,
                          uint32_t seed) {
  auto buf = logits.request();
  std::mt19937 rng(seed);
  return Sample(static_cast<const float*>(buf.ptr), buf.shape[0], config,
                previous_tokens, rng);
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
      .def_property_readonly("config", &kiln::Model::config);

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
}
