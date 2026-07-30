// The pybind11 boundary (constitution §6). Kept thin and audited: this file
// is the ONLY place allowed to know about both Python and the C++ compute
// layer. Nothing here does math; it only marshals a batch descriptor in and
// logits out. Phase 0 exposes a single no-op to prove the build/packaging
// path end-to-end before any real code depends on it.
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kiln {

// Proves the extension loads and the Python <-> C++ boundary works, before
// Phase 2 puts anything real behind it.
std::string Ping() { return "pong"; }

}  // namespace kiln

PYBIND11_MODULE(_C, m) {
  m.doc() = "Kiln compute extension -- the C++/CUDA side of the §6 boundary";
  m.def("ping", &kiln::Ping);
}
