# ADR-003: Dependency allowlist

**Status:** Accepted

**Allowed:** cuBLAS (Part II GEMM only), a minimal HTTP library
(`cpp-httplib`), `nlohmann/json`, HuggingFace `transformers`/`tokenizers`
(test/tooling only), GoogleTest + Google Benchmark, the Prometheus C++
client.

**Everything else is hand-built:** safetensors loader, BPE tokenizer, CPU
GEMM, every attention variant, the paged KV allocator, the continuous-
batching scheduler, the sampler, CUDA kernels (except GEMM), speculative
decoding, the quantizer. No PyTorch/libtorch in the engine.

**Why:** the point of the project is understanding what a tensor framework or
serving library would otherwise hide. Adding anything to this list requires a
new ADR arguing that hand-building it would teach nothing — the same bar the
constitution sets.
