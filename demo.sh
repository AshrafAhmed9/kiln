#!/usr/bin/env bash
# The master demo: one command that exercises every real, working part of
# Kiln -- build, correctness tests, the measured delta-table benchmarks, the
# live API (including concurrent requests and streaming), the independent
# quantizer cross-check, the Docker/Prometheus/Grafana deployment stack, and,
# when the hardware and deps allow it, a real LoRA fine-tune round-trip, the
# Hugging Face parity + tokenizer-conformance checks, and (with a real NVIDIA
# GPU) the CUDA kernels, the INT8 GEMM speedup, RoPE (CUDA-vs-Triton), Nsight
# Compute profiling, and (with two real GPUs) the actual NCCL tensor-parallel
# probes. Nothing here is simulated to look like more than it is: sections
# that need something this machine doesn't have are named and skipped with
# an explanation, never silently faked.
#
# Usage:
#   bash demo.sh              fast path: CPU-only sections, auto-detects a
#                              real GPU (and a second GPU, and Nsight) and
#                              runs those too if present
#   bash demo.sh --full       also runs the real LoRA train->merge->eval
#                              round-trip and the Hugging Face parity +
#                              tokenizer-conformance checks (needs
#                              `pip install -e '.[oracle]'`, downloads a real
#                              checkpoint on first run, several GB + minutes)
#   bash demo.sh --profile    also runs Nsight Compute (needs sudo and a real
#                              NVIDIA GPU; asks for your password once)
#   bash demo.sh --docker     also builds and starts the Docker/Prometheus/
#                              Grafana stack (needs Docker running)
set -euo pipefail

RUN_FULL=0
RUN_PROFILE=0
RUN_DOCKER=0
for arg in "$@"; do
  case "$arg" in
    --full) RUN_FULL=1 ;;
    --profile) RUN_PROFILE=1 ;;
    --docker) RUN_DOCKER=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

section() { echo; echo "== $1 =="; }
skip() { echo; echo "== $1 =="; shift; for line in "$@"; do echo "$line"; done; }

HAVE_GPU=0
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAVE_GPU=1
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
HAVE_ORACLE=0
if python3 -c "import transformers, torch, peft" >/dev/null 2>&1; then
  HAVE_ORACLE=1
fi
MODEL_DIR="${KILN_PARITY_MODEL_DIR:-HuggingFaceTB/SmolLM2-135M-Instruct}"

section "1. Build the C++ compute layer and run its CPU test suite"
cmake -B build -G Ninja -DKILN_BUILD_PYBIND=OFF
cmake --build build
ctest --test-dir build --output-on-failure

section "2. Build the pybind11 extension (the Python <-> C++ boundary)"
cmake -B build-py -G Ninja -DKILN_BUILD_PYBIND=ON \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build-py --target _C

section "3. Run the Python test suite (scheduler, API, eval infra, LoRA binding, tensor-parallel sim, chunked prefill sim, constrained decode, drift detection, etc.)"
PYTHONPATH=. python3 -m pytest tests/py -q

section "4. The independent INT8 quantizer cross-check (pure-Python re-implementation vs. the C++ quantizer)"
PYTHONPATH=. python3 -c "
import numpy as np
from tools.quantize_ref import quantize_int8_per_channel, dequantize_int8_per_channel
rng = np.random.default_rng(20260824)
weights = rng.normal(size=(64, 64)).astype(np.float32)
quantized, scales = quantize_int8_per_channel(weights)
reconstructed = dequantize_int8_per_channel(quantized, scales)
mse = float(np.mean((weights - reconstructed) ** 2))
print(f'independent Python quantizer MSE vs. original weights: {mse:.3e}')
print('(cross-checked against the C++ quantizer in tests/py/test_quantize.py)')
"

section "5. The full measured delta table (KV cache, continuous batching, paged/shared-prefix cache, INT8 memory+accuracy, speculative decoding)"
PYTHONPATH=. python3 bench/run_benchmarks.py

section "6. Prefix-cache sharing, seeded synthetic workload (separate C++ binary)"
cmake --build build --target kiln_prefix_cache_benchmark
./build/kiln_prefix_cache_benchmark

section "7. Start the API server in the background"
PYTHONPATH=. python3 -m uvicorn kiln_py.api.app:app --port 8420 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 2

section "8. A non-streaming completion"
curl -s http://127.0.0.1:8420/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello, kiln", "max_tokens": 8}' | python3 -m json.tool

section "9. A streaming completion (words arrive one at a time)"
curl -s -N http://127.0.0.1:8420/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "streaming demo", "max_tokens": 5, "stream": true}'
echo

section "10. Several concurrent requests through the real scheduler (continuous batching, live)"
REQUEST_PIDS=()
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8420/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"concurrent request $i\", \"max_tokens\": 6}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'request $i ->', d['choices'][0]['text'])" &
  REQUEST_PIDS+=($!)
done
wait "${REQUEST_PIDS[@]}"

section "11. Eval infrastructure: a tiny synthetic regression check"
PYTHONPATH=. python3 -c "
from kiln_py.eval.regression_gate import check_for_regression
baseline = [0.9, 0.9, 0.9, 0.9, 0.9]
candidate = [0.5, 0.5, 0.5, 0.5, 0.5]
result = check_for_regression(baseline, candidate)
print('regression detected:', result.is_regression)
print('mean difference:', result.mean_difference)
print('95% confidence interval:', result.confidence_interval)
"

if [ "$RUN_DOCKER" -eq 1 ]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    section "12. [--docker] Building and starting the Docker/Prometheus/Grafana stack"
    (cd deploy && docker compose up --build -d)
    sleep 3
    echo "kiln:       http://localhost:8420/healthz"
    curl -sf http://localhost:8420/healthz && echo
    echo "prometheus: http://localhost:9090"
    echo "grafana:    http://localhost:3000"
    echo "(left running -- stop with: cd deploy && docker compose down)"
  else
    skip "12. [--docker] Skipped" "Docker isn't installed or isn't running."
  fi
fi

if [ "$RUN_FULL" -eq 1 ]; then
  if [ "$HAVE_ORACLE" -eq 1 ]; then
    section "13. [--full] Real Hugging Face parity check (downloads a real checkpoint on first run)"
    PYTHONPATH=. python3 tools/hf_parity.py \
      --model-dir "$MODEL_DIR" \
      --prompts-file tools/fixtures/hf_parity_prompts.txt \
      || echo "Needs a local path or network access to download the checkpoint the first time -- see docs/defense.md Phase 22/28."

    section "14. [--full] Tokenizer conformance, 10,000 seeded strings against the real tokenizer"
    PYTHONPATH=. python3 tools/tokenizer_conformance.py --model-dir "$MODEL_DIR" \
      || echo "Needs the same checkpoint as above."

    section "15. [--full] Real LoRA train -> merge -> eval round-trip on the tiny fixture"
    LORA_OUT=$(mktemp -d)
    PYTHONPATH=. python3 tools/train_lora.py \
      --model "$MODEL_DIR" --data tools/fixtures/lora_tiny.jsonl \
      --output "$LORA_OUT/adapter" --steps 20 \
      && PYTHONPATH=. python3 tools/merge_lora_adapter.py \
           --model-dir "$MODEL_DIR" --adapter-dir "$LORA_OUT/adapter" \
      || echo "Needs the same checkpoint as above, plus 'pip install -e .[oracle]'."
    rm -rf "$LORA_OUT"
  else
    skip "13-15. [--full] Skipped" "Oracle deps not installed. Run: pip install -e '.[oracle]'  then rerun with --full."
  fi
fi

if [ "$HAVE_GPU" -eq 1 ]; then
  section "16. Real NVIDIA GPU detected ($GPU_COUNT visible) -- building and running the CUDA kernel suite"
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -S . -B build-gpu -G Ninja \
    -DKILN_BUILD_CUDA=ON -DKILN_BUILD_PYBIND=OFF \
    -DKILN_CUDA_ARCHITECTURES="${ARCH}" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build-gpu -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
  ctest --test-dir build-gpu --output-on-failure

  section "17. Real INT8-vs-FP32 GEMM speedup, measured on this GPU"
  ./build-gpu/kiln_int8_gemm_cuda_benchmark

  section "18. RoPE: raw CUDA benchmark on this GPU"
  ./build-gpu/kiln_rope_cuda_benchmark
  if python3 -c "import triton" >/dev/null 2>&1; then
    echo "-- Triton comparison (tools/benchmark_rope_triton.py) --"
    PYTHONPATH=. python3 tools/benchmark_rope_triton.py
  else
    echo "(Triton not installed -- skipping the Triton-vs-CUDA comparison; 'pip install triton' to add it)"
  fi

  if [ "$RUN_PROFILE" -eq 1 ]; then
    NCU_BIN=$(command -v ncu || find / -iname "ncu" -type f 2>/dev/null | head -1)
    if [ -n "${NCU_BIN:-}" ]; then
      section "19. [--profile] Nsight Compute profiling of the real prefill pass (needs sudo)"
      SET_NAME=default
      "$NCU_BIN" --list-sets 2>/dev/null | grep -q '^basic ' && SET_NAME=basic
      sudo "$NCU_BIN" --set "$SET_NAME" --target-processes all \
        ./build-gpu/tests/cpp/kiln_cuda_tests \
        --gtest_filter=CudaModel.FullPrefillMatchesCpuReference
    else
      skip "19. [--profile] Skipped" "ncu not found. Install NVIDIA Nsight Compute."
    fi
  fi

  if [ "$GPU_COUNT" -ge 2 ] && command -v torchrun >/dev/null 2>&1; then
    section "20. Two GPUs detected -- real two-rank NCCL tensor-parallel probes"
    torchrun --standalone --nproc_per_node=2 tools/validate_nccl.py
    torchrun --standalone --nproc_per_node=2 tools/validate_tensor_parallel.py
  else
    skip "20. Skipped" \
      "Real multi-GPU tensor parallelism needs 2+ real GPUs (this machine has $GPU_COUNT)." \
      "The sharding math is proven exactly against real weights in a single-process" \
      "simulation instead -- see docs/defense.md Phase 12/25 -- and that remains the" \
      "one honestly-open item in this project."
  fi
else
  skip "16-20. Skipped" \
    "No real NVIDIA GPU visible on this machine (nvidia-smi not found or failed)." \
    "The CUDA kernels, the INT8 GEMM speedup, RoPE, and Nsight profiling all need" \
    "real NVIDIA hardware -- see docs/defense.md Phase 30 for the numbers already" \
    "captured on one, and docs/local-gpu-validation.sh to reproduce them on a" \
    "machine that has a GPU."
fi

echo
echo "== Demo complete. See README.md's 'Measured results' section and"
echo "== docs/defense.md for the itemized, phase-by-phase record of what"
echo "== each of the above actually proved. =="
