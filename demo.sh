#!/usr/bin/env bash
# One-command demo of Kiln. Run: bash demo.sh
#
# Always runs: build, C++ and Python tests, CPU benchmarks, and the HTTP API.
# A cached SmolLM2-135M-Instruct is used automatically; otherwise the API serves
# a small random-weight model (real request shapes, meaningless words).
# Runs when available:
#   KILN_CHECKPOINT_DIR=<dir>   pick a checkpoint, and compare it to Hugging Face
#   KILN_DEVICE=cuda            serve it on the GPU
#   a CUDA build                INT8 vs FP32 GEMM benchmark
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
SERVER_PID=""

# Use a cached SmolLM2 checkpoint if one exists and none was given.
if [ -z "${KILN_CHECKPOINT_DIR:-}" ]; then
  for d in "$HOME"/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/; do
    [ -f "${d}model.safetensors" ] && export KILN_CHECKPOINT_DIR="${d%/}"
  done
fi
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true' EXIT

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "1/6  Build the C++ engine and the Python extension"
cmake -B build -G Ninja >/dev/null
cmake --build build 2>&1 | tail -2
"$PY" -m pip install -q -e ".[dev]" 2>&1 | tail -1 || true

step "2/6  C++ tests (tokenizer, forward pass, KV cache, paged cache, INT8, LoRA)"
ctest --test-dir build --output-on-failure | tail -4

step "3/6  Python tests (API, CUDA runtime, LoRA binding, quantization)"
"$PY" -m pytest tests/py -q 2>&1 | tail -2

step "4/6  Benchmarks: KV cache vs no cache, latency, memory, INT8"
PYTHONPATH=. "$PY" bench/run_benchmarks.py

step "5/6  Serve over HTTP and ask it for text"
if [ -n "${KILN_CHECKPOINT_DIR:-}" ]; then
  echo "Real checkpoint: $KILN_CHECKPOINT_DIR (device: ${KILN_DEVICE:-cpu})"
else
  echo "No checkpoint found: using the small random-weight model."
  echo "The request and response shapes are real; the words are not."
  echo "For real text, set KILN_CHECKPOINT_DIR to a SmolLM2 folder."
fi
"$PY" -m uvicorn kiln_py.api.app:app --port "$PORT" --log-level warning &
SERVER_PID=$!
for _ in $(seq 1 60); do
  curl -sf "localhost:$PORT/healthz" >/dev/null && break
  sleep 0.5
done
echo; echo "GET /healthz"
curl -s "localhost:$PORT/healthz"; echo
echo; echo "POST /v1/completions"
curl -s "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 20, "temperature": 0.0}'; echo
echo; echo "POST /v1/completions (streamed)"
curl -sN "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 5, "stream": true, "temperature": 0.0}'
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

step "6/6  Correctness and GPU"
if [ -n "${KILN_CHECKPOINT_DIR:-}" ] && "$PY" -c "import torch, transformers" 2>/dev/null; then
  echo "Comparing Kiln's logits to Hugging Face on 10 prompts:"
  PYTHONPATH=. "$PY" tools/hf_parity.py --model-dir "$KILN_CHECKPOINT_DIR" \
    --prompts-file tools/fixtures/hf_parity_prompts.txt
else
  echo "Skipped Hugging Face comparison. Needs a checkpoint and: pip install -e '.[oracle]'"
fi
if [ -x build/kiln_int8_gemm_cuda_benchmark ]; then
  echo; echo "INT8 vs FP32 GEMM on this GPU:"
  ./build/kiln_int8_gemm_cuda_benchmark
else
  echo "Skipped GPU benchmark. Needs a CUDA build: cmake -B build -G Ninja -DKILN_BUILD_CUDA=ON -DKILN_CUDA_ARCHITECTURES=<arch>"
fi

printf '\n\033[1mDone.\033[0m\n'
