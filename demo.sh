#!/usr/bin/env bash
# One-command demo of Kiln. Run: ./demo.sh   (or: bash demo.sh)
#
# What it does, in order:
#   1. Builds the C++ engine and the Python extension
#   2. Runs the C++ tests
#   3. Runs the Python tests
#   4. Runs the CPU benchmarks (KV cache speedup, latency, memory, INT8)
#   5. Starts the web API, asks it for text, then stops it
#   6. Compares Kiln's output to Hugging Face, and runs the GPU benchmark if
#      this machine has a CUDA build
#
# Optional settings:
#   KILN_CHECKPOINT_DIR=<dir>  use this SmolLM2 checkpoint (a cached
#                              SmolLM2-135M-Instruct is found automatically).
#                              Without one, the API serves a random-weight
#                              toy model: real request shapes, meaningless words.
#   KILN_DEVICE=cuda           serve on the GPU (needs a CUDA build)
#   PORT=<number>              web API port (default: any free port)
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="$(mktemp -d)"
SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true' EXIT
trap 'echo; echo "demo.sh stopped at line $LINENO. Logs are in $LOG_DIR"' ERR

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

# Use the project's virtualenv if there is one.
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export PYTHONPATH="$PWD"

for tool in cmake ninja; do
  command -v "$tool" >/dev/null || {
    echo "Missing '$tool'. On macOS: brew install cmake ninja nlohmann-json icu4c pybind11"
    exit 1
  }
done

step "1/6  Build the C++ engine and the Python extension"
note "Configuring (a few seconds)..."
cmake -B build -G Ninja -Wno-dev -DPython3_EXECUTABLE="$("$PY" -c 'import sys; print(sys.executable)')" \
  >"$LOG_DIR/cmake-configure.log" 2>&1 || { cat "$LOG_DIR/cmake-configure.log"; exit 1; }
note "Compiling (instant if nothing changed)..."
cmake --build build 2>&1 | tail -1
# Only install Python packages if something is missing; this is the slow step.
if ! "$PY" -c "import fastapi, uvicorn, numpy, pytest, httpx" 2>/dev/null; then
  note "Installing Python dependencies (about 30 seconds)..."
  "$PY" -m pip install -q -e ".[dev]" >"$LOG_DIR/pip.log" 2>&1 || { cat "$LOG_DIR/pip.log"; exit 1; }
fi
"$PY" -c "from kiln_py import _C" || {
  echo "The C++ extension did not build for this Python ($PY)."
  exit 1
}
note "Build OK."

step "2/6  C++ tests (tokenizer, forward pass, KV cache, paged cache, INT8, LoRA)"
ctest --test-dir build --output-on-failure | tail -4

step "3/6  Python tests (API, CUDA runtime, LoRA binding, quantization)"
"$PY" -m pytest tests/py -q 2>&1 | tail -2

step "4/6  Benchmarks: KV cache vs no cache, latency, memory, INT8"
"$PY" bench/run_benchmarks.py

step "5/6  Serve over HTTP and ask it for text"
# Use a cached SmolLM2 checkpoint if one exists and none was given.
if [ -z "${KILN_CHECKPOINT_DIR:-}" ]; then
  for d in "$HOME"/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/; do
    [ -f "${d}model.safetensors" ] && export KILN_CHECKPOINT_DIR="${d%/}"
  done
fi
if [ -n "${KILN_CHECKPOINT_DIR:-}" ]; then
  note "Real checkpoint: $(basename "$(dirname "$(dirname "$KILN_CHECKPOINT_DIR")")") (device: ${KILN_DEVICE:-cpu})"
else
  note "No checkpoint found: using the small random-weight model."
  note "The request and response shapes are real; the words are not."
  note "For real text, set KILN_CHECKPOINT_DIR to a SmolLM2 folder."
fi

# Pick a free port so something else (Docker, another server) can't block us.
if [ -z "${PORT:-}" ]; then
  PORT="$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
fi
URL="http://127.0.0.1:$PORT"

note "Starting the server on port $PORT..."
"$PY" -m uvicorn kiln_py.api.app:app --host 127.0.0.1 --port "$PORT" --log-level warning \
  >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait until the server answers, but stop early if it crashed, and never
# wait silently for long.
ready=""
for i in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "The server exited early. Its log:"; cat "$LOG_DIR/server.log"; exit 1
  fi
  if curl -sf -m 2 "$URL/healthz" >/dev/null 2>&1; then ready=yes; break; fi
  [ $((i % 4)) -eq 0 ] && note "Loading the model... ($((i / 2))s)"
  sleep 0.5
done
[ -n "$ready" ] || { echo "The server did not start in 30s. Its log:"; cat "$LOG_DIR/server.log"; exit 1; }
note "Server is up."

echo; echo "GET /healthz"
curl -s -m 10 "$URL/healthz"; echo
echo; echo 'POST /v1/completions   prompt: "The capital of France is"'
curl -s -m 60 "$URL/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 20, "temperature": 0.0}'; echo
echo; echo 'POST /v1/completions   (streamed, one word at a time)'
curl -sN -m 60 "$URL/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 5, "stream": true, "temperature": 0.0}'
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

step "6/6  Correctness and GPU"
if [ -n "${KILN_CHECKPOINT_DIR:-}" ] && "$PY" -c "import torch, transformers" 2>/dev/null; then
  note "Comparing Kiln's scores to Hugging Face on 10 prompts..."
  "$PY" tools/hf_parity.py --model-dir "$KILN_CHECKPOINT_DIR" \
    --prompts-file tools/fixtures/hf_parity_prompts.txt 2>/dev/null | tail -1
else
  note "Skipped the Hugging Face comparison. It needs a checkpoint and: pip install -e '.[oracle]'"
fi
if [ -x build/kiln_int8_gemm_cuda_benchmark ]; then
  echo; note "INT8 vs FP32 matrix multiply on this GPU:"
  ./build/kiln_int8_gemm_cuda_benchmark
else
  note "Skipped the GPU benchmark. It needs a CUDA build:"
  note "  cmake -B build -G Ninja -DKILN_BUILD_CUDA=ON -DKILN_CUDA_ARCHITECTURES=<arch>"
fi

printf '\n\033[1mDone.\033[0m\n'
