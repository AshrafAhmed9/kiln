#!/usr/bin/env bash
# The five-minute demo: cold clone -> build -> serve a (toy, untrained)
# model -> hit the OpenAI-compatible API, including streaming -> run the
# eval-gate machinery on a tiny synthetic task suite. Honest up front: the
# model this serves is randomly initialized, not a real trained checkpoint
# (see README.md and docs/walkthrough.md) -- this script demonstrates the
# ENGINE working end to end, not that the model's answers are meaningful.
set -euo pipefail

echo "== 1. Build the C++ compute layer and run its test suite =="
cmake -B build -G Ninja -DKILN_BUILD_PYBIND=OFF
cmake --build build
ctest --test-dir build --output-on-failure

echo
echo "== 2. Build the pybind11 extension (the Python <-> C++ boundary) =="
cmake -B build-py -G Ninja -DKILN_BUILD_PYBIND=ON \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build-py --target _C

echo
echo "== 3. Run the Python test suite (scheduler, API, eval infra, etc.) =="
PYTHONPATH=. python3 -m pytest tests/py -q

echo
echo "== 4. Start the API server in the background =="
PYTHONPATH=. python3 -m uvicorn kiln_py.api.app:app --port 8420 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 2

echo
echo "== 5. A non-streaming completion =="
curl -s http://127.0.0.1:8420/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello, kiln", "max_tokens": 8}' | python3 -m json.tool

echo
echo "== 6. A streaming completion (words arrive one at a time) =="
curl -s -N http://127.0.0.1:8420/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "streaming demo", "max_tokens": 5, "stream": true}'

echo
echo "== 7. Eval infrastructure: a tiny synthetic regression check =="
PYTHONPATH=. python3 -c "
from kiln_py.eval.regression_gate import check_for_regression
baseline = [0.9, 0.9, 0.9, 0.9, 0.9]
candidate = [0.5, 0.5, 0.5, 0.5, 0.5]
result = check_for_regression(baseline, candidate)
print('regression detected:', result.is_regression)
print('mean difference:', result.mean_difference)
print('95% confidence interval:', result.confidence_interval)
"

echo
echo "== Demo complete. See BENCHMARKS.md and docs/walkthrough.md for the =="
echo "== full, itemized list of what's verified vs. deferred to GPU time. =="
