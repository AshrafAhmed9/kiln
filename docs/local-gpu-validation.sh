#!/bin/bash
# Run this on a machine with a real NVIDIA GPU (WSL2 Ubuntu, or native Linux)
# to capture the two numbers this project is still missing: Nsight Compute
# profiling output, and a real GPU speed benchmark for the INT8 GEMM path.
# Nothing here needs GPU-model info in advance -- compute capability is
# auto-detected from nvidia-smi, the same way the GCP attempts were driven.
#
# Prereqs this script assumes are already true:
#   - NVIDIA driver installed and `nvidia-smi` works
#   - git, sudo access
#
# Usage: bash local-gpu-validation.sh

set -euo pipefail

echo "=== GPU CHECK ==="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found or GPU not visible. On WSL2: install the NVIDIA"
  echo "driver on the WINDOWS side (not inside WSL) -- WSL2 GPU passthrough"
  echo "uses the Windows host driver. Then relaunch WSL and retry."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "COMPUTE_CAP=${ARCH}"

echo "=== TOOLCHAIN ==="
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build nlohmann-json3-dev libicu-dev git nvidia-cuda-toolkit

# CUDA's nvcc can be picky about which GCC it accepts as a host compiler --
# this bit it us twice on GCP (nvcc 11.5 rejected both GCC 12 and the
# distro's patched GCC 11.4 with a std_function.h parse error). If the
# default build below fails with a similar "parameter packs not expanded"
# error, install an older g++ (e.g. `sudo apt-get install -y g++-10`) and
# add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10` to the cmake call.

echo "=== REPO ==="
if [ ! -d ~/kiln ]; then
  git clone --depth 1 https://github.com/AshrafAhmed9/kiln.git ~/kiln
fi
cd ~/kiln
git pull --ff-only || true

echo "=== CONFIGURE ==="
rm -rf build-gpu
cmake -S . -B build-gpu -G Ninja \
  -DKILN_BUILD_CUDA=ON -DKILN_BUILD_PYBIND=OFF \
  -DKILN_CUDA_ARCHITECTURES="${ARCH}" \
  -DCMAKE_BUILD_TYPE=Release

echo "=== BUILD ==="
cmake --build build-gpu -j"$(nproc)" 2>&1 | tee /tmp/build.log
if grep -q FAILED /tmp/build.log; then
  echo "BUILD_FAILED -- see /tmp/build.log. If the error mentions"
  echo "std_function.h / 'parameter packs not expanded', see the nvcc/GCC"
  echo "note above this script's REPO section."
  exit 1
fi
echo "BUILD_OK"

echo "=== CTEST ==="
ctest --test-dir build-gpu --output-on-failure

echo "=== INT8 GEMM BENCHMARK (the real speed number this project needs) ==="
./build-gpu/kiln_int8_gemm_cuda_benchmark

echo "=== NSIGHT COMPUTE (the profiling output this project needs) ==="
NCU_BIN=$(find / -iname "ncu" -type f 2>/dev/null | head -1)
if [ -z "${NCU_BIN:-}" ]; then
  sudo apt-get install -y -qq nvidia-nsight-compute 2>&1 | tail -5 || true
  NCU_BIN=$(find / -iname "ncu" -type f 2>/dev/null | head -1)
fi
if [ -n "${NCU_BIN:-}" ]; then
  sudo "$NCU_BIN" --set basic --target-processes all \
    ./build-gpu/tests/cpp/kiln_cuda_tests \
    --gtest_filter=CudaModel.FullPrefillMatchesCpuReference \
    | tee /tmp/ncu_output.txt
else
  echo "ncu not found -- install NVIDIA Nsight Compute for your platform"
  echo "(on WSL2: install it from the Windows side, it can profile into WSL)"
fi

echo "=== DONE -- paste the INT8 GEMM and Nsight output back for the docs ==="
