#!/usr/bin/env bash
# Run Kiln's controlled GPU validation on a short-lived GCP VM.
#
# This script intentionally refuses to create a billable resource until its
# caller explicitly acknowledges that decision. It deletes the exact instance
# it creates when the validation command returns, unless KEEP_INSTANCE=YES is
# set for interactive debugging.
set -euo pipefail

: "${KILN_GCP_ALLOW_BILLING:?Set KILN_GCP_ALLOW_BILLING=YES after choosing a cost cap.}"
if [[ "${KILN_GCP_ALLOW_BILLING}" != "YES" ]]; then
  echo "KILN_GCP_ALLOW_BILLING must be YES" >&2
  exit 2
fi

: "${KILN_GCP_PROJECT:?Set the GCP project id.}"
: "${KILN_GCP_ZONE:?Set a zone with available GPU quota.}"
: "${KILN_GCP_IMAGE_FAMILY:?Set a CUDA-enabled Deep Learning VM image family.}"

INSTANCE="${KILN_GCP_INSTANCE:-kiln-gpu-validation-$(date +%Y%m%d%H%M%S)}"
MACHINE_TYPE="${KILN_GCP_MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${KILN_GCP_GPU_TYPE:-nvidia-tesla-t4}"
MAX_MINUTES="${KILN_GCP_MAX_MINUTES:-45}"

cleanup() {
  if [[ "${KILN_GCP_KEEP_INSTANCE:-NO}" == "YES" ]]; then
    echo "Keeping ${INSTANCE} for debugging. Delete it manually when finished."
    return
  fi
  gcloud compute instances delete "${INSTANCE}" --project="${KILN_GCP_PROJECT}" \
    --zone="${KILN_GCP_ZONE}" --quiet || true
}
trap cleanup EXIT

gcloud compute instances create "${INSTANCE}" \
  --project="${KILN_GCP_PROJECT}" \
  --zone="${KILN_GCP_ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=1" \
  --maintenance-policy=TERMINATE \
  --image-family="${KILN_GCP_IMAGE_FAMILY}" \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --labels=purpose=kiln-gpu-validation \
  --quiet

read -r -d '' REMOTE_COMMAND <<'REMOTE' || true
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq cmake g++ git libicu-dev nlohmann-json3-dev ninja-build
git clone --depth 1 https://github.com/AshrafAhmed9/kiln.git kiln
cd kiln
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S . -B build-gpu -G Ninja \
  -DKILN_BUILD_CUDA=ON -DKILN_BUILD_PYBIND=ON \
  -DKILN_CUDA_ARCHITECTURES="${ARCH}"
cmake --build build-gpu
ctest --test-dir build-gpu --output-on-failure
python3 tools/validate_cuda_binding.py
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build-gpu/tests/cpp/kiln_cuda_tests \
  --gtest_filter=CudaModel.FullPrefillMatchesCpuReference
ncu --set basic --target-processes all \
  ./build-gpu/tests/cpp/kiln_cuda_tests \
  --gtest_filter=CudaModel.FullPrefillMatchesCpuReference
REMOTE

gcloud compute ssh "${INSTANCE}" --project="${KILN_GCP_PROJECT}" \
  --zone="${KILN_GCP_ZONE}" --command="timeout ${MAX_MINUTES}m bash -lc $(printf '%q' "${REMOTE_COMMAND}")"
