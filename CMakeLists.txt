cmake_minimum_required(VERSION 3.20)
project(kiln LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KILN_SANITIZE "Build with ASan+UBSan" OFF)
option(KILN_BUILD_PYBIND "Build the kiln._C pybind11 extension" ON)
option(KILN_BUILD_CUDA "Build the CUDA kernel validation suite" OFF)
set(KILN_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA compute capability, for example 75 for a T4 or 60 for a P100")

add_compile_options(-Wall -Wextra -Wpedantic)
if(KILN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(nlohmann_json 3.10 REQUIRED)
find_package(ICU REQUIRED COMPONENTS i18n uc)

# The compute layer (csrc/) -- constitution §6: everything below the
# pybind11 boundary. Kept as its own static lib so tests link it directly,
# without going through Python.
add_library(kiln_cpp STATIC
  csrc/memory/arena.cpp
  csrc/loader/safetensors.cpp
  csrc/tokenizer/bpe.cpp
  csrc/executor/gemm.cpp
  csrc/executor/rmsnorm.cpp
  csrc/executor/rope.cpp
  csrc/executor/attention.cpp
  csrc/executor/swiglu.cpp
  csrc/executor/model.cpp
  csrc/kv/kv_cache.cpp
  csrc/kv/paged_kv_cache.cpp
  csrc/kv/prefix_cache_measurement.cpp
  csrc/executor/paged_attention.cpp
  csrc/executor/sampler.cpp
  csrc/executor/batch.cpp
  csrc/quant/quantize.cpp
  csrc/executor/lora.cpp
)
target_include_directories(kiln_cpp PUBLIC csrc)
target_link_libraries(kiln_cpp PUBLIC nlohmann_json::nlohmann_json ICU::i18n ICU::uc)

# This static library gets linked into the _C shared module (below) as
# well as into the plain test binary. A static library built without
# position-independent code links fine into a plain executable, but fails
# on Linux when linked into a *shared* object -- which is exactly what
# _C is. This worked by coincidence on macOS (where PIC isn't enforced the
# same way) and only failed when actually building the Docker image on
# Linux for the first time -- a real, caught-late bug, not a hypothetical.
set_target_properties(kiln_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(KILN_BUILD_CUDA)
  if(NOT KILN_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "Set KILN_CUDA_ARCHITECTURES to the assigned GPU capability (for example 75)")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(kiln_cuda STATIC
    csrc/executor/cuda_model.cu
    csrc/kernels/cuda/attention.cu
    csrc/kernels/cuda/gemm.cu
    csrc/kernels/cuda/int8_gemm.cu
    csrc/kernels/cuda/elementwise.cu
    csrc/kernels/cuda/rmsnorm.cu
    csrc/kernels/cuda/rope.cu
    csrc/kernels/cuda/sampler.cu
  )
  target_include_directories(kiln_cuda PUBLIC csrc)
  target_link_libraries(kiln_cuda PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(kiln_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES ${KILN_CUDA_ARCHITECTURES}
  )

  # This is deliberately a separate executable instead of a Python binding:
  # it times the same raw CUDA RoPE entry point used by the GPU correctness
  # test, without adding a second language boundary to the measurement.
  add_executable(kiln_rope_cuda_benchmark bench/rope_cuda_benchmark.cu)
  target_link_libraries(kiln_rope_cuda_benchmark PRIVATE kiln_cuda CUDA::cudart)
  target_include_directories(kiln_rope_cuda_benchmark PRIVATE csrc)
  set_target_properties(kiln_rope_cuda_benchmark PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES ${KILN_CUDA_ARCHITECTURES}
  )

  add_executable(kiln_int8_gemm_cuda_benchmark bench/int8_gemm_cuda_benchmark.cu)
  target_link_libraries(kiln_int8_gemm_cuda_benchmark PRIVATE kiln_cuda CUDA::cudart CUDA::cublas)
  target_include_directories(kiln_int8_gemm_cuda_benchmark PRIVATE csrc)
  set_target_properties(kiln_int8_gemm_cuda_benchmark PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES ${KILN_CUDA_ARCHITECTURES}
  )
endif()

if(KILN_BUILD_PYBIND)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  # The one file allowed to know about both sides of the §6 boundary.
  pybind11_add_module(_C csrc/bindings.cpp)
  target_link_libraries(_C PRIVATE kiln_cpp)
  if(KILN_BUILD_CUDA)
    # Keep the public extension CPU-buildable while making the CUDA executor
    # available in a CUDA build through the same audited pybind boundary.
    target_link_libraries(_C PRIVATE kiln_cuda)
    target_compile_definitions(_C PRIVATE KILN_BUILD_CUDA=1)
  endif()
  set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/kiln_py
  )
endif()

enable_testing()
add_subdirectory(tests/cpp)

# This intentionally measures a seeded local workload, not traffic. Keeping
# it as a small C++ executable means it exercises the same paged-cache code
# that serves sequences instead of estimating sharing from a spreadsheet.
add_executable(kiln_prefix_cache_benchmark bench/prefix_cache_hit_rate.cpp)
target_link_libraries(kiln_prefix_cache_benchmark PRIVATE kiln_cpp)
