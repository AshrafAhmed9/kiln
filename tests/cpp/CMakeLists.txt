include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(kiln_cpp_tests
  memory/arena_test.cpp
  loader/safetensors_test.cpp
  tokenizer/bpe_test.cpp
  executor/gemm_test.cpp
  executor/rmsnorm_test.cpp
  executor/rope_test.cpp
  executor/attention_test.cpp
  executor/model_test.cpp
  kv/kv_cache_test.cpp
  kv/paged_kv_cache_test.cpp
  kv/prefix_cache_measurement_test.cpp
  executor/paged_attention_test.cpp
  executor/sampler_test.cpp
  executor/batch_test.cpp
  quant/quantize_test.cpp
  executor/parity_harness_test.cpp
  executor/lora_test.cpp
)
target_link_libraries(kiln_cpp_tests PRIVATE kiln_cpp gtest_main)

include(GoogleTest)
gtest_discover_tests(kiln_cpp_tests)

if(KILN_BUILD_CUDA)
  add_executable(kiln_cuda_tests cuda_kernels_test.cu)
  target_link_libraries(kiln_cuda_tests PRIVATE kiln_cpp kiln_cuda gtest_main)
  set_target_properties(kiln_cuda_tests PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES ${KILN_CUDA_ARCHITECTURES}
  )
  gtest_discover_tests(kiln_cuda_tests)
endif()
