include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

add_executable(kiln_tests memory/arena_test.cpp)
target_link_libraries(kiln_tests PRIVATE kiln_memory gtest_main)
target_include_directories(kiln_tests PRIVATE ../src)

include(GoogleTest)
gtest_discover_tests(kiln_tests)
