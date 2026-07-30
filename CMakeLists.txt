cmake_minimum_required(VERSION 3.20)
project(kiln LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KILN_SANITIZE "Build with ASan+UBSan" OFF)
option(KILN_BUILD_PYBIND "Build the kiln._C pybind11 extension" ON)

add_compile_options(-Wall -Wextra -Wpedantic)
if(KILN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# The compute layer (csrc/) -- constitution §6: everything below the
# pybind11 boundary. Kept as its own static lib so tests link it directly,
# without going through Python.
add_library(kiln_cpp STATIC
  csrc/memory/arena.cpp
)
target_include_directories(kiln_cpp PUBLIC csrc)

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
  set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/kiln_py
  )
endif()

enable_testing()
add_subdirectory(tests/cpp)
