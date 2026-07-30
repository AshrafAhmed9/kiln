cmake_minimum_required(VERSION 3.20)
project(kiln LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KILN_SANITIZE "Build with ASan+UBSan" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)
if(KILN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(kiln_memory src/memory/arena.cpp)
target_include_directories(kiln_memory PUBLIC src)

enable_testing()
add_subdirectory(tests)
