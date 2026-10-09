cmake_minimum_required(VERSION 3.20)
project(othello_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(WARNING_FLAGS -Wall -Wextra -Wpedantic)

# Libraries ---------------------------------------------------------------

# Rules, search and random numbers are header-only.
add_library(othello_core INTERFACE)
target_include_directories(othello_core INTERFACE include)

# The network. Only the vectorized kernels are compiled with AVX2, FMA and
# F16C, and the network chooses between them and the plain loops at run time.
add_library(othello_net STATIC
  src/net.cpp
  src/net_reference.cpp
  src/net_avx2.cpp
)
target_link_libraries(othello_net PUBLIC othello_core)
target_compile_options(othello_net PRIVATE ${WARNING_FLAGS})
set_target_properties(othello_net PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(src/net_avx2.cpp
  PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
)

# Tests -------------------------------------------------------------------

enable_testing()

add_executable(position_test tests/position_test.cpp)
target_link_libraries(position_test PRIVATE othello_core)
target_compile_options(position_test PRIVATE ${WARNING_FLAGS})
add_test(NAME position_test COMMAND position_test)

add_executable(mcts_test tests/mcts_test.cpp)
target_link_libraries(mcts_test PRIVATE othello_net)
target_compile_options(mcts_test PRIVATE ${WARNING_FLAGS})
add_test(NAME mcts_test COMMAND mcts_test)

# Benchmarks --------------------------------------------------------------

add_executable(bench_net bench/bench_net.cpp)
target_link_libraries(bench_net PRIVATE othello_net)
target_compile_options(bench_net PRIVATE ${WARNING_FLAGS})

add_executable(bench_mcts bench/bench_mcts.cpp)
target_link_libraries(bench_mcts PRIVATE othello_net)
target_compile_options(bench_mcts PRIVATE ${WARNING_FLAGS})

add_executable(bench_rules bench/bench_rules.cpp)
target_link_libraries(bench_rules PRIVATE othello_core)
target_compile_options(bench_rules PRIVATE ${WARNING_FLAGS})

add_executable(bench_peaks bench/bench_peaks.cpp)
target_compile_options(bench_peaks PRIVATE ${WARNING_FLAGS} -mavx2 -mfma)

# Python module -----------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Ask the interpreter where its pybind11 lives.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(othello_cpp bindings/module.cpp)
target_link_libraries(othello_cpp PRIVATE othello_core othello_net)
set_target_properties(othello_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# GCC 12 reports a false positive for pybind11 string arguments during
# compilation and link-time optimization.
target_compile_options(othello_cpp PRIVATE ${WARNING_FLAGS} -Wno-stringop-overread)
target_link_options(othello_cpp PRIVATE -Wno-stringop-overread)
