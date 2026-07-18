cmake_minimum_required(VERSION 3.18)
project(reefscan_cpp_server LANGUAGES CXX)   # CUDA language added in Phase 3 (preproc_kernel.cu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA runtime (cudart) — we call the runtime API directly, no nvcc needed until Phase 3.
find_package(CUDAToolkit REQUIRED)

# TensorRT 10.5 (matches tritonserver:24.10-py3). On the NGC image the headers/libs are on the
# default paths; otherwise pass -DTRT_ROOT=/path/to/TensorRT.
find_path(TRT_INCLUDE NvInfer.h
  HINTS ${TRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)
find_library(TRT_NVINFER nvinfer
  HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
if(NOT TRT_INCLUDE OR NOT TRT_NVINFER)
  message(FATAL_ERROR "TensorRT not found. Set -DTRT_ROOT=/path/to/TensorRT (need NvInfer.h + libnvinfer).")
endif()

add_library(trt_engine src/trt_engine.cpp)
target_include_directories(trt_engine PUBLIC include ${TRT_INCLUDE})
target_link_libraries(trt_engine PUBLIC ${TRT_NVINFER} CUDA::cudart)

# Phase 0 gate binary.
add_executable(reefscan_infer src/main_phase0.cpp)
target_link_libraries(reefscan_infer PRIVATE trt_engine)

# Phase 1: the dynamic-batching queue + its concurrency gate.
find_package(Threads REQUIRED)
add_library(batch_queue src/batch_queue.cpp)
target_link_libraries(batch_queue PUBLIC trt_engine Threads::Threads)

add_executable(reefscan_batch_test src/main_phase1.cpp)
target_link_libraries(reefscan_batch_test PRIVATE batch_queue)

# Phases 2-3 (server, kernels) register their targets here as they land.
