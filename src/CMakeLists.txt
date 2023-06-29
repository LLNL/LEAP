cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(leap LANGUAGES CXX CUDA)
enable_language(CUDA)

#find_package(CUDA QUIET REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

add_library(leap SHARED main_projector_ctype.cpp parameters.cpp projectors.cu)
#target_link_libraries(leap "${TORCH_LIBRARIES}")

set_property(TARGET leap PROPERTY CXX_STANDARD 14)
#set_property(TARGET leap PROPERTY CUDA_ARCHITECTURES OFF)

