cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(leap LANGUAGES CXX CUDA)
enable_language(CUDA)

#find_package(CUDA QUIET REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
endif()

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

set(HEADER
  main_projector_ctype.h
  parameters.h
  projectors_cpu.h
  projectors.h
  projectors_SF.h
  cuda_utils.h
  rampFilter.cuh
  noise_filters.cuh
  total_variation.cuh
)

set(SRC_CPP
  main_projector_ctype.cpp
  parameters.cpp
  projectors_cpu.cpp
  cuda_utils.cpp
)

set(SRC_CU
  projectors.cu
  projectors_SF.cu
  rampFilter.cu
  noise_filters.cu
  total_variation.cu
)

include_directories(
  ./
  ${CUDA_INCLUDE_DIRS}
)

#add_library(leap SHARED main_projector_ctype.cpp parameters.cpp projectors_cpu.cpp projectors.cu projectors_SF.cu)
#target_link_libraries(leap "${TORCH_LIBRARIES}")
add_library(leap SHARED
  ${PUB_HEADER}
  ${HEADER}
  ${SRC_CPP}
  ${SRC_CU}
)

# <<<--- Create filters for subfolders for Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(GLOB_RECURSE _source_list
  LIST_DIRECTORIES false
  ${PUB_HEADER}
  ${HEADER}
  ${SRC_CPP}
  ${SRC_CU}
)

foreach(_source IN ITEMS ${_source_list})
  get_filename_component(_source_path "${_source}" PATH)
  file(RELATIVE_PATH _source_path_rel "${PROJECT_SOURCE_DIR}" "${_source_path}")
  string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
  source_group("${_group_path}" FILES "${_source}")
endforeach()
# --->>>

target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cufft_LIBRARY}
)

set_property(TARGET leap PROPERTY CXX_STANDARD 14)
#set_property(TARGET leap PROPERTY CUDA_ARCHITECTURES OFF)
