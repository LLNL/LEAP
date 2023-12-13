cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(leap CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA 11.7 REQUIRED)
find_package(OpenMP REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
endif()

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

set(HEADER
  tomographic_models.h
  tomographic_models_c_interface.h
  parameters.h
  projectors.h
  filtered_backprojection.h
  projectors_SF.cuh
  projectors_SF_cpu.h
  projectors_extendedSF.cuh
  projectors_Joseph.cuh
  projectors_Joseph_cpu.h
  projectors_symmetric.cuh
  projectors_symmetric_cpu.h
  projectors_attenuated.cuh
  projectors_Siddon.cuh
  projectors_Siddon_cpu.h
  sensitivity_cpu.h
  sensitivity.cuh
  ramp_filter.cuh
  ramp_filter_cpu.h
  ray_weighting.cuh
  ray_weighting_cpu.h
  noise_filters.cuh
  total_variation.cuh
  cuda_utils.h
  cpu_utils.h
  phantom.h
  leap_defines.h
)

set(SRC_CPP
  tomographic_models.cpp
  tomographic_models_c_interface.cpp
  parameters.cpp
  projectors.cpp
  filtered_backprojection.cpp
  projectors_SF_cpu.cpp
  projectors_Joseph_cpu.cpp
  projectors_symmetric_cpu.cpp
  projectors_Siddon_cpu.cpp
  sensitivity_cpu.cpp
  ramp_filter_cpu.cpp
  ray_weighting_cpu.cpp
  cpu_utils.cpp
  phantom.cpp
)

set(SRC_CU
  projectors_SF.cu
  projectors_extendedSF.cu
  projectors_Joseph.cu
  projectors_symmetric.cu
  projectors_attenuated.cu
  projectors_Siddon.cu
  sensitivity.cu
  ramp_filter.cu
  noise_filters.cu
  total_variation.cu
  ray_weighting.cu
  cuda_utils.cu
)

include_directories(
  ./
  ${CUDA_INCLUDE_DIRS}
)

#add_library(leap SHARED main_projector_ctype.cpp parameters.cpp projectors_cpu.cpp projectors.cu projectors_SF.cu)
#target_link_libraries(leap "${TORCH_LIBRARIES}")
add_library(leap SHARED
  ${HEADER}
  ${SRC_CPP}
  ${SRC_CU}
)

# <<<--- Create filters for subfolders for Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(GLOB_RECURSE _source_list
  LIST_DIRECTORIES false
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
  OpenMP::OpenMP_CXX
  ${CUDA_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cufft_LIBRARY}
)

target_compile_options(leap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

#set_property(TARGET leap PROPERTY CXX_STANDARD 14)
set_property(TARGET leap PROPERTY CUDA_ARCHITECTURES native)
#set_property(TARGET leap PROPERTY CUDA_ARCHITECTURES OFF)

install(TARGETS ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION include
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
