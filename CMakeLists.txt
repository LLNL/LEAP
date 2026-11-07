# cmake version 3.23 or higher is needed to support
# the argument CUDA_ARCHITECTURES all-major
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(leapct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(HIP)
check_language(CUDA)

if(NOT DEFINED LEAP_GPU)
    if(CMAKE_HIP_COMPILER AND CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Both HIP and CUDA compilers were detected. Specify LEAP_GPU explicitly as AMD, NVIDIA, or None.")
    elseif(CMAKE_CUDA_COMPILER)
        set(LEAP_GPU "NVIDIA")
    elseif(CMAKE_HIP_COMPILER)
        set(LEAP_GPU "AMD")
    else()
        set(LEAP_GPU "None")
    endif()
endif()

set(LEAP_GPU "${LEAP_GPU}" CACHE STRING "GPU acceleration type" FORCE)
set_property(CACHE LEAP_GPU PROPERTY STRINGS "NVIDIA" "AMD" "None")
string(TOUPPER "${LEAP_GPU}" LEAP_GPU)

set(_leap_gpu_options "NVIDIA" "AMD" "NONE")
if (NOT LEAP_GPU IN_LIST _leap_gpu_options)
  message(FATAL_ERROR "Wrong value of the parameter 'LEAP_GPU': ${LEAP_GPU}. Expected NVIDIA, AMD, or None.")
endif ()
message(STATUS "LEAP_GPU selected accelerator type ${LEAP_GPU}")

if (LEAP_GPU STREQUAL "NVIDIA")
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "LEAP_GPU=NVIDIA requires a CUDA compiler")
    endif()
    # To minimize binary size and compile time, users are suggested to
    # set the CUDAARCHS environment variable or define the CMake variable CMAKE_CUDA_ARCHITECTURES
    # to "native" if building specifically for the GPU visible to the build environment,
    # or to an explicit numerical architecture code, such as "70"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES all-major)
    endif()
    enable_language(CUDA)
    set(LEAP_CUDA ON)
elseif (LEAP_GPU STREQUAL "AMD")
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "LEAP_GPU=AMD requires a HIP compiler")
    endif()
    # Users are advised to set CMAKE_HIP_ARCHITECTURES to a minimal subset.
    # See also https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html#using-hip-in-cmake
    enable_language(HIP)
    set(LEAP_HIP ON)
    set(LEAP_ORIGINAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(LEAP_HIPIFIED_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipified_src")
    include(cmake/LeapHipify.cmake)
    leap_hipify_sources()
elseif (LEAP_GPU STREQUAL "NONE")
    set(LEAP_CPU_ONLY ON)
else ()
    message(FATAL_ERROR "CMake scripting error: ${LEAP_GPU} didn't match.")
endif ()

if((BUILD_TESTING) OR (NOT DEFINED BUILD_TESTING))
    ENABLE_TESTING()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
