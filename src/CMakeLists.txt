if (CPUONLY)
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

  project(leapct_cpu CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(OpenMP REQUIRED)
  add_compile_options(-D__USE_CPU)
else()
  # cmake version 3.23 or higher is needed to support 
  # the argument CUDA_ARCHITECTURES all-major
  # You could use cmake as old as version 3.18 if you replaced
  # the line below that specifies all-major with the line below
  # it which specifies the list of CUDA architectures 
  #cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
  cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

  project(leapct CXX CUDA)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(CUDA 11.7 REQUIRED)
  find_package(OpenMP REQUIRED)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D__INCLUDE_CUFFT)
  add_compile_options(-D__INCLUDE_CUFFT)

  # Software texture emulation for GPUs/APUs that lack texture memory (e.g. AMD
  # MI300A). Auto-enabled when a ROCm/HIP toolchain is detected so NVIDIA builds
  # keep hardware textures untouched; override explicitly with
  #   cmake -DLEAP_USE_NOTEX=ON ..   (or =OFF).
  set(_leap_notex_default OFF)
  if(DEFINED ENV{ROCM_PATH} OR DEFINED ENV{HIP_PATH} OR DEFINED ENV{HIP_PLATFORM})
    set(_leap_notex_default ON)
  endif()
  option(LEAP_USE_NOTEX "Emulate texture fetches in global memory (no texture-memory hardware)" ${_leap_notex_default})
  if(LEAP_USE_NOTEX)
    message(STATUS "LEAP: software texture emulation enabled (-D__USE_NOTEX)")
    add_compile_options(-D__USE_NOTEX)
  else()
    message(STATUS "LEAP: using hardware texture memory")
  endif()
endif()

#add_compile_options(-mavx -mavx2 -mfma -msse3)
add_compile_options(-O3)
#add_compile_options(-O3 -march=native)

if (CPUONLY)
  if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "lib_cpu")
    set(CMAKE_STATIC_LIBRARY_PREFIX "lib_cpu")
  endif()
else()
  if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
    set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
  endif()
endif()


if(CPUONLY)
  set(HEADER
    tomographic_models.h
    list_of_tomographic_models.h
    tomographic_models_c_interface.h
    parameters.h
    projectors/projectors.h
    fbp/filtered_backprojection.h
    projectors/projectors_SF_cpu.h
    projectors/projectors_Joseph_cpu.h
    projectors/projectors_symmetric_cpu.h
    projectors/projectors_Siddon_cpu.h
    projectors/sensitivity_cpu.h
    fbp/ramp_filter_cpu.h
    fbp/ray_weighting_cpu.h
    geometry/geometric_calibration_cpu.h
    geometry/find_center_cpu.h
    sinogram_replacement.h
    resample_cpu.h
    cuda_utils.h
    cpu_utils.h
    ray_tracing/phantom.h
    ray_tracing/analytic_ray_tracing.h
    geometry/rebin.h
    ring_removal.h
    inpainting.h
    segmentation.h
    statistics.h
    maximally_flat_filter.h
    finite_difference_filters.h
    file_io.h
    leap_defines.h
    log.h
    fbp/pocketfft_hdronly.h
    physics/xrayphysics.h
    physics/xsec.h
    physics/xscatter.h
    physics/xrayphysics_c_interface.h
    physics/xsec_raw.h
    physics/xscatter_raw.h
    physics/xsource.h
    physics/dual_energy_decomposition.h
  )
else()
  set(HEADER
    tomographic_models.h
    list_of_tomographic_models.h
    tomographic_models_c_interface.h
    parameters.h
    projectors/projectors.h
    fbp/filtered_backprojection.h
    projectors/projectors_SF.cuh
    projectors/projectors_SF_cpu.h
    projectors/projectors_extendedSF.cuh
    projectors/projectors_Joseph.cuh
    projectors/projectors_Joseph_cpu.h
    projectors/projectors_symmetric.cuh
    projectors/projectors_symmetric_cpu.h
    projectors/projectors_attenuated.cuh
    projectors/projectors_Siddon.cuh
    projectors/projectors_Siddon_cpu.h
    projectors/backprojectors_VD.cuh
    projectors/sensitivity_cpu.h
    projectors/sensitivity.cuh
    fbp/ramp_filter.cuh
    fbp/ramp_filter_cpu.h
    fbp/ray_weighting.cuh
    fbp/ray_weighting_cpu.h
    scatter_models.cuh
    denoise/noise_filters.cuh
    denoise/total_variation.cuh
    denoise/matching_pursuit.cuh
    denoise/bilateral_filter.cuh
    denoise/guided_filter.cuh
    geometry/geometric_calibration.cuh
    geometry/geometric_calibration_cpu.h
    geometry/find_center_cpu.h
    sinogram_replacement.h
    resample_cpu.h
    resample.cuh
    cuda_utils.h
    cpu_utils.h
    ray_tracing/phantom.h
    ray_tracing/analytic_ray_tracing.h
    ray_tracing/analytic_ray_tracing_gpu.cuh
    geometry/rebin.h
    ring_removal.h
    inpainting.h
    segmentation.h
    statistics.h
    maximally_flat_filter.h
    finite_difference_filters.h
    file_io.h
    leap_defines.h
    log.h
    fbp/pocketfft_hdronly.h
    physics/xrayphysics.h
    physics/xsec.h
    physics/xscatter.h
    physics/xrayphysics_c_interface.h
    physics/xsec_raw.h
    physics/xscatter_raw.h
    physics/xsource.h
    physics/dual_energy_decomposition.h
    physics/billiards.cuh
  )

  set(SRC_CU
    projectors/projectors_SF.cu
    projectors/projectors_extendedSF.cu
    projectors/projectors_Joseph.cu
    projectors/projectors_symmetric.cu
    projectors/projectors_attenuated.cu
    projectors/projectors_Siddon.cu
    projectors/backprojectors_VD.cu
    projectors/sensitivity.cu
    fbp/ramp_filter.cu
    resample.cu
    denoise/noise_filters.cu
    denoise/total_variation.cu
    denoise/matching_pursuit.cu
    denoise/bilateral_filter.cu
    denoise/guided_filter.cu
    geometry/geometric_calibration.cu
    fbp/ray_weighting.cu
    scatter_models.cu
    ray_tracing/analytic_ray_tracing_gpu.cu
    cuda_utils.cu
    physics/billiards.cu
  )
endif()

set(SRC_CPP
  tomographic_models.cpp
  list_of_tomographic_models.cpp
  tomographic_models_c_interface.cpp
  parameters.cpp
  projectors/projectors.cpp
  fbp/filtered_backprojection.cpp
  projectors/projectors_SF_cpu.cpp
  projectors/projectors_Joseph_cpu.cpp
  projectors/projectors_symmetric_cpu.cpp
  projectors/projectors_Siddon_cpu.cpp
  projectors/sensitivity_cpu.cpp
  fbp/ramp_filter_cpu.cpp
  fbp/ray_weighting_cpu.cpp
  geometry/find_center_cpu.cpp
  geometry/geometric_calibration_cpu.cpp
  sinogram_replacement.cpp
  resample_cpu.cpp
  cpu_utils.cpp
  ray_tracing/phantom.cpp
  ray_tracing/analytic_ray_tracing.cpp
  geometry/rebin.cpp
  ring_removal.cpp
  inpainting.cpp
  segmentation.cpp
  statistics.cpp
  maximally_flat_filter.cpp
  finite_difference_filters.cpp
  file_io.cpp
  physics/xrayphysics.cpp
  physics/xsec.cpp
  physics/xscatter.cpp
  physics/xrayphysics_c_interface.cpp
  physics/xsec_raw.cpp
  physics/xscatter_raw.cpp
  physics/xsource.cpp
  physics/dual_energy_decomposition.cpp
)

if (CPUONLY)
  include_directories(
    ./
  )

  add_library(leapct_cpu SHARED
    ${HEADER}
    ${SRC_CPP}
  )
else()
  include_directories(
    ./
    ${CUDA_INCLUDE_DIRS}
  )

  add_library(leapct SHARED
    ${HEADER}
    ${SRC_CPP}
    ${SRC_CU}
  )
endif()

# <<<--- Create filters for subfolders for Visual Studio
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (CPUONLY)
  file(GLOB_RECURSE _source_list
    LIST_DIRECTORIES false
    ${HEADER}
    ${SRC_CPP}
  )
else()
  file(GLOB_RECURSE _source_list
    LIST_DIRECTORIES false
    ${HEADER}
    ${SRC_CPP}
    ${SRC_CU}
  )
endif()

foreach(_source IN ITEMS ${_source_list})
  get_filename_component(_source_path "${_source}" PATH)
  file(RELATIVE_PATH _source_path_rel "${PROJECT_SOURCE_DIR}" "${_source_path}")
  string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
  source_group("${_group_path}" FILES "${_source}")
endforeach()
# --->>>

if (CPUONLY)
  target_link_libraries(${PROJECT_NAME}
    OpenMP::OpenMP_CXX
  )

  install(TARGETS ${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION include
    RUNTIME DESTINATION bin_cpu
    ARCHIVE DESTINATION lib_cpu
    LIBRARY DESTINATION lib_cpu
  )
else()
  target_link_libraries(${PROJECT_NAME}
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cufft_LIBRARY}
  )

  target_compile_options(leapct PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
  >)

  #set_property(TARGET leapct PROPERTY CXX_STANDARD 14)
  #set_property(TARGET leapct PROPERTY CUDA_ARCHITECTURES native)
  if(CMAKE_VERSION VERSION_GREATER "3.23")
      message("Building for all major cuda architectures")
      set_property(TARGET leapct PROPERTY CUDA_ARCHITECTURES all-major)
  endif()
  #set_property(TARGET leapct PROPERTY CUDA_ARCHITECTURES all-major)
  #set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 87 89)
  #set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 86 87 89)
  #set_property(TARGET leapct PROPERTY CUDA_ARCHITECTURES OFF)

  install(TARGETS ${PROJECT_NAME}
    PUBLIC_HEADER DESTINATION include
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )
endif()
