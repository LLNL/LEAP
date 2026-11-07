find_package(OpenMP REQUIRED COMPONENTS CXX)
if(LEAP_CUDA)
    find_package(CUDAToolkit 11.7 REQUIRED)
elseif(LEAP_HIP)
    find_package(hip REQUIRED)
    find_package(hipfft REQUIRED)
endif()

add_compile_options(-O3)

if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
endif()

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

set(GPU_HEADER
  projectors/projectors_SF.cuh
  projectors/projectors_extendedSF.cuh
  projectors/projectors_Joseph.cuh
  projectors/projectors_symmetric.cuh
  projectors/projectors_attenuated.cuh
  projectors/projectors_Siddon.cuh
  projectors/backprojectors_VD.cuh
  projectors/sensitivity.cuh
  fbp/ramp_filter.cuh
  fbp/ray_weighting.cuh
  scatter_models.cuh
  denoise/noise_filters.cuh
  denoise/total_variation.cuh
  denoise/matching_pursuit.cuh
  denoise/bilateral_filter.cuh
  denoise/guided_filter.cuh
  geometry/geometric_calibration.cuh
  resample.cuh
  cuda_utils.h
  ray_tracing/analytic_ray_tracing_gpu.cuh
  physics/billiards.cuh
)

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

if(LEAP_CUDA OR LEAP_HIP)
    list(APPEND HEADER ${GPU_HEADER})
endif()

if(LEAP_HIP)
    list(TRANSFORM SRC_CPP PREPEND "${LEAP_HIPIFIED_SRC_DIR}/")
    list(TRANSFORM HEADER PREPEND "${LEAP_HIPIFIED_SRC_DIR}/")
    list(TRANSFORM SRC_CU PREPEND "${LEAP_HIPIFIED_SRC_DIR}/")
    get_hipified_list("${SRC_CPP}" SRC_CPP)
    get_hipified_list("${HEADER}" HEADER)
    get_hipified_list("${SRC_CU}" SRC_CU)
    set_source_files_properties(${SRC_CPP} ${SRC_CU} PROPERTIES LANGUAGE HIP)
endif()

add_library(leapct SHARED
  ${HEADER}
  ${SRC_CPP}
)

if(LEAP_CUDA OR LEAP_HIP)
    target_sources(leapct PRIVATE ${SRC_CU})
endif()

set(_LEAP_INCLUDE_DIRS
  ./
  fbp
  projectors
  denoise
  geometry
  ray_tracing
  physics
)

if(LEAP_HIP)
    target_include_directories(leapct PRIVATE
      ${LEAP_HIPIFIED_SRC_DIR}
      ${_LEAP_INCLUDE_DIRS}
    )
else()
    target_include_directories(leapct PRIVATE
      ${_LEAP_INCLUDE_DIRS}
    )
endif()

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

if (LEAP_CUDA)
    target_compile_definitions(
        leapct PRIVATE
        __USE_GPU
        __INCLUDE_CUFFT
    )
    target_link_libraries(leapct
      OpenMP::OpenMP_CXX
      CUDA::cudart
      CUDA::cublas
      CUDA::cufft
    )
elseif (LEAP_HIP)
    target_compile_definitions(
        leapct PRIVATE
        __USE_GPU
        __USE_NOTEX
        __INCLUDE_CUFFT
    )
    get_filename_component(_ROCM_ROOT "${hipfft_DIR}/../../.." ABSOLUTE)
    target_include_directories(leapct PRIVATE
      ${_ROCM_ROOT}/include
      ${_ROCM_ROOT}/include/hipfft
    )
    target_link_libraries(leapct
      hip::host
      hip::hipfft
      OpenMP::OpenMP_CXX
    )
else ()
    target_compile_definitions(
        leapct PRIVATE
        __USE_CPU
    )
    target_link_libraries(leapct
      OpenMP::OpenMP_CXX
    )
endif()

target_compile_options(leapct PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

if(SKBUILD)
    install(TARGETS leapct
      DESTINATION ${SKBUILD_PLATLIB_DIR}/leapctype/
    )
else()
    install(TARGETS leapct
      PUBLIC_HEADER DESTINATION include
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
    )
endif()
