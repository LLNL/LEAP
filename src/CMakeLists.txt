find_package(OpenMP REQUIRED COMPONENTS CXX)
if(LEAP_CUDA)
    find_package(CUDAToolkit 11.7 REQUIRED)
endif()

if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
  set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
endif()

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

set(HEADER
  analytic_ray_tracing.h
  analytic_ray_tracing_gpu.cuh
  backprojectors_VD.cuh
  bilateral_filter.cuh
  cpu_utils.h
  cuda_utils.h
  file_io.h
  filtered_backprojection.h
  find_center_cpu.h
  geometric_calibration.cuh
  guided_filter.cuh
  leap_defines.h
  list_of_tomographic_models.h
  log.h
  matching_pursuit.cuh
  noise_filters.cuh
  parameters.h
  phantom.h
  projectors.h
  projectors_Joseph.cuh
  projectors_Joseph_cpu.h
  projectors_SF.cuh
  projectors_SF_cpu.h
  projectors_Siddon.cuh
  projectors_Siddon_cpu.h
  projectors_attenuated.cuh
  projectors_extendedSF.cuh
  projectors_symmetric.cuh
  projectors_symmetric_cpu.h
  ramp_filter.cuh
  ramp_filter_cpu.h
  ray_weighting.cuh
  ray_weighting_cpu.h
  rebin.h
  resample.cuh
  resample_cpu.h
  scatter_models.cuh
  sensitivity.cuh
  sensitivity_cpu.h
  sinogram_replacement.h
  tomographic_models.h
  tomographic_models_c_interface.h
  total_variation.cuh
)

set(SRC_CPP
  analytic_ray_tracing.cpp
  cpu_utils.cpp
  file_io.cpp
  filtered_backprojection.cpp
  find_center_cpu.cpp
  list_of_tomographic_models.cpp
  parameters.cpp
  phantom.cpp
  projectors.cpp
  projectors_Joseph_cpu.cpp
  projectors_SF_cpu.cpp
  projectors_Siddon_cpu.cpp
  projectors_symmetric_cpu.cpp
  ramp_filter_cpu.cpp
  ray_weighting_cpu.cpp
  rebin.cpp
  resample_cpu.cpp
  sensitivity_cpu.cpp
  sinogram_replacement.cpp
  tomographic_models.cpp
  tomographic_models_c_interface.cpp
)

set(SRC_CU
  analytic_ray_tracing_gpu.cu
  backprojectors_VD.cu
  bilateral_filter.cu
  cuda_utils.cu
  geometric_calibration.cu
  guided_filter.cu
  matching_pursuit.cu
  noise_filters.cu
  projectors_Joseph.cu
  projectors_SF.cu
  projectors_Siddon.cu
  projectors_attenuated.cu
  projectors_extendedSF.cu
  projectors_symmetric.cu
  ramp_filter.cu
  ray_weighting.cu
  resample.cu
  scatter_models.cu
  sensitivity.cu
  total_variation.cu
)

include_directories(
  ./
  ${CUDA_INCLUDE_DIRS}
)

#add_library(leapct SHARED main_projector_ctype.cpp parameters.cpp projectors_cpu.cpp projectors.cu projectors_SF.cu)
#target_link_libraries(leapct "${TORCH_LIBRARIES}")
add_library(leapct SHARED
  ${HEADER}
  ${SRC_CPP}
)

if(LEAP_CUDA)
    target_sources(leapct PRIVATE ${SRC_CU})
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
        __USE_NOTEX
        __INCLUDE_CUFFT
    )
    target_link_libraries(${PROJECT_NAME}
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
    target_link_libraries(${PROJECT_NAME}
      OpenMP::OpenMP_CXX
      hip::hipfft
    )
else ()
    target_compile_definitions(
        leapct PRIVATE
        __USE_CPU
    )
    target_link_libraries(${PROJECT_NAME}
      OpenMP::OpenMP_CXX
    )
endif()

target_compile_options(leapct PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
>)

if(SKBUILD)
    install(TARGETS ${PROJECT_NAME}
    DESTINATION ${SKBUILD_PLATLIB_DIR}/leapctype/
    )
else()
    install(TARGETS ${PROJECT_NAME}
      PUBLIC_HEADER DESTINATION include
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
    )
endif()
