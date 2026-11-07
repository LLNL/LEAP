list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hipify_torch/cmake")
include(Hipify)

function(leap_hipify_sources)
    hipify(
        CUDA_SOURCE_DIR "${LEAP_ORIGINAL_SRC_DIR}"
        HIP_SOURCE_DIR "${LEAP_HIPIFIED_SRC_DIR}"
    )
endfunction()
