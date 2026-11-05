////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// basic CUDA operations
////////////////////////////////////////////////////////////////////////////////
#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#pragma once

#include <string>
#include <vector>

#include "parameters.h"

/**
 * This header and associated source file are for generic GPU-based functions that are used in LEAP
 */

#define GPU_MEMORY_SAFETY_MULTIPLIER 0.9

#ifndef __USE_CPU
#include "cuda_runtime.h"

// LEAP samples float element-read textures at fractional coordinates and relies
// on cudaFilterModeLinear hardware interpolation. HIP does not offer linear
// filtering on cudaReadModeElementType float textures, so on HIP every texture
// is sampled point-mode and the trilinear/bilinear/linear blend is done in
// software by the leapTex* helpers below. Point-bound fetch sites pass
// linear = false; a few kernels pick the bind mode at runtime and pass that
// flag through. CUDA keeps the hardware texture path unchanged.
//
// Coordinate convention matches CUDA unnormalized texture sampling: a point
// fetch at c returns texel floor(c); a linear fetch blends texels floor(c - 0.5)
// and floor(c - 0.5) + 1 with weight (c - 0.5) - floor(c - 0.5).
//
// The texture representation depends on the GPU:
// - GPUs with image support (hipDeviceAttributeImageSupport = 1) use real
//   point-mode texture objects (loadTexture* in cuda_utils.cu). The corners of
//   a blend are point fetches at texel centers, so the texture address mode
//   (Border-zero or Clamp) governs out-of-range neighbors with no past-end fault.
// - GPUs without image support (gfx942, gfx950; the device compiler defines
//   __HIP_NO_IMAGE_SUPPORT for them and rejects the texture builtins) get a
//   linear device buffer instead: a leapTextureHeader followed by the dense
//   data, and the cudaTextureObject_t handle points at it. leapTexel applies
//   the bound address mode (Border returns zero, Clamp repeats the edge texel).
// The device code of each target reports which of the two it reads (see
// leapUseLinearTexture in cuda_utils.cu), and loadTexture* refuses to create a
// handle that disagrees with it, so a fat binary hands each GPU the kind its
// code expects.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
struct leapTextureHeader
{
    int n[3]; // width, height, depth; texel (i, j, k) is at (k * n[1] + j) * n[0] + i
    int clamp; // 1: cudaAddressModeClamp, 0: cudaAddressModeBorder
};

// The call sites release either kind through the usual cudaFreeArray /
// cudaDestroyTextureObject pair; these dispatch on the kind of the handle.
// Function-like so that cuda_utils.cu can still reach the runtime functions
// as (cudaFreeArray)(...) and (cudaDestroyTextureObject)(...).
cudaError_t leapFreeArray(cudaArray* array);
cudaError_t leapDestroyTextureObject(cudaTextureObject_t tex_object);
#define cudaFreeArray(array) leapFreeArray(array)
#define cudaDestroyTextureObject(tex_object) leapDestroyTextureObject(tex_object)
#endif

// Only visible to the device compiler; host .cpp translation units that include
// this header (compiled as plain C++) must not see the texture builtins.
#if defined(__CUDACC__) || defined(__HIPCC__)
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#if __HIP_NO_IMAGE_SUPPORT
// Texel at integer-valued indices (i, j, k), with the bound address mode applied.
__device__ __forceinline__ float leapTexel(cudaTextureObject_t t, float i, float j, float k)
{
    // The buffer is read-only for the lifetime of every kernel that reads it.
    typedef const __attribute__((address_space(4))) leapTextureHeader* header_ptr;
    typedef const __attribute__((address_space(4))) float* data_ptr;
    const header_ptr h = (header_ptr)t;
    const float w = h->n[0], ht = h->n[1], d = h->n[2];
    if (h->clamp)
    {
        i = fminf(fmaxf(i, 0.0f), w - 1.0f);
        j = fminf(fmaxf(j, 0.0f), ht - 1.0f);
        k = fminf(fmaxf(k, 0.0f), d - 1.0f);
    }
    else if (!(i >= 0.0f && i < w && j >= 0.0f && j < ht && k >= 0.0f && k < d))
        return 0.0f;
    const data_ptr data = (data_ptr)(h + 1);
    return data[(size_t(int(k)) * size_t(h->n[1]) + size_t(int(j))) * size_t(h->n[0]) + size_t(int(i))];
}

// A non-finite coordinate (e.g. a 0/0 detector-edge weight upstream) returns a
// finite border/clamped value on CUDA hardware, which the call site multiplies
// by a zero weight; map it to a finite coordinate so the result stays finite.
__device__ __forceinline__ float leapTexCoord(float x)
{
    return isfinite(x) ? x : 0.0f;
}

__device__ __forceinline__ float leapTex3D(cudaTextureObject_t t, float x, float y, float z, bool linear = true)
{
    if (!linear)
        return leapTexel(t, floorf(leapTexCoord(x)), floorf(leapTexCoord(y)), floorf(leapTexCoord(z)));
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float ys = isfinite(y) ? y - 0.5f : 0.0f;
    const float zs = isfinite(z) ? z - 0.5f : 0.0f;
    const float x0 = floorf(xs), y0 = floorf(ys), z0 = floorf(zs);
    const float fx = xs - x0, fy = ys - y0, fz = zs - z0;
    const float x1 = x0 + 1.0f, y1 = y0 + 1.0f, z1 = z0 + 1.0f;

    const float c000 = leapTexel(t, x0, y0, z0);
    const float c100 = leapTexel(t, x1, y0, z0);
    const float c010 = leapTexel(t, x0, y1, z0);
    const float c110 = leapTexel(t, x1, y1, z0);
    const float c001 = leapTexel(t, x0, y0, z1);
    const float c101 = leapTexel(t, x1, y0, z1);
    const float c011 = leapTexel(t, x0, y1, z1);
    const float c111 = leapTexel(t, x1, y1, z1);

    const float c00 = c000 + fx * (c100 - c000);
    const float c10 = c010 + fx * (c110 - c010);
    const float c01 = c001 + fx * (c101 - c001);
    const float c11 = c011 + fx * (c111 - c011);
    const float c0 = c00 + fy * (c10 - c00);
    const float c1 = c01 + fy * (c11 - c01);
    return c0 + fz * (c1 - c0);
}

__device__ __forceinline__ float leapTex2D(cudaTextureObject_t t, float x, float y, bool linear = true)
{
    if (!linear)
        return leapTexel(t, floorf(leapTexCoord(x)), floorf(leapTexCoord(y)), 0.0f);
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float ys = isfinite(y) ? y - 0.5f : 0.0f;
    const float x0 = floorf(xs), y0 = floorf(ys);
    const float fx = xs - x0, fy = ys - y0;
    const float x1 = x0 + 1.0f, y1 = y0 + 1.0f;

    const float c00 = leapTexel(t, x0, y0, 0.0f);
    const float c10 = leapTexel(t, x1, y0, 0.0f);
    const float c01 = leapTexel(t, x0, y1, 0.0f);
    const float c11 = leapTexel(t, x1, y1, 0.0f);

    const float c0 = c00 + fx * (c10 - c00);
    const float c1 = c01 + fx * (c11 - c01);
    return c0 + fy * (c1 - c0);
}

__device__ __forceinline__ float leapTex1D(cudaTextureObject_t t, float x, bool linear = true)
{
    if (!linear)
        return leapTexel(t, floorf(leapTexCoord(x)), 0.0f, 0.0f);
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float x0 = floorf(xs);
    const float fx = xs - x0;
    const float c0 = leapTexel(t, x0, 0.0f, 0.0f);
    const float c1 = leapTexel(t, x0 + 1.0f, 0.0f, 0.0f);
    return c0 + fx * (c1 - c0);
}
#else
__device__ __forceinline__ float leapTex3D(cudaTextureObject_t t, float x, float y, float z, bool linear = true)
{
    if (!linear)
        return tex3D<float>(t, x, y, z);
    // A non-finite coordinate (e.g. a 0/0 detector-edge weight upstream) returns a
    // finite border/clamped value on CUDA hardware; sanitize so the point-mode
    // corner fetches and the blend stay finite and the result matches.
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float ys = isfinite(y) ? y - 0.5f : 0.0f;
    const float zs = isfinite(z) ? z - 0.5f : 0.0f;
    const float x0 = floorf(xs), y0 = floorf(ys), z0 = floorf(zs);
    const float fx = xs - x0, fy = ys - y0, fz = zs - z0;
    const float ix0 = x0 + 0.5f, iy0 = y0 + 0.5f, iz0 = z0 + 0.5f;
    const float ix1 = ix0 + 1.0f, iy1 = iy0 + 1.0f, iz1 = iz0 + 1.0f;

    const float c000 = tex3D<float>(t, ix0, iy0, iz0);
    const float c100 = tex3D<float>(t, ix1, iy0, iz0);
    const float c010 = tex3D<float>(t, ix0, iy1, iz0);
    const float c110 = tex3D<float>(t, ix1, iy1, iz0);
    const float c001 = tex3D<float>(t, ix0, iy0, iz1);
    const float c101 = tex3D<float>(t, ix1, iy0, iz1);
    const float c011 = tex3D<float>(t, ix0, iy1, iz1);
    const float c111 = tex3D<float>(t, ix1, iy1, iz1);

    const float c00 = c000 + fx * (c100 - c000);
    const float c10 = c010 + fx * (c110 - c010);
    const float c01 = c001 + fx * (c101 - c001);
    const float c11 = c011 + fx * (c111 - c011);
    const float c0 = c00 + fy * (c10 - c00);
    const float c1 = c01 + fy * (c11 - c01);
    return c0 + fz * (c1 - c0);
}

__device__ __forceinline__ float leapTex2D(cudaTextureObject_t t, float x, float y, bool linear = true)
{
    if (!linear)
        return tex2D<float>(t, x, y);
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float ys = isfinite(y) ? y - 0.5f : 0.0f;
    const float x0 = floorf(xs), y0 = floorf(ys);
    const float fx = xs - x0, fy = ys - y0;
    const float ix0 = x0 + 0.5f, iy0 = y0 + 0.5f;
    const float ix1 = ix0 + 1.0f, iy1 = iy0 + 1.0f;

    const float c00 = tex2D<float>(t, ix0, iy0);
    const float c10 = tex2D<float>(t, ix1, iy0);
    const float c01 = tex2D<float>(t, ix0, iy1);
    const float c11 = tex2D<float>(t, ix1, iy1);

    const float c0 = c00 + fx * (c10 - c00);
    const float c1 = c01 + fx * (c11 - c01);
    return c0 + fy * (c1 - c0);
}

__device__ __forceinline__ float leapTex1D(cudaTextureObject_t t, float x, bool linear = true)
{
    if (!linear)
        return tex1D<float>(t, x);
    const float xs = isfinite(x) ? x - 0.5f : 0.0f;
    const float x0 = floorf(xs);
    const float fx = xs - x0;
    const float c0 = tex1D<float>(t, x0 + 0.5f);
    const float c1 = tex1D<float>(t, x0 + 1.5f);
    return c0 + fx * (c1 - c0);
}
#endif
#else
__device__ __forceinline__ float leapTex3D(cudaTextureObject_t t, float x, float y, float z, bool linear = true) { return tex3D<float>(t, x, y, z); }
__device__ __forceinline__ float leapTex2D(cudaTextureObject_t t, float x, float y, bool linear = true) { return tex2D<float>(t, x, y); }
__device__ __forceinline__ float leapTex1D(cudaTextureObject_t t, float x, bool linear = true) { return tex1D<float>(t, x); }
#endif
#endif // __CUDACC__ || __HIPCC__

#define CUDA_CHECK(expr) cudaSafeCall((expr), __FILE__, __LINE__, #expr)
#define CHECK_LAST_ERROR() cudaCall(cudaGetLastError(), NULL, __FILE__, __LINE__)

int getSPcores(int whichGPU = 0);
void printGPUproperties(int whichGPU = 0);
extern int numberOfGPUs();
extern float getAvailableGPUmemory(int whichGPU);
extern float getAvailableGPUmemory(std::vector<int> whichGPUs);

extern dim3 setBlockSize(int3 N);
extern dim3 setGridSize(int3 N, dim3 dimBlock);
extern dim3 setBlockSize(int4 N);
extern dim3 setGridSize(int4 N, dim3 dimBlock);

extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

extern cudaArray* loadTexture1D(cudaTextureObject_t& tex_object, float* data, const int N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern cudaArray* loadTexture2D(cudaTextureObject_t& tex_object, float* data, const int2 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

// Utility Functions for pushing/pulling data to/from CPU/GPU
extern float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU);
extern bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU);
extern float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU);
extern bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU);
extern float* copy3DdataToGPU(float* g, int3 N, int whichGPU);
extern bool pull3DdataFromGPU(float* g, int3 N, float* dev_g, int whichGPU);
extern float* copy1DdataToGPU(float* x, int N, int whichGPU);
extern bool* copy1DbooleanToGPU(bool* x, int N, int whichGPU);

extern float* copyAngleArrayToGPU(parameters* params);
bool setProjectionGPUparams(parameters*, int4&, float4&, float4&, bool doNormalize = false);
bool setVolumeGPUparams(parameters*, int4&, float4&, float4&);

extern cudaError_t setToConstant(float* dev_lhs, const float c, const int3 N, int whichGPU = 0);
extern cudaError_t equal(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t multiply(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t divide(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t add(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t add(float* dev_lhs, const float c, const int3 N, int whichGPU = 0);
extern cudaError_t sub(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t scale(float* dev_lhs, const float c, const int3 N, int whichGPU = 0);
extern cudaError_t scalarAdd(float* dev_lhs, const float c, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern cudaError_t mean_over_slices(float* dev_lhs, const int3 N, int whichGPU = 0);

extern cudaError_t replaceZeros(float* dev_lhs, const int3 N, int whichGPU = 0, float newVal = 1.0);

extern cudaError_t clip(float* dev_lhs, const int3 N, int whichGPU = 0, float clipVal = 0.0);
extern cudaError_t cosFcn(float* dev_lhs, const int3 N, int whichGPU = 0);
extern cudaError_t sinFcn(float* dev_lhs, const int3 N, int whichGPU = 0);
extern cudaError_t expFcn(float* dev_lhs, const int3 N, int whichGPU = 0);
extern cudaError_t negExpFcn(float* dev_lhs, const int3 N, int whichGPU = 0);

extern float sum(const float* dev_lhs, const int3 N, int whichGPU = 0);
extern float innerProduct(const float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU = 0);
extern float weightedInnerProduct(const float* dev_lhs, const float* dev_w, const float* dev_rhs, const int3 N, int whichGPU = 0);

bool windowFOV_gpu(float* f, parameters* params);

bool copy_volume_data_to_mask_gpu(float* f, float* mask, parameters* params, bool do_forward);

bool applyTransferFunction_gpu(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, int whichGPU, bool data_on_cpu);
bool applyDualTransferFunction_gpu(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, int whichGPU, bool data_on_cpu);
#else
extern int numberOfGPUs();
extern float getAvailableGPUmemory(int whichGPU);
extern float getAvailableGPUmemory(std::vector<int> whichGPUs);
#endif

#endif
