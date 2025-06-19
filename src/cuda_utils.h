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

#ifdef __USE_NOTEX
#define TEX_DATA float*
#define TEX_ARRAY float*
#define TEX1D_N(img, img_dim, x)            getTex1D_nearest(img, img_dim, x)
#define TEX1D_L(img, img_dim, x)            getTex1D_linear(img, img_dim, x)
#define TEX3D_N1(img, img_dim, x, y, z)     getTex3D_nearest1(img, img_dim, x, y, z)
#define TEX3D_N2(img, img_dim, z, y, x)     getTex3D_nearest2(img, img_dim, z, y, x)
#define TEX3D_L1(img, img_dim, x, y, z)     getTex3D_linear1(img, img_dim, x, y, z)  // zyx order: image space, border color in most cases
#define TEX3D_L2(img, img_dim, z, y, x)     getTex3D_linear2(img, img_dim, z, y, x)  // xyz order: projection space, clamp in most cases
#else
#define TEX_DATA cudaTextureObject_t 
#define TEX_ARRAY cudaArray*
#define TEX1D_N(img, img_dim, x)            tex1D<float>(img, x)
#define TEX1D_L(img, img_dim, x)            tex1D<float>(img, x)
#define TEX3D_N1(img, img_dim, x, y, z)     tex3D<float>(img, x, y, z)
#define TEX3D_N2(img, img_dim, x, y, z)     tex3D<float>(img, x, y, z)
#define TEX3D_L1(img, img_dim, x, y, z)     tex3D<float>(img, x, y, z)
#define TEX3D_L2(img, img_dim, x, y, z)     tex3D<float>(img, x, y, z)
#endif


//// these linear interpolation functions are for GPUs with no texture memory support (e.g., AMD MI300)
////
#ifdef __USE_NOTEX

#define __MIN__(a, b) ((a) < (b) ? (a) : (b))
#define __MAX__(a, b) ((a) > (b) ? (a) : (b))
#define __CLAMP__(x, minval, maxval)  (__MAX__(minval, __MIN__(x, maxval)))


template<typename T> __device__ __forceinline__ T ldg(const T* ptr) 
{
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__device__ inline float getTex1D_nearest(const float* img, int img_dim, float x)
{
    int x0 = (int)(x - 0.5);
    if (x0 < 0 || x0 >= img_dim)
        return 0;
    
    float result = ldg(&img[x0]);
    return result;  
}

__device__ inline float getTex1D_linear(const float* img, int img_dim, float x)
{
    x -= 0.5;
    if (x < 0 || x >= img_dim)
        return 0;
        
    int x0 = static_cast<int>(x);
    int x1 = __MIN__(x0 + 1, img_dim - 1);
    float c0 = ldg(&img[x0]);
    float c1 = ldg(&img[x1]);
    if (x0 == x1) {
        return c0;
    }
    else {
        return c0 + ((x - x0) * (c1 - c0)) / (x1 - x0);
    }
}

__device__ inline float getTex3D_nearest1(const float* img, int4 img_dim, float x, float y, float z)
{
    int x0 = (int)(x - 0.5);
    int y0 = (int)(y - 0.5);
    int z0 = (int)(z - 0.5);
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 >= img_dim.x || y0 >= img_dim.y || z0 >= img_dim.z)
        return 0;
    
    float result = ldg(&img[z0 * img_dim.y * img_dim.x + y0 * img_dim.x + x0]);
    return result;  
}

__device__ inline float getTex3D_nearest1(const float* img, int3 img_dim, float x, float y, float z)
{
    int4 img_dim2;
    img_dim2.x = img_dim.x; img_dim2.y = img_dim.y; img_dim2.z = img_dim.z;
    return getTex3D_nearest1(img, img_dim2, x, y, z);
}

__device__ inline float getTex3D_nearest2(const float* img, int4 img_dim, float z, float y, float x)
{
    int x0 = (int)(x - 0.5);
    int y0 = (int)(y - 0.5);
    int z0 = (int)(z - 0.5);
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 >= img_dim.x || y0 >= img_dim.y || z0 >= img_dim.z)
        return 0;
    
    float result = ldg(&img[x0 * img_dim.y * img_dim.z + y0 * img_dim.z + z0]);
    return result;  
}

__device__ inline float getTex3D_nearest2(const float* img, int3 img_dim, float z, float y, float x)
{
    int4 img_dim2;
    img_dim2.x = img_dim.x; img_dim2.y = img_dim.y; img_dim2.z = img_dim.z;
    return getTex3D_nearest2(img, img_dim2, z, y, x);
}

__device__ inline float getTex3D_linear1(const float* img, int4 img_dim, float x, float y, float z)
{
    x -= 0.5;
    y -= 0.5;
    z -= 0.5;
    if (x < 0 || y < 0 || z < 0 || x >= img_dim.x || y >= img_dim.y || z >= img_dim.z)
        return 0;
        
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int z0 = static_cast<int>(z);
    int x1 = __MIN__(x0 + 1, img_dim.x - 1);
    int y1 = __MIN__(y0 + 1, img_dim.y - 1);
    int z1 = __MIN__(z0 + 1, img_dim.z - 1);

    float tx = x - x0;
    float ty = y - y0;
    float tz = z - z0;

    int w = img_dim.x;
    int wh = img_dim.y * img_dim.x;
    float c000 = ldg(&img[z0 * wh + y0 * w + x0]);
    float c100 = ldg(&img[z0 * wh + y0 * w + x1]);
    float c010 = ldg(&img[z0 * wh + y1 * w + x0]);
    float c110 = ldg(&img[z0 * wh + y1 * w + x1]);
    float c001 = ldg(&img[z1 * wh + y0 * w + x0]);
    float c101 = ldg(&img[z1 * wh + y0 * w + x1]);
    float c011 = ldg(&img[z1 * wh + y1 * w + x0]);
    float c111 = ldg(&img[z1 * wh + y1 * w + x1]);

    // interpolate x-direction
    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    // interpolate y-direction
    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    // interpolate z-direction
    float result = c0 * (1 - tz) + c1 * tz;

    return result;
}

__device__ inline float getTex3D_linear1(const float* img, int3 img_dim, float x, float y, float z)
{
    int4 img_dim2;
    img_dim2.x = img_dim.x; img_dim2.y = img_dim.y; img_dim2.z = img_dim.z;
    return getTex3D_linear1(img, img_dim2, x, y, z);
}

__device__ inline float getTex3D_linear2(const float* img, int4 img_dim, float z, float y, float x)
{
    x -= 0.5;
    y -= 0.5;
    z -= 0.5;

    // for border color
    if (x < 0 || y < 0 || z < 0 || x >= img_dim.x || y >= img_dim.y || z >= img_dim.z)
        return 0;
        
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int z0 = static_cast<int>(z);
    int x1 = __MIN__(x0 + 1, img_dim.x - 1);
    int y1 = __MIN__(y0 + 1, img_dim.y - 1);
    int z1 = __MIN__(z0 + 1, img_dim.z - 1);

/*
    // for clamp
    x = __MIN__(x, img_dim.x-1);
    y = __MIN__(y, img_dim.y-1);
    z = __MIN__(z, img_dim.z-1);
    x = __MAX__(x, 0);
    y = __MAX__(y, 0);
    z = __MAX__(z, 0);
    int x0 = (int)x;
    int y0 = (int)y;
    int z0 = (int)z;
    int x1 = __MIN__(x0 + 1, img_dim.x - 1);
    int y1 = __MIN__(y0 + 1, img_dim.y - 1);
    int z1 = __MIN__(z0 + 1, img_dim.z - 1);
    x1 = __MAX__(x1, 0);
    y1 = __MAX__(y1, 0);
    z1 = __MAX__(z1, 0);
*/

    float tx = x - x0;
    float ty = y - y0;
    float tz = z - z0;

    int d = img_dim.z;
    int dh = img_dim.y * img_dim.z;
    float c000 = ldg(&img[x0 * dh + y0 * d + z0]);
    float c100 = ldg(&img[x0 * dh + y0 * d + z1]);
    float c010 = ldg(&img[x0 * dh + y1 * d + z0]);
    float c110 = ldg(&img[x0 * dh + y1 * d + z1]);
    float c001 = ldg(&img[x1 * dh + y0 * d + z0]);
    float c101 = ldg(&img[x1 * dh + y0 * d + z1]);
    float c011 = ldg(&img[x1 * dh + y1 * d + z0]);
    float c111 = ldg(&img[x1 * dh + y1 * d + z1]);

    // interpolate x-direction
    float c00 = c000 * (1 - tx) + c100 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    // interpolate y-direction
    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    // interpolate z-direction
    float result = c0 * (1 - tz) + c1 * tz;

    return result;
}

__device__ inline float getTex3D_linear2(const float* img, int3 img_dim, float z, float y, float x)
{
    int4 img_dim2;
    img_dim2.x = img_dim.x; img_dim2.y = img_dim.y; img_dim2.z = img_dim.z;
    return getTex3D_linear2(img, img_dim2, z, y, x);
}

#endif

extern TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

extern TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
extern TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

extern TEX_ARRAY loadTexture1D_from_cpu(TEX_DATA& tex_object, float* data, const int N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern TEX_ARRAY loadTexture1D(TEX_DATA& tex_object, float* dev_data, const int N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
extern TEX_ARRAY loadTexture2D(TEX_DATA& tex_object, float* data, const int2 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

extern void freeTexture(TEX_ARRAY& tex_array, TEX_DATA& tex_object, bool volume_on_cpu);
extern void freeTexture1D(TEX_ARRAY& tex_array, TEX_DATA& tex_object, bool volume_on_cpu);

// temp
//extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
//extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
//extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
//extern cudaArray* loadTexture_from_cpu(cudaTextureObject_t& tex_object, float* data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
//extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
//extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
//extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
//extern cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
// temp

/*
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
*/

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
