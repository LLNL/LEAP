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
