
#pragma once

#include <string>
#include <vector>

#include "cuda_runtime.h"
//#include "cufft.h"
#include "parameters.h"


cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);
cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions);

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);
cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation = true, bool useLinearInterpolation = true);

// Utility Functions for pushing/pulling data to/from CPU/GPU
float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU);
bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU);
float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU);
bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU);

float* copy3DdataToGPU(float* g, int3 N, int whichGPU);
bool pull3DdataFromGPU(float* g, int3 N, float* dev_g, int whichGPU);
