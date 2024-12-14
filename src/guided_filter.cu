////////////////////////////////////////////////////////////////////////////////
// Copyright 2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for guided filter algorithms
////////////////////////////////////////////////////////////////////////////////
#include "guided_filter.cuh"
#include "cuda_runtime.h"
#include "cuda_utils.h"

#include <iostream>
#include <vector>

__global__ void calcScaleAndShift(const float* f, float* scale, float* shift, const int3 N, const int r, const float epsilon)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, r);
    const int di_max = min(N.x - 1 - i, r);

    const int dj_min = -min(j, r);
    const int dj_max = min(N.y - 1 - j, r);

    const int dk_min = -min(k, r);
    const int dk_max = min(N.z - 1 - k, r);

    const float weight = 1.0f / float((di_max - di_min + 1) * (dj_max - dj_min + 1) * (dk_max - dk_min + 1));

    float x = 0.0f;
    float xx = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        // const int i_plus_di = max(0, min(N.x, i + di));
        //const float* f_slice = &f[uint64(i + di) * uint64(N.z * N.y)];
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            //const float* f_line = &f_slice[uint64((j + dj) * N.z)];
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                //const float neighborVal = f_slice[uint64((j + dj) * N.z + (k + dk))];
                //const float neighborVal = f_line[k + dk];

                const float curVal = f[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
                x += curVal;
                xx += curVal * curVal;
            }
        }
    }

    const float meanI = x * weight;
    const float varI = xx * weight - meanI * meanI;
    const float a = varI / (varI + epsilon); // absolute
    //const float a = varI / (varI + epsilon*max(meanI,1.0e-8)); // relative
    const float b = meanI - a * meanI;
    
    const uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    scale[ind] = a;
    shift[ind] = b;
}

__global__ void guidedFilterSecondStep(float* f, const float* a, const float* b, const int3 N, const int r)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, r);
    const int di_max = min(N.x - 1 - i, r);

    const int dj_min = -min(j, r);
    const int dj_max = min(N.y - 1 - j, r);

    const int dk_min = -min(k, r);
    const int dk_max = min(N.z - 1 - k, r);

    const float weight = 1.0f / float((di_max - di_min + 1) * (dj_max - dj_min + 1) * (dk_max - dk_min + 1));

    float mean_a = 0.0f;
    float mean_b = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        // const int i_plus_di = max(0, min(N.x, i + di));
        //const float* f_slice = &f[uint64(i + di) * uint64(N.z * N.y)];
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            //const float* f_line = &f_slice[uint64((j + dj) * N.z)];
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                //const float neighborVal = f_slice[uint64((j + dj) * N.z + (k + dk))];
                //const float neighborVal = f_line[k + dk];

                mean_a += a[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
                mean_b += b[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
            }
        }
    }
    mean_a *= weight;
    mean_b *= weight;

    const uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    f[ind] = mean_a * f[ind] + mean_b;
}

bool guidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu, int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_scale = 0;
    if (cudaMalloc((void**)&dev_scale, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }
    float* dev_shift = 0;
    if (cudaMalloc((void**)&dev_shift, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);

    bool useTexture = false;

    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = NULL;
    if (useTexture)
    {
        //f_data_array = loadTexture(f_data_txt, dev_f, N, false, false);
        //bilateralFilterKernel_txt <<< dimGrid, dimBlock >>> (f_data_txt, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    else
    {
        for (int iter = 0; iter < numIter; iter++)
        {
            calcScaleAndShift <<< dimGrid, dimBlock >>> (dev_f, dev_scale, dev_shift, N, r, epsilon);
            guidedFilterSecondStep <<< dimGrid, dimBlock >>> (dev_f, dev_scale, dev_shift, N, r);
        }
    }
    cudaDeviceSynchronize();

    // Clean up
    if (useTexture)
    {
        cudaFreeArray(f_data_array);
        cudaDestroyTextureObject(f_data_txt);
    }
    if (data_on_cpu)
    {
        // pull result off GPU
        /*
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else //*/
        pull3DdataFromGPU(f, N, dev_f, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }

    cudaFree(dev_scale);
    cudaFree(dev_shift);

    return true;
}
