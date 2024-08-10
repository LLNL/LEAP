////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for bilateral filter algorithms
// This file is an adaptation of code written by myself
// (Kyle) several years ago in a package called "3Ddensoing"
////////////////////////////////////////////////////////////////////////////////
#include "bilateral_filter.cuh"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.h"
#include "noise_filters.cuh"

#include <iostream>
#include <vector>

__global__ void bilateralFilterKernel(float* f, float* f_filtered, const int3 N, const float sigma_d_sq_inv,
                                      const float sigma_i_sq_inv, const int w)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i > N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, w);
    const int di_max = min(N.x - 1 - i, w);

    const int dj_min = -min(j, w);
    const int dj_max = min(N.y - 1 - j, w);

    const int dk_min = -min(k, w);
    const int dk_max = min(N.z - 1 - k, w);

    uint64 ind = uint64(i) * uint64(N.z * N.y) + uint64(j * N.z + k);
    const float curVal = f[ind];

    // f_filtered[ind] = curVal;

    float val = 0.0f;
    float w_sum = 0.0f;
    //*
    for (int di = di_min; di <= di_max; di++)
    {
        // const int i_plus_di = max(0, min(N.x, i + di));
        const float* f_slice = &f[uint64(i + di) * uint64(N.z * N.y)];
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            const float* f_line = &f_slice[uint64((j + dj) * N.z)];
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                //const float neighborVal = f_slice[uint64((j + dj) * N.z + (k + dk))];
                const float neighborVal = f_line[k + dk];

                // const float x = (di * di + dj * dj + dk * dk) * sigma_d_sq_inv + (curVal - neighborVal)*(curVal -
                // neighborVal) * sigma_i_sq_inv; const float w_cur = (x <= 10.0) ? exp_int[int(x)] * (1.0 + (x -
                // floorf(x))*(linTerm + quadTerm * (x - floorf(x)))) : 0.0;
                const float w_cur = expf(-((di * di + dj * dj + dk * dk) * sigma_d_sq_inv +
                                          (curVal - neighborVal) * (curVal - neighborVal) * sigma_i_sq_inv));
                val += w_cur * neighborVal;
                w_sum += w_cur;
            }
        }
    }

    if (w_sum > 0.0f)
        f_filtered[ind] = val / w_sum;
    else
        f_filtered[ind] = curVal;
}

__global__ void bilateralFilterKernel_txt(cudaTextureObject_t f, float* f_filtered, const int3 N, const float sigma_d_sq_inv,
    const float sigma_i_sq_inv, const int w)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i > N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, w);
    const int di_max = min(N.x - 1 - i, w);

    const int dj_min = -min(j, w);
    const int dj_max = min(N.y - 1 - j, w);

    const int dk_min = -min(k, w);
    const int dk_max = min(N.z - 1 - k, w);

    uint64 ind = uint64(i) * uint64(N.z * N.y) + uint64(j * N.z + k);
    //const float curVal = f[ind];
    const float curVal = tex3D<float>(f, k, j, i);

    // f_filtered[ind] = curVal;

    float val = 0.0f;
    float w_sum = 0.0f;
    //*
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
                const float neighborVal = tex3D<float>(f, k+dk, j+dj, i+di);

                // const float x = (di * di + dj * dj + dk * dk) * sigma_d_sq_inv + (curVal - neighborVal)*(curVal -
                // neighborVal) * sigma_i_sq_inv; const float w_cur = (x <= 10.0) ? exp_int[int(x)] * (1.0 + (x -
                // floorf(x))*(linTerm + quadTerm * (x - floorf(x)))) : 0.0;
                const float w_cur = expf(-((di * di + dj * dj + dk * dk) * sigma_d_sq_inv +
                    (curVal - neighborVal) * (curVal - neighborVal) * sigma_i_sq_inv));
                val += w_cur * neighborVal;
                w_sum += w_cur;
            }
        }
    }

    if (w_sum > 0.0f)
        f_filtered[ind] = val / w_sum;
    else
        f_filtered[ind] = curVal;
}

__global__ void scaledBilateralFilterKernel(float* f, float* Bf, float* f_filtered, const int3 N, const float sigma_d_sq_inv,
                                      const float sigma_i_sq_inv, const int w)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i > N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, w);
    const int di_max = min(N.x - 1 - i, w);

    const int dj_min = -min(j, w);
    const int dj_max = min(N.y - 1 - j, w);

    const int dk_min = -min(k, w);
    const int dk_max = min(N.z - 1 - k, w);

    uint64 ind = uint64(i) * uint64(N.z * N.y) + uint64(j * N.z + k);
    const float curVal = f[ind];

    // f_filtered[ind] = curVal;

    float val = 0.0f;
    float w_sum = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        // const int i_plus_di = max(0, min(N.x, i + di));
        const float* Bf_slice = &Bf[uint64(i + di) * uint64(N.z * N.y)];
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            const float* Bf_line = &Bf_slice[uint64((j + dj) * N.z)];
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                //const float neighborVal = Bf[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
                const float neighborVal = Bf_line[k + dk];

                // const float x = (di * di + dj * dj + dk * dk) * sigma_d_sq_inv + (curVal - neighborVal)*(curVal -
                // neighborVal) * sigma_i_sq_inv; const float w_cur = (x <= 10.0) ? exp_int[int(x)] * (1.0 + (x -
                // floorf(x))*(linTerm + quadTerm * (x - floorf(x)))) : 0.0;
                const float w_cur = expf(-((di * di + dj * dj + dk * dk) * sigma_d_sq_inv +
                                           (curVal - neighborVal) * (curVal - neighborVal) * sigma_i_sq_inv));
                val += w_cur * neighborVal;
                w_sum += w_cur;
            }
        }
    }

    if (w_sum > 0.0f)
        f_filtered[ind] = val / w_sum;
    else
        f_filtered[ind] = curVal;
}

__global__ void scaledBilateralFilterKernel_txt(cudaTextureObject_t f, cudaTextureObject_t Bf, float* f_filtered, const int3 N, const float sigma_d_sq_inv,
    const float sigma_i_sq_inv, const int w)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i > N.x || j >= N.y || k >= N.z) return;

    const int di_min = -min(i, w);
    const int di_max = min(N.x - 1 - i, w);

    const int dj_min = -min(j, w);
    const int dj_max = min(N.y - 1 - j, w);

    const int dk_min = -min(k, w);
    const int dk_max = min(N.z - 1 - k, w);

    uint64 ind = uint64(i) * uint64(N.z * N.y) + uint64(j * N.z + k);
    //const float curVal = f[ind];
    const float curVal = tex3D<float>(f, k, j, i);

    // f_filtered[ind] = curVal;

    float val = 0.0f;
    float w_sum = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        // const int i_plus_di = max(0, min(N.x, i + di));
        //const float* Bf_slice = &Bf[uint64(i + di) * uint64(N.z * N.y)];
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            //const float* Bf_line = &Bf_slice[uint64((j + dj) * N.z)];
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                //const float neighborVal = Bf[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
                //const float neighborVal = Bf_line[k + dk];
                const float neighborVal = tex3D<float>(Bf, k+dk, j+dj, i+di);

                // const float x = (di * di + dj * dj + dk * dk) * sigma_d_sq_inv + (curVal - neighborVal)*(curVal -
                // neighborVal) * sigma_i_sq_inv; const float w_cur = (x <= 10.0) ? exp_int[int(x)] * (1.0 + (x -
                // floorf(x))*(linTerm + quadTerm * (x - floorf(x)))) : 0.0;
                const float w_cur = expf(-((di * di + dj * dj + dk * dk) * sigma_d_sq_inv +
                    (curVal - neighborVal) * (curVal - neighborVal) * sigma_i_sq_inv));
                val += w_cur * neighborVal;
                w_sum += w_cur;
            }
        }
    }

    if (w_sum > 0.0f)
        f_filtered[ind] = val / w_sum;
    else
        f_filtered[ind] = curVal;
}

bool bilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, bool data_on_cpu, int whichGPU)
{
    if (f == NULL) return false;

    spatialFWHM = float(max(0.25, min(100.0, spatialFWHM)));
    intensityFWHM = float(max(1.0e-8, min(1.0e8, intensityFWHM)));

    // convert FWHM input parameters to sigmas
    float sigma_d_sq = float(spatialFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_d_sq *= sigma_d_sq;
    float sigma_i_sq = float(intensityFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_i_sq *= sigma_i_sq;

    float sigma_d_sq_inv = float(0.5 / sigma_d_sq);
    float sigma_i_sq_inv = float(0.5 / sigma_i_sq);

    // define filter window width by full width, tenth max
    int w = max(1, int(ceil(sqrt(2.0 * log(10.0) * sigma_d_sq))));

    cudaSetDevice(whichGPU);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);

    //bool useTexture = true;
    bool useTexture = false;

    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = NULL;
    if (useTexture)
    {
        f_data_array = loadTexture(f_data_txt, dev_f, N, false, false);
        bilateralFilterKernel_txt <<< dimGrid, dimBlock >>> (f_data_txt, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    else
    {
        bilateralFilterKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
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
        pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    }

    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

//*
bool priorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* Bf, bool data_on_cpu, int whichGPU)
{
    if (Bf == NULL)
        return bilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, data_on_cpu, whichGPU);
    if (f == NULL) return false;

    spatialFWHM = float(max(0.25, min(100.0, spatialFWHM)));
    intensityFWHM = float(max(1.0e-8, min(1.0e8, intensityFWHM)));

    // convert FWHM input parameters to sigmas
    float sigma_d_sq = float(spatialFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_d_sq *= sigma_d_sq;
    float sigma_i_sq = float(intensityFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_i_sq *= sigma_i_sq;

    float sigma_d_sq_inv = float(0.5 / sigma_d_sq);
    float sigma_i_sq_inv = float(0.5 / sigma_i_sq);

    // define filter window width by full width, tenth max
    int w = max(1, int(ceil(sqrt(2.0 * log(10.0) * sigma_d_sq))));

    cudaSetDevice(whichGPU);


    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Bf = 0;
    if (data_on_cpu)
        dev_Bf = copy3DdataToGPU(Bf, N, whichGPU);
    else
        dev_Bf = Bf;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);

    //bool useTexture = true;
    bool useTexture = false;

    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = NULL;
    cudaTextureObject_t Bf_data_txt = NULL;
    cudaArray* Bf_data_array = NULL;
    if (useTexture)
    {
        f_data_array = loadTexture(f_data_txt, dev_f, N, false, false);
        Bf_data_array = loadTexture(Bf_data_txt, dev_Bf, N, false, false);
        scaledBilateralFilterKernel_txt <<< dimGrid, dimBlock >>> (f_data_txt, Bf_data_txt, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    else
    {
        scaledBilateralFilterKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Bf, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    cudaDeviceSynchronize();

    // Clean up
    if (useTexture)
    {
        cudaFreeArray(f_data_array);
        cudaDestroyTextureObject(f_data_txt);
        cudaFreeArray(Bf_data_array);
        cudaDestroyTextureObject(Bf_data_txt);
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
        pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }
    if (dev_Bf != 0 && data_on_cpu == true)
    {
        cudaFree(dev_Bf);
    }

    return true;
}

bool scaledBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu, int whichGPU)
{
    if (scale <= 1.0)
        return bilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, data_on_cpu, whichGPU);
    if (f == NULL) return false;

    spatialFWHM = float(max(0.25, min(100.0, spatialFWHM)));
    intensityFWHM = float(max(1.0e-8, min(1.0e8, intensityFWHM)));

    // convert FWHM input parameters to sigmas
    float sigma_d_sq = float(spatialFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_d_sq *= sigma_d_sq;
    float sigma_i_sq = float(intensityFWHM / (2.0 * sqrt(2.0 * log(2.0))));
    sigma_i_sq *= sigma_i_sq;

    float sigma_d_sq_inv = float(0.5 / sigma_d_sq);
    float sigma_i_sq_inv = float(0.5 / sigma_i_sq);

    // define filter window width by full width, tenth max
    int w = max(1, int(ceil(sqrt(2.0 * log(10.0) * sigma_d_sq))));

    cudaSetDevice(whichGPU);


    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Bf = 0;
    if (cudaMalloc((void**)&dev_Bf, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }
    blurFilter(dev_f, N_1, N_2, N_3, scale, 3, 0, false, whichGPU, -1, -1, dev_Bf);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);

    //bool useTexture = true;
    bool useTexture = false;

    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = NULL;
    cudaTextureObject_t Bf_data_txt = NULL;
    cudaArray* Bf_data_array = NULL;
    if (useTexture)
    {
        f_data_array = loadTexture(f_data_txt, dev_f, N, false, false);
        Bf_data_array = loadTexture(Bf_data_txt, dev_Bf, N, false, false);
        scaledBilateralFilterKernel_txt <<< dimGrid, dimBlock >>> (f_data_txt, Bf_data_txt, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    else
    {
        scaledBilateralFilterKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Bf, dev_Df, N, sigma_d_sq_inv, sigma_i_sq_inv, w);
    }
    cudaDeviceSynchronize();

    // Clean up
    if (useTexture)
    {
        cudaFreeArray(f_data_array);
        cudaDestroyTextureObject(f_data_txt);
        cudaFreeArray(Bf_data_array);
        cudaDestroyTextureObject(Bf_data_txt);
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
        pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    }
    if (dev_Df != 0)
    {
       cudaFree(dev_Df);
    }
    if (dev_Bf != 0)
    {
        cudaFree(dev_Bf);
    }

    return true;
}
//*/
