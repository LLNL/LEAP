////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CUDA-based thresholded median filter and 3D low pass filter calculations
////////////////////////////////////////////////////////////////////////////////
#include "noise_filters.cuh"

#include <math.h>

#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void medianFilterKernel(float* f, float* f_filtered, int3 N, float threshold, int sliceStart, int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[i * N.y * N.z + j * N.z + k] = 0.0f;
        return;
    }

    float v[27];
    int ind = 0;
    for (int di = -1; di <= 1; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -1; dj <= 1; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -1; dk <= 1; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f[i_shift * N.y * N.z + j_shift * N.z + k_shift];
                ind += 1;
            }
        }
    }
    const float curVal = v[13];

    // bubble-sort for first 14 samples
    for (int i = 0; i < 14; i++)
    {
        for (int j = i + 1; j < 27; j++)
        {
            if (v[i] > v[j])
            {  // swap?
                const float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // fabs(curVal-v[13])/fabs(curVal) > threshold
    if (fabs(curVal - v[13]) >= threshold * fabs(v[13]))
        f_filtered[i * N.y * N.z + j * N.z + k] = v[13];
    else
        f_filtered[i * N.y * N.z + j * N.z + k] = curVal;
}

__global__ void BlurFilterKernel(float* f, float* f_filtered, int3 N, float FWHM, const int sliceStart, const int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[i * N.y * N.z + j * N.z + k] = 0.0f;
        return;
    }

    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0;
    float sum = 0.0;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));

                const float theWeight = 0.5f +
                    0.5f * cosf(3.141592653589793f* min(sqrtf(float(di * di + dj * dj + dk * dk)) * denom, 1.0f));

                if (theWeight > 0.0001f)
                {
                    val += theWeight * f[i_shift * N.y * N.z + j_shift * N.z + k_shift];
                    sum += theWeight;
                }
            }
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void BlurFilter2DKernel(float* f, float* f_filtered, int3 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;

    for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        //const float j_dist_sq = float((j - j_shift) * (j - j_shift));
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            //const float k_dist_sq = float((k - k_shift) * (k - k_shift));

            //const float theWeight = exp(-denom * (j_dist_sq + k_dist_sq));

            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(sqrtf(float(dj * dj + dk * dk)) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                val += theWeight * f[i * N.y * N.z + j_shift * N.z + k_shift];
                sum += theWeight;
            }
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void BlurFilter1DKernel(float* f, float* f_filtered, int3 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0;
    float sum = 0.0;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));

        //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
        const float theWeight = 0.5f +
            0.5f * cosf(3.141592653589793f * min(fabsf(di) * denom, 1.0f));

        if (theWeight > 0.0001f)
        {
            val += theWeight * f[i_shift * N.y * N.z + j * N.z + k];
            sum += theWeight;
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    if (f == NULL) return false;

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    if (numDims == 1)
        BlurFilter1DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else if (numDims == 2)
        BlurFilter2DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else
        BlurFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM, sliceStart, sliceEnd);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * N_1 * N_2 * N_3, cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    if (f == NULL) return false;

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    medianFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, threshold, sliceStart, sliceEnd);
    // medianFilterKernel(float* f, float* f_filtered, int4 N, float threshold)

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * N_1*N_2*N_3, cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}
