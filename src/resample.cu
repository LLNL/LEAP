////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// GPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "leap_defines.h"
#include "resample.cuh"

__device__ float bumpFcn(float W, float delay, int l)
{
    //delay *= -1.0;
    //int L = int(floor(W));
    if (fabs(float(l) - delay) > W)
        return 0.0f;
    else
    {
        const float h = cos(0.5f * PI * (float(l) - delay) / W);
        return h * h;
    }
}

__global__ void downSampleKernel(cudaTextureObject_t I, const int3 N, float* I_dn, const int3 N_dn, const float3 L)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_dn.x || j >= N_dn.y || k >= N_dn.z)
        return;

    float x = (i - 0.5f * float(N_dn.x - 1)) * L.x + 0.5f * float(N.x - 1);
    float y = (j - 0.5f * float(N_dn.y - 1)) * L.y + 0.5f * float(N.y - 1);
    float z = (k - 0.5f * float(N_dn.z - 1)) * L.z + 0.5f * float(N.z - 1);

    const float ii = int(floor(0.5 + x));
    const float jj = int(floor(0.5 + y));
    const float kk = int(floor(0.5 + z));

    const float delay_0 = x - float(ii);
    const float delay_1 = y - float(jj);
    const float delay_2 = z - float(kk);

    /*
    float x = (i - 0.5 * float(N_dn[0] - 1)) * factors[0];
    int ii = int(floor(0.5 + x + u_c));
    float delay_0 = x + u_c - float(ii);
    //*/

    const int filterWidth_0 = int(floor(L.x));
    const int filterWidth_1 = int(floor(L.y));
    const int filterWidth_2 = int(floor(L.z));
    float val = 0.0f;
    float accum = 0.0f;
    for (int l_0 = -filterWidth_0; l_0 <= filterWidth_0; l_0++)
    {
        const float h_0 = bumpFcn(L.x, delay_0, l_0);
        if (h_0 == 0.0f)
            continue;
        for (int l_1 = -filterWidth_1; l_1 <= filterWidth_1; l_1++)
        {
            const float h_1 = bumpFcn(L.y, delay_1, l_1);
            if (h_0 == 0.0f)
                continue;
            for (int l_2 = -filterWidth_2; l_2 <= filterWidth_2; l_2++)
            {
                const float h_2 = bumpFcn(L.z, delay_2, l_2);
                accum += h_0 * h_1 * h_2;
                val += tex3D<float>(I, kk+l_2, jj+l_1, ii+l_0) * h_0 * h_1 * h_2;
            }
        }
    }

    uint64 ind = uint64(i) * uint64(N_dn.y * N_dn.z) + uint64(j * N_dn.z + k);
    //I_dn[ind] = tex3D<float>(I, z, y, x);
    I_dn[ind] = val / accum;
}

__global__ void upSampleKernel(cudaTextureObject_t I, const int3 N, float* I_up, const int3 N_up, const float3 L)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_up.x || j >= N_up.y || k >= N_up.z)
        return;

    float x = (i - 0.5f * float(N_up.x - 1)) / L.x + 0.5f * float(N.x - 1) + 0.5f;
    float y = (j - 0.5f * float(N_up.y - 1)) / L.y + 0.5f * float(N.y - 1) + 0.5f;
    float z = (k - 0.5f * float(N_up.z - 1)) / L.z + 0.5f * float(N.z - 1) + 0.5f;

    uint64 ind = uint64(i) * uint64(N_up.y * N_up.z) + uint64(j * N_up.z + k);
    I_up[ind] = tex3D<float>(I, z, y, x);
}

bool downSample(float* I, int* N, float* I_dn, int* N_dn, float* factors, int whichGPU)
{
    if (I == NULL || N == NULL || I_dn == NULL || N_dn == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;

    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;

    float* dev_I = I;
    float* dev_I_dn = I_dn;

    int3 N_g = make_int3(N_dn[0], N_dn[1], N_dn[2]);
    int3 N_f = make_int3(N[0], N[1], N[2]);
    float3 L = make_float3(factors[0], factors[1], factors[2]);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_I, N_f, true, false);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    downSampleKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_f, dev_I_dn, N_g, L);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);

    return true;
}

bool upSample(float* I, int* N, float* I_up, int* N_up, float* factors, int whichGPU)
{
    if (I == NULL || N == NULL || I_up == NULL || N_up == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;

    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;

    float* dev_I = I;
    float* dev_I_up = I_up;

    int3 N_g = make_int3(N_up[0], N_up[1], N_up[2]);
    int3 N_f = make_int3(N[0], N[1], N[2]);
    float3 L = make_float3(factors[0], factors[1], factors[2]);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_I, N_f, true, true);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    upSampleKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_f, dev_I_up, N_g, L);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);

    return true;
}
