////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for gpu-based sensitivity calculation (P*1)
////////////////////////////////////////////////////////////////////////////////

#include "sensitivity.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.h"

__global__ void parallelBeamSensitivityKernel(int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float u_min = startVals_g.z;
    const float u_max = float(N_g.z - 1) * T_g.z + u_min;

    const float scalar = T_f.x * T_f.x / T_g.z;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float cos_phi = cos(phis[l]);
        const float sin_phi = sin(phis[l]);

        const float u_arg = -sin_phi * x + cos_phi * y;
        if (u_min <= u_arg && u_arg <= u_max)
            val += 1.0f;
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
}

__global__ void fanBeamSensitivityKernel(int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float u_min = startVals_g.z;
    const float u_max = float(N_g.z - 1) * T_g.z + u_min;

    const float scalar = (T_f.x * T_f.x / (R * T_g.z)) * R;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float cos_phi = cos(phis[l]);
        const float sin_phi = sin(phis[l]);

        const float v_denom_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
        const float u_arg = (-sin_phi * x + cos_phi * y + tau) * v_denom_inv;
        if (u_min <= u_arg && u_arg <= u_max)
            val += sqrt(1.0f + u_arg * u_arg) * v_denom_inv;
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
}

__global__ void coneBeamSensitivityKernel(int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float u_min = startVals_g.z;
    const float u_max = float(N_g.z - 1) * T_g.z + u_min;

    const float v_min = startVals_g.y;
    const float v_max = float(N_g.y - 1) * T_g.y + startVals_g.y;

    const float scalar = (T_f.x * T_f.x / (R * T_g.z)) * (T_f.z / (R * T_g.y)) * R * R;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float cos_phi = cos(phis[l]);
        const float sin_phi = sin(phis[l]);
        const float z_source = phis[l] * T_g.w + startVals_g.w;

        const float v_denom_inv = 1.0 / (R - x * cos_phi - y * sin_phi);
        const float u_arg = (-sin_phi * x + cos_phi * y + tau) * v_denom_inv;
        const float v_arg = (z - z_source) * v_denom_inv;
        if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
            val += sqrt(1.0f + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
}

bool sensitivity_gpu(float*& f, parameters* params, bool data_on_cpu)
{
    if (f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_f = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
        }
    }
    else
        dev_f = f;

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float rFOVsq = params->rFOV() * params->rFOV();

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::CONE)
    {
        coneBeamSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (data_on_cpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
    cudaFree(dev_phis);

    if (data_on_cpu)
    {
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

