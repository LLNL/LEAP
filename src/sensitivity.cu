////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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
//#include "device_launch_parameters.h"
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
    //const float z = k * T_f.z + startVals_f.z;

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

__global__ void modularBeamSensitivityKernel(const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, const int volumeDimensionOrder)
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

    const float scalar = (T_f.x * T_f.x / T_g.z) * (T_f.z / T_g.y);

    float val = 0.0;
    /*
    for (int l = 0; l < N_g.x; l++)
    {
        float* sourcePosition = &sourcePositions[3 * l];
        float* moduleCenter = &moduleCenters[3 * l];
        float* v_vec = &rowVectors[3 * l];
        float* u_vec = &colVectors[3 * l];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        const float3 sourceToVoxel = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);

        const float l_phi_denom = (fabs(sourceToVoxel.x) >= fabs(sourceToVoxel.y)) ? fabs(sourceToVoxel.x) : fabs(sourceToVoxel.y);
        const float l_phi = sqrtf(sourceToVoxel.x * sourceToVoxel.x + sourceToVoxel.y * sourceToVoxel.y + sourceToVoxel.z * sourceToVoxel.z) / l_phi_denom;

        const float3 c_minus_s = make_float3(moduleCenter[0] - sourcePosition[0], moduleCenter[1] - sourcePosition[1], moduleCenter[2] - sourcePosition[2]);
        //const float D = sqrtf(c_minus_s.x * c_minus_s.x + c_minus_s.y * c_minus_s.y);

        //const float phi = atan2(-c_minus_s.y, -c_minus_s.x);
        //const float cos_phi = cos(phi);
        //const float sin_phi = sin(phi);

        //const float R = sourcePosition[0] * cos_phi + sourcePosition[1] * sin_phi;
        const float D_sq = c_minus_s.x * c_minus_s.x + c_minus_s.y * c_minus_s.y + c_minus_s.z * c_minus_s.z;
        //const float one_over_D = rsqrtf(D_sq);
        //const float scalar = (T_f.x * T_f.x / (one_over_D * T_g.z)) * (T_f.z / (one_over_D * T_g.y));

        const float c_minus_s_dot_u = c_minus_s.x * u_vec[0] + c_minus_s.y * u_vec[1] + c_minus_s.z * u_vec[2];
        const float c_minus_s_dot_v = c_minus_s.x * v_vec[0] + c_minus_s.y * v_vec[1] + c_minus_s.z * v_vec[2];
        //const float c_minus_s_dot_n = c_minus_s.x * detNormal.x + c_minus_s.y * detNormal.y + c_minus_s.z * detNormal.z;

        //const float denom = (x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z;
        //const float t_C = c_minus_s_dot_n / denom;
        const float t_C = (c_minus_s.x * detNormal.x + c_minus_s.y * detNormal.y + c_minus_s.z * detNormal.z) / ((x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z);

        const float u_arg = t_C * ((x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;
        const float v_arg = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;

        if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
            val += l_phi * t_C * t_C;
            //val += sqrtf(D_sq + u_arg * u_arg + v_arg * v_arg) * one_over_D * t_C * t_C;
    }
    //*/

    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        float3 r = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float r_dot_d_inv = 1.0f / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
        const float D = -p_minus_c_dot_n * r_dot_d_inv;

        const float p_minus_c_dot_u = p_minus_c.x * u_vec[0] + p_minus_c.y * u_vec[1] + p_minus_c.z * u_vec[2];
        const float p_minus_c_dot_v = p_minus_c.x * v_vec[0] + p_minus_c.y * v_vec[1] + p_minus_c.z * v_vec[2];

        const float r_dot_u = r.x * u_vec[0] + r.y * u_vec[1] + r.z * u_vec[2];
        const float r_dot_v = r.x * v_vec[0] + r.y * v_vec[1] + r.z * v_vec[2];

        const float u_val = p_minus_c_dot_u + D * r_dot_u;
        const float v_val = p_minus_c_dot_v + D * r_dot_v;

        //const float num = p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n );
        //const float backprojectionWeight = p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n )*r_dot_d_inv*r_dot_d_inv;

        //const float u_arg = (u_val - startVals_g.z) * T_u_inv + 0.5f;
        //const float v_arg = (v_val - startVals_g.y) * T_v_inv + 0.5f;
        //val += tex3D<float>(g, u_arg, v_arg, L) * backprojectionWeight;

        if (u_min <= u_val && u_val <= u_max && v_min <= v_val && v_val <= v_max)
            val += p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n )*r_dot_d_inv*r_dot_d_inv;
    }

    if (val == 0.0f)
        val = 1.0f;
    else
        val *= scalar;
    f[ind] = val;

    /*
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

        const float v_denom_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
        const float u_arg = (-sin_phi * x + cos_phi * y + tau) * v_denom_inv;
        const float v_arg = (z - z_source) * v_denom_inv;
        if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
            val += sqrtf(1.0f + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
    //*/
}

__global__ void coneBeamSensitivityKernel(const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, const bool isCurved)
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

        if (isCurved)
        {
            const float u_denom_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
            const float u_arg = atan(-sin_phi * x + cos_phi * y + tau) * u_denom_inv;

            const float dist_from_source_components_x = fabs(R * cos_phi + tau * sin_phi - x);
            const float dist_from_source_components_y = fabs(R * sin_phi - tau * cos_phi - y);
            const float v_denom_inv = rsqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);
            const float v_arg = (z - z_source) * v_denom_inv;
            if (T_g.w == 0.0f)
            {
                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                    val += sqrtf(1.0f + v_arg * v_arg) * v_denom_inv * v_denom_inv;
            }
            else
            {
                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                    val += sqrtf(1.0f + v_arg * v_arg) * v_denom_inv * v_denom_inv;
            }
        }
        else
        {
            const float v_denom_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
            const float u_arg = (-sin_phi * x + cos_phi * y + tau) * v_denom_inv;
            const float v_arg = (z - z_source) * v_denom_inv;
            if (T_g.w == 0.0f)
            {
                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                    val += sqrtf(1.0f + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
            }
            else
            {
                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                    val += sqrtf(1.0f + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
            }
        }
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
}

__global__ void coneParallelSensitivityKernel(const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, const bool isCurved)
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

    const float scalar = (T_f.x * T_f.x / (T_g.z)) * (T_f.z / (R * T_g.y)) * R;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float cos_phi = cos(phis[l]);
        const float sin_phi = sin(phis[l]);

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x;
        const float x_dot_theta = x * cos_phi + y * sin_phi;
        const float alpha = asin(x_dot_theta_perp / R) + asin(tau / R);

        const float z_source = (phis[l] + alpha) * T_g.w + startVals_g.w;

        const float v_denom_inv = 1.0f / (sqrtf(R * R - x_dot_theta_perp * x_dot_theta_perp) - x_dot_theta);
        //const float v_denom_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
        const float u_arg = x_dot_theta_perp;
        const float v_arg = (z - z_source) * v_denom_inv;
        if (T_g.w == 0.0f)
        {
            if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                val += sqrtf(1.0f + v_arg * v_arg) * v_denom_inv;
        }
        else
        {
            if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                val += sqrtf(1.0f + v_arg * v_arg) * v_denom_inv;
        }
    }
    if (val == 0.0f)
        val = 1.0f / scalar;
    f[ind] = val * scalar;
}

bool sensitivity_gpu(float*& f, parameters* params, bool data_on_cpu)
{
    if (f == NULL || params == NULL || params->allDefined() == false)
        return false;
    if (params->geometry == parameters::MODULAR)
        return sensitivity_modular_gpu(f, params, data_on_cpu);

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
        coneBeamSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, bool(params->detectorType == parameters::CURVED));
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        coneParallelSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, bool(params->detectorType == parameters::CURVED));
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

bool sensitivity_modular_gpu(float*& f, parameters* params, bool data_on_cpu)
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

    float* dev_sourcePositions = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_sourcePositions, 3 * params->numAngles * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_sourcePositions, params->sourcePositions, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(sourcePositions) failed!\n");

    float* dev_moduleCenters = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_moduleCenters, 3 * params->numAngles * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_moduleCenters, params->moduleCenters, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(moduleCenters) failed!\n");

    float* dev_rowVectors = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_rowVectors, 3 * params->numAngles * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_rowVectors, params->rowVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(rowVectors) failed!\n");

    float* dev_colVectors = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_colVectors, 3 * params->numAngles * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_colVectors, params->colVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(colVectors) failed!\n");

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    modularBeamSensitivityKernel <<< dimGrid, dimBlock >>> (N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);

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
    cudaFree(dev_sourcePositions);
    cudaFree(dev_moduleCenters);
    cudaFree(dev_rowVectors);
    cudaFree(dev_colVectors);

    if (data_on_cpu)
    {
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

