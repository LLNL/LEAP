////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for the voxel-driven backprojection
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "backprojectors_VD.cuh"
#include "cuda_utils.h"

//#define NUM_SLICES_PER_THREAD 1
#define NUM_SLICES_PER_THREAD 8

__global__ void modularBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    //*
    const float x = float(i) * T_f.x + startVals_f.x;
    const float y = float(j) * T_f.y + startVals_f.y;
    const float z = float(k) * T_f.z + startVals_f.z;
    //*/
    /*
    const double x = double(i) * double(T_f.x) + double(startVals_f.x);
    const double y = double(j) * double(T_f.y) + double(startVals_f.y);
    const double z = double(k) * double(T_f.z) + double(startVals_f.z);
    //*/

    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float T_x_inv = 1.0f / T_f.x;
    //const float T_x_over2 = 0.5f * T_f.x;

    const float T_v_inv = 1.0f / T_g.y;
    const float T_u_inv = 1.0f / T_g.z;

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        float L = float(iphi) + 0.5f;
        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        float3 r = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);
        const float R = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float D = -p_minus_c_dot_n / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);

        //const float sod = sqrtf(sourcePosition[0] * sourcePosition[0] + sourcePosition[1] * sourcePosition[1] + sourcePosition[2] * sourcePosition[2]);
        //const float backprojectionWeight = sod * sod / (R * R);
        const float sdd = sqrtf(p_minus_c.x * p_minus_c.x + p_minus_c.y * p_minus_c.y + p_minus_c.z * p_minus_c.z);
        const float backprojectionWeight = sdd * sdd / (R*R);

        //<p_minus_c + lineLength*r, u>
        //<p_minus_c + lineLength*r, v>
        const float u_val = (p_minus_c.x + D * r.x) * u_vec[0] + (p_minus_c.y + D * r.y) * u_vec[1] + (p_minus_c.z + D * r.z) * u_vec[2];
        const float v_val = (p_minus_c.x + D * r.x) * v_vec[0] + (p_minus_c.y + D * r.y) * v_vec[1] + (p_minus_c.z + D * r.z) * v_vec[2];

        const float u_arg = (u_val - startVals_g.z) * T_u_inv + 0.5f;
        const float v_arg = (v_val - startVals_g.y) * T_v_inv + 0.5f;

        val += tex3D<float>(g, u_arg, v_arg, L) * backprojectionWeight;
    }
    f[ind] = val * T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
}

__global__ void coneParallelBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float asin_tau_over_R = asin(tau / R);

    float vals[NUM_SLICES_PER_THREAD];
    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi = phis[l];
        //const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        //const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        const float s = cos_phi * y - sin_phi * x;
        const float s_arg = (s - startVals_g.z) * Tu_inv + 0.5f;

        float z_source_over_T_v = 0.0f;
        if (T_g.w != 0.0f)
        {
            const float alpha = asin(s / R) + asin_tau_over_R;
            z_source_over_T_v = ((phi + alpha) * T_g.w + startVals_g.w) * Tv_inv;
        }

        const float v_denom = sqrtf(R * R - s * s) - (cos_phi * x + sin_phi * y);
        const float backprojectionWeight = doWeight ? v_denom : R;

        const float v_phi_x_step_A = Tz_over_Tv / v_denom;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) / v_denom - v0_over_Tv + 0.5f;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, s_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight;
    }

    //const float scalar = doWeight ? T_f.x / R : T_f.x;
    const float scalar = T_f.x * T_f.y * T_f.z / (R*R*T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
    }
}

__global__ void fanBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool doWeight)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    float vals[NUM_SLICES_PER_THREAD];
    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    const float iv = (z - startVals_g.y) / T_g.y + 0.5f;
    const float T_x_over_2 = 0.5f * T_f.x;
    const float Tu_inv = 1.0f / T_g.z;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi = phis[l];
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        //const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
        //const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;
        const float backprojectionWeight = doWeight ? R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv : R_minus_x_dot_theta_inv;

        const float u_arg = ((cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv - startVals_g.z) * Tu_inv + 0.5f;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, u_arg, iv + float(k_offset), L) * backprojectionWeight;
    }

    //const float scalar = doWeight ? T_f.x * R * R : T_f.x * R;
    const float scalar = T_f.x * T_f.y / (T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
    }
}

__global__ void parallelBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
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

    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0;
        return;
    }

    const float Tu_inv = 1.0f / T_g.z;
    const float s_shift = -startVals_g.z * Tu_inv + 0.5f;

    float vals[NUM_SLICES_PER_THREAD];
    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float sin_phi = sinf(phis[l]);
        const float cos_phi = cosf(phis[l]);

        const float u_arg = (cos_phi * y - sin_phi * x) * Tu_inv + s_shift;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, u_arg, float(k + k_offset) + 0.5f, float(l) + 0.5f);
    }

    const float scalar = T_f.x * T_f.y / (T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
    }
}

__global__ void curvedConeBeamBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    float vals[NUM_SLICES_PER_THREAD];
    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi = phis[l];
        const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        const float dist_from_source_components_x = fabs(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabs(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source_inv = rsqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);

        const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        const float backprojectionWeight = dist_from_source_inv * dist_from_source_inv;

        const float u_arg = (atan((cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv) - startVals_g.z) * Tu_inv + 0.5f;

        const float v_phi_x_step_A = Tz_over_Tv * dist_from_source_inv;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * dist_from_source_inv - v0_over_Tv + 0.5f;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight;
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
    }
}

__global__ void coneBeamBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    float vals[NUM_SLICES_PER_THREAD];
    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi = phis[l];
        const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;

        const float u_arg = ((cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv - startVals_g.z) * Tu_inv + 0.5f;

        const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv + 0.5f;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight;
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
            f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
    }
}

__global__ void applyPolarWeight_vox(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= rsqrt(1.0f + v * v);
}

__global__ void applyInversePolarWeight_vox(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= sqrt(1.0f + v * v);
}


bool backproject_VD_fan(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_VD(g, f, params, data_on_cpu);
}

bool backproject_VD_cone(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_VD(g, f, params, data_on_cpu);
}

bool backproject_VD_coneParallel(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_VD(g, f, params, data_on_cpu);
}

bool backproject_VD_parallel(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_VD(g, f, params, data_on_cpu);
}

/*
bool backproject_VD(float *g, float *&f, parameters* params, bool data_on_cpu)
{
    return false;
}
//*/

//*
bool backproject_VD(float *g, float *&f, parameters* params, bool data_on_cpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;
    if (params->doWeightedBackprojection && params->helicalPitch != 0.0)
        return false;
    if (params->geometry == parameters::MODULAR)
        return backproject_VD_modular(g, f, params, data_on_cpu);

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
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
    
    float rFOVsq = params->rFOV()*params->rFOV();
    
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);
    bool applyInverseWeight = false;
    if (params->geometry == parameters::CONE)
        applyInverseWeight = true;
    else if (params->geometry == parameters::CONE_PARALLEL && params->doWeightedBackprojection == false)
        applyInverseWeight = true;

    if (applyInverseWeight)
    {
        applyInversePolarWeight_vox <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
    }

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, true);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z)/float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);
    if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection);
    }
    else if (params->geometry == parameters::CONE)
    {
        if (params->detectorType == parameters::FLAT)
            coneBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
        else
            curvedConeBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        coneParallelBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection);
    }
    else
        return false;

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
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_phis);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else if (applyInverseWeight)
    {
        applyPolarWeight_vox <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
        cudaStatus = cudaDeviceSynchronize();
    }

    return true;
}
//*/

bool backproject_VD_modular(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

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
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;


    bool doLinearInterpolation = true;

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);
    float* w_polar = NULL;

    /*
    if (modularbeamIsAxiallyAligned == true)
    {
        w_polar = setViewDependentPolarWeights(params);
        //applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, true, false);
        applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, false, false);
    }
    //*/

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, doLinearInterpolation);

    float rFOV_sq = params->rFOV() * params->rFOV();

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    modularBeamBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    /*
    if (modularbeamIsAxiallyAligned == true && data_on_cpu == false)
    {
        //applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, true, true);
        applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, false, true);
    }
    //*/

    if (data_on_cpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_sourcePositions);
    cudaFree(dev_moduleCenters);
    cudaFree(dev_rowVectors);
    cudaFree(dev_colVectors);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }
    if (w_polar != NULL)
        free(w_polar);

    return true;
}
