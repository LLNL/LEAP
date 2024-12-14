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
#include "log.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "backprojectors_VD.cuh"
#include "cuda_utils.h"
//#include "ray_weighting_cpu.h"
//#include "ray_weighting.cuh"

//#define NUM_SLICES_PER_THREAD 1
#define NUM_SLICES_PER_THREAD 8

__constant__ float d_q_helical;
__constant__ float d_v_min_inv;
__constant__ float d_v_max_inv;
__constant__ float d_weightFcnTransition;
__constant__ float d_weightFcnParameter;
__constant__ float d_phi_start;
__constant__ float d_phi_end;

__device__ float helicalConeWeight_vox(float v)
{
    const float abs_v_hat = (v >= 0.0f) ? v * d_v_max_inv : v * d_v_min_inv;

    if (abs_v_hat <= d_q_helical)
        return 1.0f;
    else if (abs_v_hat > 1.0f)
        return 0.0f;
    else if (abs_v_hat <= d_weightFcnTransition)
        return d_weightFcnParameter * (abs_v_hat - d_q_helical) * (abs_v_hat - d_q_helical) + 1.0f;
    else
        return -1.0f * d_weightFcnParameter * (abs_v_hat - 1.0f) * (abs_v_hat - 1.0f);
}

__global__ void modularBeamBackprojectorKernel_vox_stack(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    const float x = float(i) * T_f.x + startVals_f.x;
    const float y = float(j) * T_f.y + startVals_f.y;
    const float z = float(k) * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOV_sq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }

        //f[ind] = 0.0f;
        return;
    }

    const float T_v_inv = 1.0f / T_g.y;
    const float T_u_inv = 1.0f / T_g.z;

    const float u_ind_shift = -startVals_g.z * T_u_inv + 0.5f;
    const float v_ind_shift = -startVals_g.y * T_v_inv + 0.5f;

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        const float3 sourcePosition = make_float3(sourcePositions[3 * iphi + 0], sourcePositions[3 * iphi + 1], sourcePositions[3 * iphi + 2]);
        const float3 moduleCenter = make_float3(moduleCenters[3 * iphi + 0], moduleCenters[3 * iphi + 1], moduleCenters[3 * iphi + 2]);
        const float3 v_vec = make_float3(rowVectors[3 * iphi + 0], rowVectors[3 * iphi + 1], rowVectors[3 * iphi + 2]);
        const float3 u_vec = make_float3(colVectors[3 * iphi + 0], colVectors[3 * iphi + 1], colVectors[3 * iphi + 2]);

        float L = float(iphi) + 0.5f;
        const float3 detNormal = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
                                             u_vec.z * v_vec.x - u_vec.x * v_vec.z,
                                             u_vec.x * v_vec.y - u_vec.y * v_vec.x);

        const float3 p_minus_c = make_float3(sourcePosition.x - moduleCenter.x, sourcePosition.y - moduleCenter.y, sourcePosition.z - moduleCenter.z);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float p_minus_c_dot_u = p_minus_c.x * u_vec.x + p_minus_c.y * u_vec.y + p_minus_c.z * u_vec.z;
        const float p_minus_c_dot_v = p_minus_c.x * v_vec.x + p_minus_c.y * v_vec.y + p_minus_c.z * v_vec.z;
        
        if (fabs(detNormal.z) <= 0.00001f)
        {
            float3 r = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

            const float r_dot_d_inv = 1.0f / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
            const float D = -p_minus_c_dot_n * r_dot_d_inv;

            const float r_dot_u_0 = r.x * u_vec.x + r.y * u_vec.y + r.z * u_vec.z;
            const float r_dot_v_0 = r.x * v_vec.x + r.y * v_vec.y + r.z * v_vec.z;

            const float r_dot_u_inc = T_f.z * u_vec.z;
            const float r_dot_v_inc = T_f.z * v_vec.z;

            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float r_dot_u = r_dot_u_0 + k_offset * r_dot_u_inc;
                const float r_dot_v = r_dot_v_0 + k_offset * r_dot_v_inc;
                const float backprojectionWeight = p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n )*r_dot_d_inv*r_dot_d_inv;

                vals[k_offset] += tex3D<float>(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
            }
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                float3 r = make_float3(x - sourcePosition.x, y - sourcePosition.y, z+k_offset*T_f.z - sourcePosition.z);
                const float r_dot_d_inv = 1.0f / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
                const float D = -p_minus_c_dot_n * r_dot_d_inv;

                const float r_dot_u = r.x * u_vec.x + r.y * u_vec.y + r.z * u_vec.z;
                const float r_dot_v = r.x * v_vec.x + r.y * v_vec.y + r.z * v_vec.z;

                const float backprojectionWeight = p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n )*r_dot_d_inv*r_dot_d_inv;

                vals[k_offset] += tex3D<float>(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
            }
        }
    }

    const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
    }
}

// This function is not used anymore because modularBeamBackprojectorKernel_vox works faster
__global__ void modularBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, bool accum)
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

    const float x = float(i) * T_f.x + startVals_f.x;
    const float y = float(j) * T_f.y + startVals_f.y;
    const float z = float(k) * T_f.z + startVals_f.z;
    
    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float T_v_inv = 1.0f / T_g.y;
    const float T_u_inv = 1.0f / T_g.z;

    const float u_ind_shift = -startVals_g.z * T_u_inv + 0.5f;
    const float v_ind_shift = -startVals_g.y * T_v_inv + 0.5f;

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

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float p_minus_c_dot_u = p_minus_c.x * u_vec[0] + p_minus_c.y * u_vec[1] + p_minus_c.z * u_vec[2];
        const float p_minus_c_dot_v = p_minus_c.x * v_vec[0] + p_minus_c.y * v_vec[1] + p_minus_c.z * v_vec[2];

        float3 r = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);
        const float r_dot_d_inv = 1.0f / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
        const float D = -p_minus_c_dot_n * r_dot_d_inv;

        const float r_dot_u = r.x * u_vec[0] + r.y * u_vec[1] + r.z * u_vec[2];
        const float r_dot_v = r.x * v_vec[0] + r.y * v_vec[1] + r.z * v_vec[2];

        const float backprojectionWeight = p_minus_c_dot_n * sqrtf( D * D * (r_dot_u * r_dot_u + r_dot_v * r_dot_v) + p_minus_c_dot_n * p_minus_c_dot_n )*r_dot_d_inv*r_dot_d_inv;

        val += tex3D<float>(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
    }
    if (accum)
        f[ind] += val * T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    else
        f[ind] = val * T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);// *14.0f * 14.0f / (11.0f * 11.0f);
}

//#####################################################################################################################
//#####################################################################################################################
__global__ void coneParallelWeightedHelicalBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
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

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    const float asin_tau_over_R = asin(tau / R);

    float val = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi_cur = phis[l];
        //const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

        //const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        const float s = cos_phi * y - sin_phi * x;
        const float s_arg = (s - startVals_g.z) * Tu_inv + 0.5f;

        const float x_dot_theta = x * cos_phi + y * sin_phi;
        const float v_denom = sqrtf(R * R - s * s) - x_dot_theta;
        const float v_denom_inv = 1.0f / v_denom;

        const float alpha = asin(s / R) + asin_tau_over_R;
        const float z_source = (phi_cur + alpha) * T_g.w + startVals_g.w;

        const float v_val = (z - z_source) * v_denom_inv;

        const float centralWeight = helicalConeWeight_vox(v_val);
        if (centralWeight > 0.0f)
        {
            const float v_arg = (v_val - startVals_g.y) * Tv_inv + 0.5f;
            const float backprojectionWeight = doWeight ? R * sqrtf(1.0f + v_val*v_val) : R * R * v_denom_inv * sqrtf(1.0f + v_val*v_val);

            const float v_denom_conj = v_denom + 2.0f * x_dot_theta;
            const float v_denom_conj_inv = 1.0f / v_denom_conj;
            const float phi_cur_conj = phi_cur + PI;

            float sumWeights = 0.0f;

            const float v_val_shift = neg_twoPI_pitch * v_denom_inv;

            const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_val) * v_denom * neg_twoPI_pitch_inv : (v_max - v_val) * v_denom * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_val) * v_denom * neg_twoPI_pitch_inv : (v_max - v_val) * v_denom * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight_vox(v_val + iturn * v_val_shift);
            }

            const float alpha_conj = asin(-s / R) + asin_tau_over_R;
            const float v_val_conj = (z - ((phi_cur_conj + alpha_conj) * T_g.w + startVals_g.w)) * v_denom_conj_inv;
            const float v_val_shift_conj = neg_twoPI_pitch * v_denom_conj_inv;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight_vox(v_val_conj + iturn * v_val_shift_conj);

            val += tex3D<float>(g, s_arg, v_arg, L) * backprojectionWeight * centralWeight / (centralWeight + sumWeights);
        }
    }

    //const float scalar = doWeight ? T_f.x / R : T_f.x;
    const float scalar = T_f.x * T_f.y * T_f.z / (R * R * T_g.y * T_g.z);
    if (accum)
        f[ind] += val * scalar;
    else
        f[ind] = val * scalar;
}

__global__ void curvedConeBeamWeightedHelicalBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float atan_term = atan(2.0f * tau * R / (R * R - tau * tau));
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;
    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;

    float val = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float phi_cur = phis[l];
        //const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float z_source = (phi_cur * T_g.w + startVals_g.w);
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

        const float dist_from_source_components_x = fabs(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabs(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source_inv = rsqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);

        const float v_val = (z - z_source) * dist_from_source_inv;
        const float centralWeight = helicalConeWeight_vox(v_val);
        if (centralWeight > 0.0f)
        {
            const float L = (float)l + 0.5f;
            const float dist_from_source = 1.0f / dist_from_source_inv;

            const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

            //const float backprojectionWeight = dist_from_source_inv * dist_from_source_inv;
            const float backprojectionWeight = sqrtf(1.0f + v_val * v_val) * dist_from_source_inv;

            const float u_val = atan((cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv);
            const float u_arg = (u_val - startVals_g.z) * Tu_inv + 0.5f;
            const float v_arg = (v_val - startVals_g.y) * Tv_inv + 0.5f;

            // Calculate the View Redundancy Weight
            const float phi_cur_conj = phi_cur - 2.0f * u_val + atan_term + PI;
            const float cos_phi_conj = cos(phi_cur_conj);
            const float sin_phi_conj = sin(phi_cur_conj);
            const float dist_from_source_components_x_conj = fabs(R * cos_phi_conj + tau * sin_phi_conj - x);
            const float dist_from_source_components_y_conj = fabs(R * sin_phi_conj - tau * cos_phi_conj - y);
            const float dist_from_source_conj = sqrtf(dist_from_source_components_x_conj * dist_from_source_components_x_conj + dist_from_source_components_y_conj * dist_from_source_components_y_conj);
            const float dist_from_source_inv_conj = 1.0f / dist_from_source_conj;

            float sumWeights = 0.0f;

            const float v_val_shift = neg_twoPI_pitch * dist_from_source_inv;

            const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_val) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_val) * dist_from_source * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_val) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_val) * dist_from_source * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight_vox(v_val + iturn * v_val_shift);
            }

            const float v_val_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * dist_from_source_inv_conj;
            const float v_val_shift_conj = neg_twoPI_pitch * dist_from_source_inv_conj;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight_vox(v_val_conj + iturn * v_val_shift_conj);

            val += tex3D<float>(g, u_arg, v_arg, L) * backprojectionWeight * centralWeight / (centralWeight + sumWeights);
        }
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (R*T_g.y * T_g.z);
    if (accum)
        f[ind] += val * scalar;
    else
        f[ind] = val * scalar;
}

__global__ void coneBeamWeightedHelicalBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;
    const float atan_term = atan(2.0f * tau * R / (R * R - tau * tau));
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    float val = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float phi_cur = phis[l];
        //const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
        const float z_source = (phi_cur * T_g.w + startVals_g.w);
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float v_val = (z - z_source) * R_minus_x_dot_theta_inv;
        const float centralWeight = helicalConeWeight_vox(v_val);
        if (centralWeight > 0.0f)
        {
            const float L = (float)l + 0.5f;
            const float u_val = (cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv;
            const float u_arg = (u_val - startVals_g.z) * Tu_inv + 0.5f;
            const float v_arg = (v_val - startVals_g.y) * Tv_inv + 0.5f;

            //const float backprojectionWeight = sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
            const float backprojectionWeight = sqrtf(1.0f + u_val * u_val + v_val * v_val) * R_minus_x_dot_theta_inv;

            // Calculate the View Redundancy Weight
            const float phi_cur_conj = phi_cur - 2.0f * atan(u_val) + atan_term + PI;
            const float R_minus_x_dot_theta_conj = R - x * cos(phi_cur_conj) - y * sin(phi_cur_conj);
            const float R_minus_x_dot_theta_inv_conj = 1.0f / R_minus_x_dot_theta_conj;
            float sumWeights = 0.0f;

            const float v_val_shift = neg_twoPI_pitch * R_minus_x_dot_theta_inv;

            const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_val) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_val) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_val) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_val) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight_vox(v_val + iturn * v_val_shift);
            }

            const float v_val_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * R_minus_x_dot_theta_inv_conj;
            const float v_val_shift_conj = neg_twoPI_pitch * R_minus_x_dot_theta_inv_conj;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight_vox(v_val_conj + iturn * v_val_shift_conj);

            val += tex3D<float>(g, u_arg, v_arg, L) * backprojectionWeight * centralWeight / (centralWeight + sumWeights);
        }
    }

    const float scalar = T_f.x * T_f.y * T_f.z / (R*T_g.y * T_g.z);
    if (accum)
        f[ind] += val * scalar;
    else
        f[ind] = val * scalar;
}
//#####################################################################################################################
//#####################################################################################################################

__global__ void coneParallelBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }
        //f[ind] = 0.0f;
        return;
    }

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float asin_tau_over_R = asin(tau / R);

    float vals[NUM_SLICES_PER_THREAD];
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
        const float v_denom_inv = 1.0f / v_denom;
        const float backprojectionWeight = doWeight ? R : R*R * v_denom_inv;

        const float v_phi_x_step_A = Tz_over_Tv * v_denom_inv;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * v_denom_inv - v0_over_Tv + 0.5f;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            const float v_arg = (v_phi_x_first + k_offset * v_phi_x_step_A + v0_over_Tv - 0.5f)*T_g.y;
            vals[k_offset] += tex3D<float>(g, s_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0f + v_arg*v_arg);
        }
    }

    //const float scalar = doWeight ? T_f.x / R : T_f.x;
    const float scalar = T_f.x * T_f.y * T_f.z / (R*R*T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
    }
}

__global__ void curvedConeBeamBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }

        //f[ind] = 0.0f;
        return;
    }

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    float vals[NUM_SLICES_PER_THREAD];
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
        {
            const float v_arg = (v_phi_x_first + k_offset * v_phi_x_step_A + v0_over_Tv - 0.5f)*T_g.y;
            vals[k_offset] += tex3D<float>(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0 + v_arg * v_arg);
        }
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
    }
}

__global__ void coneBeamBackprojectorKernel_vox(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }
        //f[ind] = 0.0f;
        return;
    }

    const float cos_tilt = cos(tiltAngle);
    const float sin_tilt = sin(tiltAngle);

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    float vals[NUM_SLICES_PER_THREAD];
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

        //const float backprojectionWeight = sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
        const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;

        if (tiltAngle == 0.0f)
        {
            const float u_val = (cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv;
            const float u_arg = (u_val - startVals_g.z) * Tu_inv + 0.5f;

            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;
            const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv + 0.5f;
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float v_arg = (v_phi_x_first + k_offset * v_phi_x_step_A + v0_over_Tv - 0.5f) * T_g.y;
                vals[k_offset] += tex3D<float>(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0f + u_val * u_val + v_arg * v_arg);
            }
        }
        else
        {
            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float u_val = (x_dot_theta_perp * cos_tilt + (z+k_offset*T_f.z) * sin_tilt) * R_minus_x_dot_theta_inv;
                const float v_val = ((z + k_offset * T_f.z) * cos_tilt - x_dot_theta_perp * sin_tilt) * R_minus_x_dot_theta_inv;

                vals[k_offset] += tex3D<float>(g, (u_val - startVals_g.z) * Tu_inv + 0.5f, (v_val - startVals_g.y) * Tv_inv + 0.5f, L) * backprojectionWeight * sqrtf(1.0f + u_val * u_val + v_val * v_val);
            }
        }
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
    }
}

__global__ void fanBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool doWeight, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }
        //f[ind] = 0.0f;
        return;
    }

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    const float iv = (z - startVals_g.y) / T_g.y + 0.5f;
    const float Tu_inv = 1.0f / T_g.z;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float phi = phis[l];
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);
        const float u_val = (cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv;
        const float u_arg = (u_val - startVals_g.z) * Tu_inv + 0.5f;

        //const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
        //const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;
        const float backprojectionWeight = doWeight ? sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv : sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, u_arg, iv + float(k_offset), L) * backprojectionWeight;
    }

    //const float scalar = doWeight ? T_f.x * R * R : T_f.x * R;
    //const float scalar = T_f.x * T_f.y / (T_g.z);
    const float scalar = doWeight ? R * T_f.x * T_f.y / (T_g.z) : T_f.x * T_f.y / (T_g.z);
    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
    }
}

__global__ void parallelBeamBackprojectorKernel_vox(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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

    int numZ = min(NUM_SLICES_PER_THREAD, N_f.z - k);

    if (x * x + y * y > rFOVsq)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = 0.0f;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = 0.0f;
        }
        //f[ind] = 0.0;
        return;
    }

    const float Tu_inv = 1.0f / T_g.z;
    const float s_shift = -startVals_g.z * Tu_inv + 0.5f;

    float vals[NUM_SLICES_PER_THREAD];
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
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * scalar;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * scalar;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * scalar;
        }
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
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= rsqrtf(1.0f + v * v);
}

__global__ void applyInversePolarWeight_vox(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= sqrtf(1.0f + v * v);
}

bool backproject_VD(float *g, float *&f, parameters* params, bool data_on_cpu)
{
    return backproject_VD(g, f, params, data_on_cpu, data_on_cpu);
}

bool backproject_VD(float *g, float *&f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;
    if (params->geometry == parameters::MODULAR)
        return backproject_VD_modular(g, f, params, data_on_cpu, volume_on_cpu, accum);

    LOG(logDEBUG, "backprojectors_VD", "backproject_VD") << "Performing voxel-based backprojection..." << std::endl;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (volume_on_cpu)
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

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = NULL;

    /*
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    bool applyInverseWeight = false;
    if (params->geometry == parameters::CONE)
        applyInverseWeight = true;
    else if (params->geometry == parameters::CONE_PARALLEL && params->doWeightedBackprojection == true)
        applyInverseWeight = true;

    if (applyInverseWeight)
    {
        applyInversePolarWeight_vox <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
    }
    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, true);
    //*/

    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, N_g, params->doExtrapolation, true);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, true);
    
    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z)/float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);
    if (params->helicalPitch != 0.0 && params->doWeightedBackprojection == true && (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL))
    {
        float q_helical = float(params->helicalFBPWeight);
        float weightFcnParameter = float(-2.0 / ((1.0 - q_helical) * (1.0 - q_helical)));
        float weightFcnTransition = float((q_helical + 1.0) / 2.0);
        float v_min_inv = float((params->v(0) - 0.5 * params->pixelHeight) / params->sdd);
        v_min_inv = float(1.0 / v_min_inv);
        float v_max_inv = float((params->v(params->numRows - 1) + 0.5 * params->pixelHeight) / params->sdd);
        v_max_inv = float(1.0 / v_max_inv);
        float phi_start = params->get_phi_start();
        float phi_end = params->get_phi_end();

        cudaMemcpyToSymbol(d_q_helical, &q_helical, sizeof(float));
        cudaMemcpyToSymbol(d_v_min_inv, &v_min_inv, sizeof(float));
        cudaMemcpyToSymbol(d_v_max_inv, &v_max_inv, sizeof(float));
        cudaMemcpyToSymbol(d_weightFcnTransition, &weightFcnTransition, sizeof(float));
        cudaMemcpyToSymbol(d_weightFcnParameter, &weightFcnParameter, sizeof(float));
        cudaMemcpyToSymbol(d_phi_start, &phi_start, sizeof(float));
        cudaMemcpyToSymbol(d_phi_end, &phi_end, sizeof(float));

        if (params->geometry == parameters::CONE_PARALLEL)
        {
            coneParallelWeightedHelicalBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
        }
        else if (params->detectorType == params->FLAT)
        {
            coneBeamWeightedHelicalBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
        else
        {
            curvedConeBeamWeightedHelicalBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
    }
    else if (params->geometry == parameters::CONE)
    {
        if (params->detectorType == parameters::FLAT)
            coneBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        else
            curvedConeBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        coneParallelBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
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
    if (volume_on_cpu)
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
    }
    if (volume_on_cpu)
    {
        if (dev_f != 0)
            cudaFree(dev_f);
    }
    /*
    else if (applyInverseWeight)
    {
        applyPolarWeight_vox <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
        cudaStatus = cudaDeviceSynchronize();
    }
    //*/

    return true;
}

bool backproject_VD_modular(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_VD_modular(g, f, params, data_on_cpu, data_on_cpu);
}

bool backproject_VD_modular(float* g, float*& f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    LOG(logDEBUG, "backprojectors_VD", "backproject_VD_modular") << "Performing voxel-based backprojection..." << std::endl;

    //printf("source = %f, %f, %f\n", params->sourcePositions[0], params->sourcePositions[1], params->sourcePositions[2]);
    //printf("detector = %f, %f, %f\n", params->moduleCenters[0], params->moduleCenters[1], params->moduleCenters[2]);

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (volume_on_cpu)
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

    bool doLinearInterpolation = true;
    bool modularbeamIsAxiallyAligned = params->modularbeamIsAxiallyAligned();

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = NULL;
    /*
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;
    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, doLinearInterpolation);
    //*/
    //*
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, N_g, params->doExtrapolation, doLinearInterpolation);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, doLinearInterpolation);
    //*/

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
    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, doLinearInterpolation);
    //*/

    float rFOV_sq = params->rFOV() * params->rFOV();

    // Call Kernel
    /*
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    modularBeamBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
    //*/

    //*
    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z)/float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);
    modularBeamBackprojectorKernel_vox_stack <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
    //*/

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

    if (volume_on_cpu)
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
    }
    if (volume_on_cpu)
    {
        if (dev_f != 0)
            cudaFree(dev_f);
    }
    if (w_polar != NULL)
        free(w_polar);

    return true;
}
