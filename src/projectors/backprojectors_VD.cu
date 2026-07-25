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
__constant__ float d_v_min;
__constant__ float d_v_max;
__constant__ float d_v_min_inv;
__constant__ float d_v_max_inv;
__constant__ float d_v_abs_max_inv;
__constant__ float d_weightFcnTransition;
__constant__ float d_weightFcnParameter;
__constant__ float d_phi_start;
__constant__ float d_phi_end;
__constant__ int d_numRowsExtrapolate;
__constant__ float d_cos_tilt;
__constant__ float d_sin_tilt;
__constant__ float d_cos_pitch;
__constant__ float d_sin_pitch;
__constant__ float3 d_n_vec_0;
__constant__ float3 d_u_vec_0;
__constant__ float3 d_v_vec_0;
__constant__ float d_R_tau;
__constant__ float2 d_zFOV;
__constant__ bool d_doDBP;

__device__ inline float dot3(const float3 a, const float3 b)
{
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ inline float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        fmaf(a.y, b.z, -a.z * b.y),
        fmaf(a.z, b.x, -a.x * b.z),
        fmaf(a.x, b.y, -a.y * b.x)
    );
}

__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }


__device__ __forceinline__ float helicalConeWeight_vox(const float v)
{
    /*
    const float abs_v_hat = (v >= 0.0f) ? v * d_v_max_inv : v * d_v_min_inv;

    if (abs_v_hat <= d_q_helical)
        return 1.0f;
    else if (abs_v_hat > 1.0f)
        return 0.0f;
    else if (abs_v_hat <= d_weightFcnTransition)
        return d_weightFcnParameter * (abs_v_hat - d_q_helical) * (abs_v_hat - d_q_helical) + 1.0f;
    else
        return -1.0f * d_weightFcnParameter * (abs_v_hat - 1.0f) * (abs_v_hat - 1.0f);
    //*/

    //*
    if (d_v_min <= v && v <= d_v_max)
        return 1.0f - fabsf(v)*d_v_abs_max_inv;
    else
        return 0.0f;
    //*/
}

__device__ __forceinline__ float sum_x_plus_i_y(float x, float y, int p, int q)
{
    return fmaf((p+q)*0.5f, y, x) * fmaxf(0.0f, float(q - p + 1));
}

__device__ __forceinline__ float helicalConeWeight_vox_sum(const float x, const float y, const int M, const int N)
{
    // S = sum_{i=M..N} (1 - |x + i*y|) in O(1).
    // Assumes M <= N.
    /*
    float retVal = 0.0f;
    for (int i = M; i <= N; i++)
        retVal += helicalConeWeight_vox(x+i*y);
    return retVal;
    //*/

    //*
    const float L = (float)(N - M + 1);
    if (y > 0.0f)
    {
        const int B = (int)ceilf(-x / y) - 1;
        return L + (sum_x_plus_i_y(x, y, M, imin(B, N)) - sum_x_plus_i_y(x, y, imax(B + 1, M), N)) * d_v_abs_max_inv;
    }
    else
    {
        const int A = (int)floorf(-x / y) + 1;
        return L + (sum_x_plus_i_y(x, y, imax(A, M), N) - sum_x_plus_i_y(x, y, M, imin(A - 1, N))) * d_v_abs_max_inv;
    }
    //*/
}

__global__ void modularBeamBackprojectorKernel_vox_stack(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, bool accum)
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
        
        if (fabsf(detNormal.z) <= 0.00001f)
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

                vals[k_offset] += TEX3D(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
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

                vals[k_offset] += TEX3D(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
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
__global__ void modularBeamBackprojectorKernel_vox(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, bool accum)
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

        val += TEX3D(g, (p_minus_c_dot_u + D * r_dot_u) * T_u_inv + u_ind_shift, (p_minus_c_dot_v + D * r_dot_v) * T_v_inv + v_ind_shift, L) * backprojectionWeight;
    }
    if (accum)
        f[ind] += val * T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    else
        f[ind] = val * T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);// *14.0f * 14.0f / (11.0f * 11.0f);
}

//#####################################################################################################################
//#####################################################################################################################
__global__ void coneParallelWeightedHelicalBackprojectorKernel_vox(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
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

            const int N_turns_below = max(int(ceilf((d_phi_start - phi_cur) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi_cur) * twoPI_inv)), int(floorf(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight_vox(v_val + iturn * v_val_shift);
            }

            //const float alpha_conj = asin(-s / R) + asin_tau_over_R;
            const float alpha_conj = -alpha + 2.0f*asin_tau_over_R;
            const float v_val_conj = (z - ((phi_cur_conj + alpha_conj) * T_g.w + startVals_g.w)) * v_denom_conj_inv;
            const float v_val_shift_conj = neg_twoPI_pitch * v_denom_conj_inv;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * v_denom_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight_vox(v_val_conj + iturn * v_val_shift_conj);

            val += TEX3D(g, s_arg, v_arg, L) * backprojectionWeight * centralWeight / (centralWeight + sumWeights);
        }
    }

    //const float scalar = doWeight ? T_f.x / R : T_f.x;
    const float scalar = T_f.x * T_f.y * T_f.z / (R * R * T_g.y * T_g.z);
    if (accum)
        f[ind] += val * scalar;
    else
        f[ind] = val * scalar;
}

__global__ void curvedConeBeamWeightedHelicalBackprojectorKernel_vox(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

        const float dist_from_source_components_x = fabsf(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabsf(R * sin_phi - tau * cos_phi - y);
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
            const float dist_from_source_components_x_conj = fabsf(R * cos_phi_conj + tau * sin_phi_conj - x);
            const float dist_from_source_components_y_conj = fabsf(R * sin_phi_conj - tau * cos_phi_conj - y);
            const float dist_from_source_conj = sqrtf(dist_from_source_components_x_conj * dist_from_source_components_x_conj + dist_from_source_components_y_conj * dist_from_source_components_y_conj);
            const float dist_from_source_inv_conj = 1.0f / dist_from_source_conj;

            float sumWeights = 0.0f;

            const float v_val_shift = neg_twoPI_pitch * dist_from_source_inv;

            const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_val) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_val) * dist_from_source * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_val) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_val) * dist_from_source * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceilf((d_phi_start - phi_cur) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi_cur) * twoPI_inv)), int(floorf(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight_vox(v_val + iturn * v_val_shift);
            }

            const float v_val_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * dist_from_source_inv_conj;
            const float v_val_shift_conj = neg_twoPI_pitch * dist_from_source_inv_conj;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight_vox(v_val_conj + iturn * v_val_shift_conj);

            val += TEX3D(g, u_arg, v_arg, L) * backprojectionWeight * centralWeight / (centralWeight + sumWeights);
        }
    }

    //const float scalar = T_f.x * R * R;
    const float scalar = T_f.x * T_f.y * T_f.z / (R*T_g.y * T_g.z);
    if (accum)
        f[ind] += val * scalar;
    else
        f[ind] = val * scalar;
}
//#####################################################################################################################
//#####################################################################################################################

__global__ void coneParallelBackprojectorKernel_vox(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
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
            vals[k_offset] += TEX3D(g, s_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0f + v_arg*v_arg);
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

__global__ void curvedConeBeamBackprojectorKernel_vox(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

        const float dist_from_source_components_x = fabsf(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabsf(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source_inv = rsqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);

        const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

        const float backprojectionWeight = dist_from_source_inv * dist_from_source_inv;

        const float u_arg = (atan((cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv) - startVals_g.z) * Tu_inv + 0.5f;

        const float v_phi_x_step_A = Tz_over_Tv * dist_from_source_inv;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * dist_from_source_inv - v0_over_Tv + 0.5f;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            const float v_arg = (v_phi_x_first + k_offset * v_phi_x_step_A + v0_over_Tv - 0.5f)*T_g.y;
            vals[k_offset] += TEX3D(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0 + v_arg * v_arg);
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

__global__ void coneBeamBackprojectorKernel_rot_vox_slab(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* __restrict__ f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* __restrict__ phis, const int volumeDimensionOrder, bool accum)
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
    const float z_0 = k * T_f.z + startVals_f.z;

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
        return;
    }

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;
    const float v_min = (-0.5f-float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;
    const float v_max = (float(N_g.y)-0.5f+float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;

    const float u_length = T_g.z*(N_g.z-1);// + 0.5f*T_g.z;

    const float y_dot_n = R*d_n_vec_0.x - tau*d_n_vec_0.y;
    const float y_dot_u = R*d_u_vec_0.x - tau*d_u_vec_0.y;
    const float y_dot_v = R*d_v_vec_0.x - tau*d_v_vec_0.y;

    const float z_f = z_0 + (numZ-1)*T_f.z;
    const float football_param = (d_R_tau-sqrtf(x * x + y * y))/d_R_tau;
    if (d_doDBP || (d_zFOV.x != d_zFOV.y && (d_zFOV.y*football_param < z_f || z_0 < d_zFOV.x*football_param)))
    {
        const float psi = atan(-tau/R);

        for (int iphi = 0; iphi < N_g.x; iphi++)
        {
            const float iphi_plus_half = float(iphi) + 0.5f;
            const float phi = phis[iphi];
            float cos_phi, sin_phi;
            __sincosf(phi, &sin_phi, &cos_phi);

            const float u_0 = startVals_g.z;
            const float v_0 = startVals_g.y;

            // fmaf(x, y, z) = x*y + z
            const float x_phi = x*cos_phi + y*sin_phi; // 3
            const float y_phi = y*cos_phi - x*sin_phi; // 3
            const float z_offset = z_0 - (phis[iphi] * T_g.w + startVals_g.w); // 3

            const float L_xy = y_dot_n - (x_phi*d_n_vec_0.x + y_phi*d_n_vec_0.y); // 4
            const float v_num_xy = x_phi*d_v_vec_0.x + y_phi*d_v_vec_0.y - y_dot_v; // 4
            const float u_num_xy = x_phi*d_u_vec_0.x + y_phi*d_u_vec_0.y - y_dot_u; // 4
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = k_offset*T_f.z + z_offset; // 2

                const float L_inv = 1.0f / (L_xy - z*d_n_vec_0.z); // (2, 1)
                const float v_phi_x = (v_num_xy + z*d_v_vec_0.z) * L_inv; // 3

                if (v_min <= v_phi_x && v_phi_x <= v_max)
                {
                    const float u_phi_x = (u_num_xy + z*d_u_vec_0.z) * L_inv; // 3

                    const float phi_conj = phis[iphi] + 2.0f*(atan2(-d_n_vec_0.y + u_phi_x*d_u_vec_0.y + v_phi_x*d_v_vec_0.y, -d_n_vec_0.x + u_phi_x*d_u_vec_0.x + v_phi_x*d_v_vec_0.x) - psi) + PI;
                    float cos_phi_conj, sin_phi_conj;
                    __sincosf(phi_conj, &sin_phi_conj, &cos_phi_conj);

                    const float3 y_phi_conj_minus_x = make_float3(R*cos_phi_conj + tau*sin_phi_conj - x, R*sin_phi_conj - tau*cos_phi_conj - y, phi_conj * T_g.w + startVals_g.w - z);
                    const float3 n_vec_conj = make_float3(d_n_vec_0.x*cos_phi_conj - d_n_vec_0.y*sin_phi_conj, d_n_vec_0.x*sin_phi_conj + d_n_vec_0.y*cos_phi_conj, d_n_vec_0.z);
                    const float3 v_vec_conj = make_float3(d_v_vec_0.x*cos_phi_conj - d_v_vec_0.y*sin_phi_conj, d_v_vec_0.x*sin_phi_conj + d_v_vec_0.y*cos_phi_conj, d_v_vec_0.z);
                    const float L_conj_inv = 1.0f / dot3(y_phi_conj_minus_x, n_vec_conj);

                    const float v_phi_conj_x = -dot3(y_phi_conj_minus_x, v_vec_conj) * L_conj_inv;

                    bool has_conjugate = (v_min <= v_phi_conj_x && v_phi_conj_x <= v_max);

                    if (d_doDBP && has_conjugate == true)
                    {
                        const float3 u_vec_conj = make_float3(d_u_vec_0.x*cos_phi_conj - d_u_vec_0.y*sin_phi_conj, d_u_vec_0.x*sin_phi_conj + d_u_vec_0.y*cos_phi_conj, d_u_vec_0.z);
                        const float u_phi_x_conj = -dot3(y_phi_conj_minus_x, u_vec_conj) * L_conj_inv;
                        if (u_0 > u_phi_x_conj || u_phi_x_conj > u_0+u_length)
                            has_conjugate = false;
                    }

                    if (has_conjugate)
                        vals[k_offset] += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi_plus_half) * L_inv * L_inv * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
                    else
                        vals[k_offset] += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi_plus_half) * L_inv * (L_inv + L_conj_inv) * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
                }
            }
        }
    }
    else
    {
        for (int iphi = 0; iphi < N_g.x; iphi++)
        {
            const float iphi_plus_half = float(iphi) + 0.5f;
            const float phi = phis[iphi];
            float cos_phi, sin_phi;
            __sincosf(phi, &sin_phi, &cos_phi);

            const float u_0 = startVals_g.z;
            const float v_0 = startVals_g.y;

            // fmaf(x, y, z) = x*y + z
            const float x_phi = x*cos_phi + y*sin_phi; // 3
            const float y_phi = y*cos_phi - x*sin_phi; // 3
            const float z_offset = z_0 - (phis[iphi] * T_g.w + startVals_g.w); // 3

            const float L_xy = y_dot_n - (x_phi*d_n_vec_0.x + y_phi*d_n_vec_0.y); // 4
            const float v_num_xy = x_phi*d_v_vec_0.x + y_phi*d_v_vec_0.y - y_dot_v; // 4
            const float u_num_xy = x_phi*d_u_vec_0.x + y_phi*d_u_vec_0.y - y_dot_u; // 4
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = k_offset*T_f.z + z_offset; // 2

                const float L_inv = 1.0f / (L_xy - z*d_n_vec_0.z); // (2, 1)
                const float v_phi_x = (v_num_xy + z*d_v_vec_0.z) * L_inv; // 3

                if (v_min <= v_phi_x && v_phi_x <= v_max)
                {
                    const float u_phi_x = (u_num_xy + z*d_u_vec_0.z) * L_inv; // 3
                    vals[k_offset] += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi_plus_half) * L_inv * L_inv * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
                }
            }
        }
    }

    const float scalar = T_f.x * T_f.y * T_f.z * (Tv_inv * Tu_inv);
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

__global__ void coneBeamBackprojectorKernel_rot_vox(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* __restrict__ f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* __restrict__ phis, const int volumeDimensionOrder, bool do_helicalFBP, bool do_inconsistency, bool accum)
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
    const float v_min = (-0.5f-float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;
    const float v_max = (float(N_g.y)-0.5f+float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;

    float val = 0.0f;
    if (do_helicalFBP)
    {
        //const float atan_term = atan(2.0f * tau * R / (R * R - tau * tau));
        const float twoPI_inv = 1.0f / (2.0f * PI);
        const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
        const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;
        const float psi = atan(-tau/R);

        if (do_inconsistency)
        {
            for (int iphi = 0; iphi < N_g.x; iphi++)
            {
                float cos_phi, sin_phi;
                const float phi = phis[iphi];
                __sincosf(phi, &sin_phi, &cos_phi);

                const float u_0 = startVals_g.z;
                const float v_0 = startVals_g.y;

                const float3 y_phi_minus_x = make_float3(R*cos_phi + tau*sin_phi - x, R*sin_phi - tau*cos_phi - y, phi * T_g.w + startVals_g.w - z);

                const float3 n_vec = make_float3(d_n_vec_0.x*cos_phi - d_n_vec_0.y*sin_phi, d_n_vec_0.x*sin_phi + d_n_vec_0.y*cos_phi, d_n_vec_0.z);
                const float3 u_vec = make_float3(d_u_vec_0.x*cos_phi - d_u_vec_0.y*sin_phi, d_u_vec_0.x*sin_phi + d_u_vec_0.y*cos_phi, d_u_vec_0.z);
                const float3 v_vec = make_float3(d_v_vec_0.x*cos_phi - d_v_vec_0.y*sin_phi, d_v_vec_0.x*sin_phi + d_v_vec_0.y*cos_phi, d_v_vec_0.z);

                const float L = dot3(y_phi_minus_x, n_vec);
                const float L_inv = 1.0f / L;
                const float u_phi_x = -dot3(y_phi_minus_x, u_vec) * L_inv;
                const float v_phi_x = -dot3(y_phi_minus_x, v_vec) * L_inv;

                float centralWeight = helicalConeWeight_vox(v_phi_x);
                if (centralWeight > 0.0f)
                {
                    float biggest_v = v_phi_x;
                    int numContributions = 1;

                    //const float phi_conj = phi - 2.0f * atan(u_val) + atan_term + PI;
                    //traj = -n_vec_0 + u*u_vec_0 + v*v_vec_0
                    //alpha = np.arctan2(traj[1,0], traj[0,0])
                    const float phi_conj = phi + 2.0f*(atan2(-d_n_vec_0.y + u_phi_x*d_u_vec_0.y + v_phi_x*d_v_vec_0.y, -d_n_vec_0.x + u_phi_x*d_u_vec_0.x + v_phi_x*d_v_vec_0.x) - psi) + PI;
                    float cos_phi_conj, sin_phi_conj;
                    __sincosf(phi_conj, &sin_phi_conj, &cos_phi_conj);

                    const float3 y_phi_conj_minus_x = make_float3(R*cos_phi_conj + tau*sin_phi_conj - x, R*sin_phi_conj - tau*cos_phi_conj - y, phi_conj * T_g.w + startVals_g.w - z);
                    const float3 n_vec_conj = make_float3(d_n_vec_0.x*cos_phi_conj - d_n_vec_0.y*sin_phi_conj, d_n_vec_0.x*sin_phi_conj + d_n_vec_0.y*cos_phi_conj, d_n_vec_0.z);
                    const float3 v_vec_conj = make_float3(d_v_vec_0.x*cos_phi_conj - d_v_vec_0.y*sin_phi_conj, d_v_vec_0.x*sin_phi_conj + d_v_vec_0.y*cos_phi_conj, d_v_vec_0.z);
                    const float L_conj = dot3(y_phi_conj_minus_x, n_vec_conj);
                    const float L_conj_inv = 1.0f / L_conj;

                    float sumWeights = 0.0f;

                    // Calculate contributions from 2n\pi shifts
                    const float v_val_shift = neg_twoPI_pitch * L_inv;
                    const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_phi_x) * L * neg_twoPI_pitch_inv : (v_max - v_phi_x) * L * neg_twoPI_pitch_inv;
                    const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_phi_x) * L * neg_twoPI_pitch_inv : (v_max - v_phi_x) * L * neg_twoPI_pitch_inv;
                    const int N_turns_below = max(int(ceilf((d_phi_start - phi) * twoPI_inv)), int(ceilf(v_bound_A)));
                    const int N_turns_above = min(int(floorf((d_phi_end - phi) * twoPI_inv)), int(floorf(v_bound_B)));
                    for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
                    {
                        if (iturn != 0)
                        {
                            float new_weight = helicalConeWeight_vox(v_phi_x + iturn * v_val_shift);
                            if (new_weight > 0.0f)
                            {
                                numContributions += 1;
                                if (fabsf(biggest_v) < fabsf(v_phi_x + iturn * v_val_shift))
                                    biggest_v = v_phi_x + iturn * v_val_shift;
                                sumWeights += new_weight;
                            }
                        }
                    }

                    // Calculate contributions from (2n+1)\pi shifts
                    //const float v_phi_conj_x = -v_phi_x;
                    const float v_phi_conj_x = -dot3(y_phi_conj_minus_x, v_vec_conj) * L_conj_inv;
                    const float v_val_shift_conj = neg_twoPI_pitch * L_conj_inv;
                    const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv : (v_max - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv;
                    const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv : (v_max - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv;

                    const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
                    const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
                    for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                    {
                        float new_weight = helicalConeWeight_vox(v_phi_conj_x + iturn * v_val_shift_conj);
                        if (new_weight > 0.0f)
                        {
                            numContributions += 1;
                            if (fabsf(biggest_v) < fabsf(v_phi_conj_x + iturn * v_val_shift_conj))
                                    biggest_v = v_phi_conj_x + iturn * v_val_shift_conj;
                            sumWeights += new_weight;
                        }
                    }

                    if (numContributions % 2 == 1)
                    {
                        if (biggest_v == v_phi_x)
                        {
                            centralWeight = 0.0f;
                            sumWeights = 1.0f;
                            continue;
                        }
                        else
                            sumWeights -= helicalConeWeight_vox(biggest_v);
                    }

                    val += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi + 0.5f) * L_inv * centralWeight / (centralWeight + sumWeights);
                }
            }
        }
        else
        {
            for (int iphi = 0; iphi < N_g.x; iphi++)
            {
                float cos_phi, sin_phi;
                const float phi = phis[iphi];
                __sincosf(phi, &sin_phi, &cos_phi);

                const float3 y_phi_minus_x = make_float3(R*cos_phi + tau*sin_phi - x, R*sin_phi - tau*cos_phi - y, phi * T_g.w + startVals_g.w - z);
                const float3 n_vec = make_float3(d_n_vec_0.x*cos_phi - d_n_vec_0.y*sin_phi, d_n_vec_0.x*sin_phi + d_n_vec_0.y*cos_phi, d_n_vec_0.z);
                const float3 v_vec = make_float3(d_v_vec_0.x*cos_phi - d_v_vec_0.y*sin_phi, d_v_vec_0.x*sin_phi + d_v_vec_0.y*cos_phi, d_v_vec_0.z);

                const float L = dot3(y_phi_minus_x, n_vec);
                const float L_inv = 1.0f / L;
                const float v_phi_x = -dot3(y_phi_minus_x, v_vec) * L_inv;

                const float centralWeight = helicalConeWeight_vox(v_phi_x);
                if (centralWeight > 0.0f)
                {
                    const float u_0 = startVals_g.z;
                    const float v_0 = startVals_g.y;

                    const float u_min = u_0;// - 0.5f*T_g.z;
                    const float u_max = T_g.z*(N_g.z-1) + u_0;// + 0.5f*T_g.z;
                    const float3 u_vec = make_float3(d_u_vec_0.x*cos_phi - d_u_vec_0.y*sin_phi, d_u_vec_0.x*sin_phi + d_u_vec_0.y*cos_phi, d_u_vec_0.z);
                    const float u_phi_x = -dot3(y_phi_minus_x, u_vec) * L_inv;
                    if (u_min <= u_phi_x && u_phi_x <= u_max)
                    {
                        const float phi_conj = phi + 2.0f*(atan2(-d_n_vec_0.y + u_phi_x*d_u_vec_0.y + v_phi_x*d_v_vec_0.y, -d_n_vec_0.x + u_phi_x*d_u_vec_0.x + v_phi_x*d_v_vec_0.x) - psi) + PI;
                        float cos_phi_conj, sin_phi_conj;
                        __sincosf(phi_conj, &sin_phi_conj, &cos_phi_conj);

                        const float3 y_phi_conj_minus_x = make_float3(R*cos_phi_conj + tau*sin_phi_conj - x, R*sin_phi_conj - tau*cos_phi_conj - y, phi_conj * T_g.w + startVals_g.w - z);
                        const float3 n_vec_conj = make_float3(d_n_vec_0.x*cos_phi_conj - d_n_vec_0.y*sin_phi_conj, d_n_vec_0.x*sin_phi_conj + d_n_vec_0.y*cos_phi_conj, d_n_vec_0.z);
                        const float3 v_vec_conj = make_float3(d_v_vec_0.x*cos_phi_conj - d_v_vec_0.y*sin_phi_conj, d_v_vec_0.x*sin_phi_conj + d_v_vec_0.y*cos_phi_conj, d_v_vec_0.z);
                        const float L_conj = dot3(y_phi_conj_minus_x, n_vec_conj);
                        const float L_conj_inv = 1.0f / L_conj;

                        float sumWeights = 0.0f;

                        // Calculate contributions from 2n\pi shifts
                        const float v_val_shift = neg_twoPI_pitch * L_inv;
                        const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_phi_x) * L * neg_twoPI_pitch_inv : (v_max - v_phi_x) * L * neg_twoPI_pitch_inv;
                        const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_phi_x) * L * neg_twoPI_pitch_inv : (v_max - v_phi_x) * L * neg_twoPI_pitch_inv;
                        const int N_turns_below = max(int(ceilf((d_phi_start - phi) * twoPI_inv)), int(ceilf(v_bound_A)));
                        const int N_turns_above = min(int(floorf((d_phi_end - phi) * twoPI_inv)), int(floorf(v_bound_B)));
                        sumWeights = helicalConeWeight_vox_sum(v_phi_x, v_val_shift, N_turns_below, N_turns_above);

                        const float3 u_vec_conj = make_float3(d_u_vec_0.x*cos_phi_conj - d_u_vec_0.y*sin_phi_conj, d_u_vec_0.x*sin_phi_conj + d_u_vec_0.y*cos_phi_conj, d_u_vec_0.z);
                        const float u_phi_x_conj = -dot3(y_phi_conj_minus_x, u_vec_conj) * L_conj_inv;
                        if (u_min <= u_phi_x_conj && u_phi_x_conj <= u_max)
                        {
                            // Calculate contributions from (2n+1)\pi shifts
                            //const float v_phi_conj_x = -v_phi_x;
                            const float v_phi_conj_x = -dot3(y_phi_conj_minus_x, v_vec_conj) * L_conj_inv;
                            const float v_val_shift_conj = neg_twoPI_pitch * L_conj_inv;
                            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv : (v_max - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv;
                            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv : (v_max - v_phi_conj_x) * L_conj * neg_twoPI_pitch_inv;

                            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
                            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));

                            sumWeights += helicalConeWeight_vox_sum(v_phi_conj_x, v_val_shift_conj, N_turns_below_conj, N_turns_above_conj);
                        }
                        if (sumWeights > 0.0f)
                            val += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi + 0.5f) * L_inv * centralWeight / sumWeights;
                    }
                }
            }
        }
    }
    else
    {
        for (int iphi = 0; iphi < N_g.x; iphi++)
        {
            float cos_phi, sin_phi;
            const float phi = phis[iphi];
            __sincosf(phi, &sin_phi, &cos_phi);

            const float u_0 = startVals_g.z;
            const float v_0 = startVals_g.y;

            const float3 y_phi_minus_x = make_float3(R*cos_phi + tau*sin_phi - x, R*sin_phi - tau*cos_phi - y, phis[iphi] * T_g.w + startVals_g.w - z);

            const float3 n_vec = make_float3(d_n_vec_0.x*cos_phi - d_n_vec_0.y*sin_phi, d_n_vec_0.x*sin_phi + d_n_vec_0.y*cos_phi, d_n_vec_0.z);
            const float3 u_vec = make_float3(d_u_vec_0.x*cos_phi - d_u_vec_0.y*sin_phi, d_u_vec_0.x*sin_phi + d_u_vec_0.y*cos_phi, d_u_vec_0.z);
            const float3 v_vec = make_float3(d_v_vec_0.x*cos_phi - d_v_vec_0.y*sin_phi, d_v_vec_0.x*sin_phi + d_v_vec_0.y*cos_phi, d_v_vec_0.z);

            const float L = dot3(y_phi_minus_x, n_vec);
            const float L_inv = 1.0f / L;
            const float u_phi_x = -dot3(y_phi_minus_x, u_vec) * L_inv;
            const float v_phi_x = -dot3(y_phi_minus_x, v_vec) * L_inv;

            if (v_min <= v_phi_x && v_phi_x <= v_max)
                val += TEX3D(g, (u_phi_x - u_0) * Tu_inv + 0.5f, (v_phi_x - v_0) * Tv_inv + 0.5f, iphi + 0.5f) * L_inv * L_inv * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
        }
    }

    if (do_helicalFBP)
        val /= R;
    val *= T_f.x * T_f.y * T_f.z * Tv_inv * Tu_inv;
    if (accum)
        f[ind] += val;
    else
        f[ind] = val;
}

__global__ void coneBeamBackprojectorKernel_vox(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* __restrict__ f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float pitchAngle, const float rFOVsq, const float* __restrict__ phis, const int volumeDimensionOrder, bool accum)
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

    const float v_min_ind = -0.5f-float(d_numRowsExtrapolate);
    const float v_max_ind = float(N_g.y)-0.5f+float(d_numRowsExtrapolate);
    const float v_min = v_min_ind*T_g.y + startVals_g.y;
    const float v_max = v_max_ind*T_g.y + startVals_g.y;

    //if (i == N_f.x/2 && j == N_f.y/2 && k == N_f.z/2)
    //    printf("%f, %f, %f, %f\n", v_min_ind, v_max_ind, v_min, v_max);

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    if (fabsf(pitchAngle) > 1.0e-6)
    {
        const float cos_pitch = cos(pitchAngle);
        const float sin_pitch = sin(pitchAngle);

        //const float3 n_vec_0 = make_float3(cos_pitch, 0.0f, -sin_pitch);
        const float3 u_vec_0 = make_float3(sin_pitch*sin_tilt, cos_tilt, cos_pitch*sin_tilt);
        const float3 v_vec_0 = make_float3(sin_pitch*cos_tilt, -sin_tilt, cos_pitch*cos_tilt);

        for (int l = 0; l < N_g.x; l++)
        {
            const float L = (float)l + 0.5f;
            const float phi = phis[l];
            //const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
            const float z_source = phi * T_g.w + startVals_g.w;
            const float sin_phi = sin(phi);
            const float cos_phi = cos(phi);

            const float u_0 = startVals_g.z;

            const float dist_from_source_0 = (R - x * cos_phi - y * sin_phi)*cos_pitch;

            //const float3 u_vec = make_float3(cos_phi*u_vec_0.x - sin_phi*u_vec_0.y, sin_phi*u_vec_0.x + cos_phi*u_vec_0.y, u_vec_0.z);
            //const float3 v_vec = make_float3(cos_phi*v_vec_0.x - sin_phi*v_vec_0.y, sin_phi*v_vec_0.x + cos_phi*v_vec_0.y, v_vec_0.z);

            const float u_vec_dot_x = x * (cos_phi*u_vec_0.x - sin_phi*u_vec_0.y) + y * (sin_phi*u_vec_0.x + cos_phi*u_vec_0.y);
            const float v_vec_dot_x = x * (cos_phi*v_vec_0.x - sin_phi*v_vec_0.y) + y * (sin_phi*v_vec_0.x + cos_phi*v_vec_0.y);

            const float u_val_num_shift = R*u_vec_0.x - tau*u_vec_0.y;
            const float v_val_num_shift = R*v_vec_0.x - tau*v_vec_0.y;

            //const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

            //const float backprojectionWeight = sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
            //const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;

            //const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z_val = z + k_offset*T_f.z - z_source;
                const float dist_from_source_inv = 1.0f / (dist_from_source_0 + sin_pitch*z_val);

                const float v_val = (v_vec_dot_x + z_val * v_vec_0.z - v_val_num_shift) * dist_from_source_inv;

                if (v_min <= v_val && v_val <= v_max)
                {
                    const float u_val = (u_vec_dot_x + z_val * u_vec_0.z - u_val_num_shift) * dist_from_source_inv;
                    //const float backprojectionWeight = dist_from_source_inv * dist_from_source_inv;
                    vals[k_offset] += TEX3D(g, (u_val - u_0) * Tu_inv + 0.5f, (v_val - startVals_g.y) * Tv_inv + 0.5f, L) * dist_from_source_inv * dist_from_source_inv * sqrtf(1.0f + u_val * u_val + v_val * v_val);
                }
            }
        }
    }
    else
    {
        for (int l = 0; l < N_g.x; l++)
        {
            const float L = (float)l + 0.5f;
            const float phi = phis[l];
            const float sin_phi = sin(phi);
            const float cos_phi = cos(phi);

            const float u_0 = startVals_g.z;

            const float R_minus_x_dot_theta_inv = 1.0f / (R - x * cos_phi - y * sin_phi);

            //const float backprojectionWeight = sqrtf(1.0f + u_val * u_val) * R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;
            const float backprojectionWeight = R_minus_x_dot_theta_inv * R_minus_x_dot_theta_inv;

            if (tiltAngle == 0.0f)
            {
                const float z_source_over_T_v = (phi * T_g.w + startVals_g.w) * Tv_inv;
                const float u_val = (cos_phi * y - sin_phi * x + tau) * R_minus_x_dot_theta_inv;
                const float u_arg = (u_val - u_0) * Tu_inv + 0.5f;

                const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;
                const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv + 0.5f;
                for (int k_offset = 0; k_offset < numZ; k_offset++)
                {
                    const float v_arg = (v_phi_x_first + k_offset * v_phi_x_step_A + v0_over_Tv - 0.5f) * T_g.y;
                    if (v_min <= v_arg && v_arg <= v_max)
                        vals[k_offset] += TEX3D(g, u_arg, v_phi_x_first + k_offset * v_phi_x_step_A, L) * backprojectionWeight * sqrtf(1.0f + u_val * u_val + v_arg * v_arg);
                }
            }
            else
            {
                const float z_source = phi * T_g.w + startVals_g.w;
                const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
                for (int k_offset = 0; k_offset < numZ; k_offset++)
                {
                    const float u_val = (x_dot_theta_perp * cos_tilt + (z+k_offset*T_f.z - z_source) * sin_tilt) * R_minus_x_dot_theta_inv;
                    const float v_val = ((z + k_offset * T_f.z - z_source) * cos_tilt - x_dot_theta_perp * sin_tilt) * R_minus_x_dot_theta_inv;
                    if (v_min <= v_val && v_val <= v_max)
                        vals[k_offset] += TEX3D(g, (u_val - u_0) * Tu_inv + 0.5f, (v_val - startVals_g.y) * Tv_inv + 0.5f, L) * backprojectionWeight * sqrtf(1.0f + u_val * u_val + v_val * v_val);
                }
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

__global__ void fanBeamBackprojectorKernel_vox(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool doWeight, bool accum)
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
            vals[k_offset] += TEX3D(g, u_arg, iv + float(k_offset), L) * backprojectionWeight;
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

__global__ void parallelBeamBackprojectorKernel_vox(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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
            vals[k_offset] += TEX3D(g, u_arg, float(k + k_offset) + 0.5f, float(l) + 0.5f);
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

void initialize_rotation_constants(parameters* params)
{
    float cos_tilt = cos(params->tiltAngle*PI/180.0);
    float sin_tilt = sin(params->tiltAngle*PI/180.0);
    float cos_pitch = cos(params->pitchAngle*PI/180.0);
    float sin_pitch = sin(params->pitchAngle*PI/180.0);
    if (fabs(params->tiltAngle) < 1.0e-8)
    {
        cos_tilt = 1.0;
        sin_tilt = 0.0;
    }
    if (fabs(params->pitchAngle) < 1.0e-8)
    {
        cos_pitch = 1.0;
        sin_pitch = 0.0;
    }

    /*
    cos_tilt = 1.0;
    sin_tilt = 0.0;
    cos_pitch = 1.0;
    sin_pitch = 0.0;
    //*/

    //printf("cos_pitch = %f, sin_pitch = %f\n", cos_pitch, sin_pitch);
    //printf("cos_tilt = %f, sin_tilt = %f\n", cos_tilt, sin_tilt);

    cudaMemcpyToSymbol(d_cos_tilt, &cos_tilt, sizeof(float));
    cudaMemcpyToSymbol(d_sin_tilt, &sin_tilt, sizeof(float));
    cudaMemcpyToSymbol(d_cos_pitch, &cos_pitch, sizeof(float));
    cudaMemcpyToSymbol(d_sin_pitch, &sin_pitch, sizeof(float));
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

    bool rotated_detector = setConstantMemoryGeometryParameters(params);

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

    bool doExtrapolation = params->doExtrapolation;
    float rFOVsq = params->rFOV()*params->rFOV();
    if (params->doDBP)
    {
        rFOVsq = params->furthestFromCenter() + params->voxelWidth;
        rFOVsq *= rFOVsq;
        doExtrapolation = false;
    }

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);

    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = NULL;

    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, params, doExtrapolation, true);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, doExtrapolation, true);

    if (d_data_array == nullptr)
    {
        // Texture allocation failed (true OOM even after CUDA pool reclaim).
        cudaFree(dev_phis);
        if (volume_on_cpu && dev_f != 0)
            cudaFree(dev_f);
        return false;
    }

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z)/float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);
    if (params->helicalPitch != 0.0 && params->doWeightedBackprojection == true && (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL))
    {
        if (params->geometry == parameters::CONE_PARALLEL)
        {
            coneParallelWeightedHelicalBackprojectorKernel_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
        }
        else if (params->detectorType == params->FLAT)
        {
            coneBeamBackprojectorKernel_rot_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, true, params->inconsistencyReconstruction, accum);
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
        {
            //coneBeamBackprojectorKernel_vox <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, params->pitchAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
            //coneBeamBackprojectorKernel_rot_vox <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, false, params->inconsistencyReconstruction, accum);
            coneBeamBackprojectorKernel_rot_vox_slab <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
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
    freeTexture(d_data_array, d_data_txt);
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

    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = NULL;
    /*
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;
    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, doLinearInterpolation);
    //*/
    //*
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, params, params->doExtrapolation, doLinearInterpolation);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, doLinearInterpolation);
    //*/

     if (d_data_array == NULL)
     {
         LOG(logERROR, "backprojectors_VD", "backproject_VD_modular") << "Error: failed to load projection data texture." << std::endl;
         cudaFree(dev_sourcePositions);
         cudaFree(dev_moduleCenters);
         cudaFree(dev_rowVectors);
         cudaFree(dev_colVectors);
         if (volume_on_cpu && dev_f != 0)
             cudaFree(dev_f);
         return false;
     }

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
    freeTexture(d_data_array, d_data_txt);
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

bool setConstantMemoryGeometryParameters(parameters* params)
{
    float paddedRows = 0.5;
    if (params->is_essentially_axial_scan())
        paddedRows += float(params->numRowsExtrapolate);
    
    // Set helical parameters
    float q_helical = float(params->helicalFBPWeight);
    if (params->inconsistencyReconstruction)
        q_helical = 1.0;
    float weightFcnParameter = float(-2.0 / ((1.0 - q_helical) * (1.0 - q_helical)));
    float weightFcnTransition = float((q_helical + 1.0) / 2.0);
    float v_min = float((params->v(0) - paddedRows * params->pixelHeight) / params->sdd);
    float v_max = float((params->v(params->numRows - 1) + paddedRows * params->pixelHeight) / params->sdd);
    float v_min_inv = v_min;
    float v_max_inv = v_max;
    if (v_max_inv < 0.0)
        v_max_inv = -v_min_inv;
    if (v_min_inv > 0.0)
        v_min_inv = -v_max_inv;
    v_min_inv = float(1.0 / v_min_inv);
    v_max_inv = float(1.0 / v_max_inv);
    float phi_start = params->get_phi_start();
    float phi_end = params->get_phi_end();

    float v_abs_max_inv = 1.0 / max(fabs(v_min), fabs(v_max));

    cudaMemcpyToSymbol(d_q_helical, &q_helical, sizeof(float));
    cudaMemcpyToSymbol(d_v_min, &v_min, sizeof(float));
    cudaMemcpyToSymbol(d_v_max, &v_max, sizeof(float));
    cudaMemcpyToSymbol(d_v_min_inv, &v_min_inv, sizeof(float));
    cudaMemcpyToSymbol(d_v_max_inv, &v_max_inv, sizeof(float));
    cudaMemcpyToSymbol(d_v_abs_max_inv, &v_abs_max_inv, sizeof(float));
    cudaMemcpyToSymbol(d_weightFcnTransition, &weightFcnTransition, sizeof(float));
    cudaMemcpyToSymbol(d_weightFcnParameter, &weightFcnParameter, sizeof(float));
    cudaMemcpyToSymbol(d_phi_start, &phi_start, sizeof(float));
    cudaMemcpyToSymbol(d_phi_end, &phi_end, sizeof(float));

    // Set cos, sin functions
    float cos_tilt = cos(params->tiltAngle*PI/180.0);
    float sin_tilt = sin(params->tiltAngle*PI/180.0);
    float cos_pitch = cos(params->pitchAngle*PI/180.0);
    float sin_pitch = sin(params->pitchAngle*PI/180.0);
    if (fabs(params->tiltAngle) < 1.0e-8 || params->geometry != parameters::CONE)
    {
        cos_tilt = 1.0;
        sin_tilt = 0.0;
    }
    if (fabs(params->pitchAngle) < 1.0e-8 || params->geometry != parameters::CONE)
    {
        cos_pitch = 1.0;
        sin_pitch = 0.0;
    }

    cudaMemcpyToSymbol(d_cos_tilt, &cos_tilt, sizeof(float));
    cudaMemcpyToSymbol(d_sin_tilt, &sin_tilt, sizeof(float));
    cudaMemcpyToSymbol(d_cos_pitch, &cos_pitch, sizeof(float));
    cudaMemcpyToSymbol(d_sin_pitch, &sin_pitch, sizeof(float));

    // Set d_numRowsExtrapolate
    if (params->helicalPitch == 0.0)
        cudaMemcpyToSymbol(d_numRowsExtrapolate, &(params->numRowsExtrapolate), sizeof(int));
    else
    {
        int numRowsExtrapolate = 0;
        cudaMemcpyToSymbol(d_numRowsExtrapolate, &numRowsExtrapolate, sizeof(int));
    }

    // Set vectors defining detector rotation
    float3 n_vec = make_float3(cos_pitch, 0.0, -sin_pitch);
    float3 u_vec = make_float3(sin_pitch * sin_tilt, cos_tilt, cos_pitch * sin_tilt);
    float3 v_vec = make_float3(sin_pitch * cos_tilt, -sin_tilt, cos_pitch * cos_tilt);
    cudaMemcpyToSymbol(d_n_vec_0, &n_vec, sizeof(float3));
    cudaMemcpyToSymbol(d_u_vec_0, &u_vec, sizeof(float3));
    cudaMemcpyToSymbol(d_v_vec_0, &v_vec, sizeof(float3));


    //float R_tau = params->sod;
    float R_tau = sqrt(params->sod*params->sod + params->tau*params->tau);
    cudaMemcpyToSymbol(d_R_tau, &R_tau, sizeof(float));
    float2 zFOV = make_float2(params->zFOV_peaks[0], params->zFOV_peaks[1]);
    if (params->inconsistencyReconstruction || params->doWeightedBackprojection == false || params->offsetScan)
    {
        zFOV.x = 0.0;
        zFOV.y = 0.0;
    }
    cudaMemcpyToSymbol(d_zFOV, &zFOV, sizeof(float2));

    cudaMemcpyToSymbol(d_doDBP, &(params->doDBP), sizeof(bool));


    //* Return whether or not the detector is rotated
    if (sin_pitch == 0.0 && sin_tilt == 0.0)
        return false;
    else
        return true;
    //*/
    //return true;
}
