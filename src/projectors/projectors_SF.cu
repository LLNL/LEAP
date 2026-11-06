////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for the primary projectors models in LEAP
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors.h"
#include "projectors_SF.cuh"
#include "cuda_utils.h"

#include "projectors_extendedSF.cuh"

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

__device__ __forceinline__ float helicalConeWeight(const float v)
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

__device__ __forceinline__ float helicalConeWeight_sum(const float x, const float y, const int M, const int N)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void coneParallelWeightedHelicalBackprojectorKernel_SF(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0;
        return;
    }

    const float maxWeight = T_f.x * T_f.y / T_g.z;
    const float maxWeight_inv = 1.0f / maxWeight;

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;
    const float C_num = 0.5f * T_u_inv * T_f.x;
    const float C_num_T_x = T_f.x * C_num;
    //const float x_mult = x * T_u_inv;
    //const float y_mult = y * T_u_inv;
    const float s_shift = -startVals_g.z * T_u_inv;

    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    const float asin_tau_over_R = asin(tau / R);

    float val = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float L = float(l) + 0.5f;
        const float phi_cur = phis[l];
        const float sin_phi = sinf(phi_cur);
        const float cos_phi = cosf(phi_cur);
        const float C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));

        const float s = cos_phi * y - sin_phi * x;
        const float x_dot_theta = cos_phi * x + sin_phi * y;

        //float s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
        float s_argInd = s_shift + s * T_u_inv;
        const float ds = modf(s_argInd, &s_argInd);
        const float s_ind_A = s_argInd - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

        const float v_denom = sqrtf(R * R - s * s) - x_dot_theta;
        const float v_denom_inv = 1.0f / v_denom;

        const float alpha = asin(s / R) + asin_tau_over_R;
        const float z_source = (phis[l] + alpha) * T_g.w + startVals_g.w;

        const float v_phi_x = (v_phi_x_start_num - z_source * T_v_inv) * v_denom_inv - v0_over_Tv;

        const float v_arg = v_phi_x * T_g.y + startVals_g.y;
        const float centralWeight = helicalConeWeight(v_arg);
        if (centralWeight > 0.0f)
        {
            const float v_denom_conj = v_denom + 2.0f * x_dot_theta;
            const float v_denom_conj_inv = 1.0f / v_denom_conj;
            const float phi_cur_conj = phi_cur + PI;

            float sumWeights = 0.0f;

            const float v_arg_shift = neg_twoPI_pitch * v_denom_inv;

            const float v_bound_A = (v_arg_shift > 0.0f) ? (v_min - v_arg) * v_denom * neg_twoPI_pitch_inv : (v_max - v_arg) * v_denom * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_arg_shift < 0.0f) ? (v_min - v_arg) * v_denom * neg_twoPI_pitch_inv : (v_max - v_arg) * v_denom * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceilf((d_phi_start - phi_cur) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi_cur) * twoPI_inv)), int(floorf(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight(v_arg + iturn * v_arg_shift);
            }

            const float alpha_conj = asin(-s / R) + asin_tau_over_R;
            const float v_arg_conj = (z - ((phi_cur_conj+alpha_conj) * T_g.w + startVals_g.w)) * v_denom_conj_inv;
            const float v_arg_shift_conj = neg_twoPI_pitch * v_denom_conj_inv;

            const float v_bound_A_conj = (v_arg_shift_conj > 0.0f) ? (v_min - v_arg_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * v_denom_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_arg_shift_conj < 0.0f) ? (v_min - v_arg_conj) * v_denom_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * v_denom_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_arg_conj + iturn * v_arg_shift_conj);
            //sumWeights = 0.0f;

            const float v_phi_x_step = Tz_over_Tv * v_denom_inv;
            const float bpWeight = v_denom * rsqrtf(R * R + v_arg * v_arg) * centralWeight / (centralWeight + sumWeights);

            const float row_high = floorf(v_phi_x - 0.5f * v_phi_x_step + 0.5f) + 0.5f;
            const float z_high = v_phi_x + 0.5f * v_phi_x_step - row_high;

            const float v_weight_one = min(v_phi_x_step, v_phi_x_step - z_high);
            const float v_weight_two = max(0.0f, min(z_high, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two = row_high + 2.0f;

            if (z_high > 1.0f)
            {
                val += (TEX3D(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two)
                    + TEX3D(g, s_ind_A, row_high_plus_two, L) * (z_high - 1.0f)) * bpWeight;
            }
            else
            {
                val += TEX3D(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two) * bpWeight;
            }
        }
    }
    if (accum)
        f[ind] += val * maxWeight;
    else
        f[ind] = val * maxWeight;
}

__global__ void coneParallelBackprojectorKernel_SF(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_SLICES_PER_THREAD;
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

    const float maxWeight = T_f.x * T_f.y / T_g.z;
    const float maxWeight_inv = 1.0f / maxWeight;

    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;
    const float C_num = 0.5f * T_u_inv * T_f.x;
    const float C_num_T_x = T_f.x * C_num;
    //const float x_mult = x * T_u_inv;
    //const float y_mult = y * T_u_inv;
    const float s_shift = -startVals_g.z * T_u_inv;

    const float asin_tau_over_R = asin(tau / R);

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        const float L = float(l) + 0.5f;
        const float sin_phi = sinf(phis[l]);
        const float cos_phi = cosf(phis[l]);
        const float C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));

        const float s = cos_phi * y - sin_phi * x;
        const float x_dot_theta = cos_phi * x + sin_phi * y;

        //float s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
        float s_argInd = s_shift + s * T_u_inv;
        const float ds = modf(s_argInd, &s_argInd);
        const float s_ind_A = s_argInd - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

        const float v_denom = sqrtf(R * R - s * s) - x_dot_theta;
        const float v_denom_inv = 1.0f / v_denom;

        float z_source = 0.0f;
        if (T_g.w != 0.0f)
        {
            const float alpha = asin(s / R) + asin_tau_over_R;
            z_source = (phis[l] + alpha) * T_g.w + startVals_g.w;
        }
        const float v_phi_x_step = Tz_over_Tv * v_denom_inv;
        const float v_phi_x_first = (v_phi_x_start_num - z_source * T_v_inv) * v_denom_inv - v0_over_Tv;
        //const float v_arg = (z - z_source) / v_denom;
        //const float v_argInd = (v_arg - T_g.y) * T_v_inv;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            const float v_phi_x = v_phi_x_first + k_offset * v_phi_x_step;

            float bpWeight = 1.0f;
            if (doWeight)
            {
                const float v_arg = v_phi_x * T_g.y + startVals_g.y;
                bpWeight = v_denom * rsqrtf(R * R + v_arg * v_arg);
            }

            const float row_high = floorf(v_phi_x - 0.5f * v_phi_x_step + 0.5f) + 0.5f;
            const float z_high = v_phi_x + 0.5f * v_phi_x_step - row_high;

            const float v_weight_one = min(v_phi_x_step, v_phi_x_step - z_high);
            const float v_weight_two = max(0.0f, min(z_high, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two = row_high + 2.0f;

            if (z_high > 1.0f)
            {
                vals[k_offset] += (TEX3D(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two)
                    + TEX3D(g, s_ind_A, row_high_plus_two, L) * (z_high - 1.0f)) * bpWeight;
            }
            else
            {
                vals[k_offset] += TEX3D(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two) * bpWeight;
            }
        }
    }

    if (accum)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] += vals[k_offset] * maxWeight;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * maxWeight;
        }
    }
    else
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = vals[k_offset] * maxWeight;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * maxWeight;
        }
    }
}

__global__ void coneParallelProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, TEX_DATA f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float n_minus_half = (float)n - 0.5f;
    const float n_plus_half = (float)n + 0.5f;
    const float l_phi = T_f.x / max(fabsf(cos_phi), fabsf(sin_phi));
    const float C = T_f.x * T_f.x / (2.0f * T_g.z * l_phi);

    const float ds_ind_di = -T_f.x * sin_phi / T_g.z;
    const float ds_ind_dj = T_f.y * cos_phi / T_g.z;
    const float s_ind_offset = (startVals_f.y * cos_phi - startVals_f.x * sin_phi - startVals_g.z) / T_g.z;
    // s_ind(i,j) = (float)i * ds_ind_di + (float)j * ds_ind_dj + s_ind_offset

    const float m_minus_half = (float)m - 0.5f;
    const float m_plus_half = (float)m + 0.5f;

    const float v0_over_Tv = startVals_g.y / T_g.y;

    const float alpha = asin(u / R) + asin(tau / R);
    const float z_source = (phis[l]+alpha) * T_g.w + startVals_g.w;

    const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
    const float z_ind_offset = -z0_over_Tz_plus_half + z_source / T_f.z;

    const float z_ind_slope = (v - 0.5f * T_g.y) / T_f.z;

    float g_output = 0.0f;
    if (fabsf(cos_phi) > fabsf(sin_phi))
    {
        const float ds_ind_dj_inv = 1.0f / ds_ind_dj;
        float shiftConstant;
        if (ds_ind_dj > 0.0f)
            shiftConstant = (n_minus_half - C) * ds_ind_dj_inv;
        else
            shiftConstant = (n_plus_half + C) * ds_ind_dj_inv;
        for (int i = 0; i < N_f.x; i++)
        {
            const float s_ind_base = (float)i * ds_ind_di + s_ind_offset;
            const int j_min_A = (int)ceilf(shiftConstant - s_ind_base * ds_ind_dj_inv);
            const float s_ind_A = s_ind_base + (float)j_min_A * ds_ind_dj;

            const float x = float(i) * T_f.x + startVals_f.x;
            const float y = float(j_min_A) * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float x_dot_theta = cos_phi * x + sin_phi * y;
            const float v_denom = sqrtf(R * R - u * u) - x_dot_theta;

            const int k = (int)ceilf(z_ind_slope * v_denom + z_ind_offset);

            const float hWeight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float hWeight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C));
            const float hWeight_2 = max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C));

            const float v_phi_x_step = T_f.z / (T_g.y * v_denom);
            const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
            const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
            const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
            const float y_12 = float(j_min_A) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);

            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(j_min_A + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(k + 2) + 0.5f, float(j_min_A + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
    }
    else
    {
        const float ds_ind_di_inv = 1.0f / ds_ind_di;
        float shiftConstant;
        if (ds_ind_di > 0.0f)
            shiftConstant = (n_minus_half - C) * ds_ind_di_inv;
        else
            shiftConstant = (n_plus_half + C) * ds_ind_di_inv;
        for (int j = 0; j < N_f.y; j++)
        {
            const float s_ind_base = (float)j * ds_ind_dj + s_ind_offset;
            const int i_min_A = (int)ceilf(shiftConstant - s_ind_base * ds_ind_di_inv);
            const float s_ind_A = s_ind_base + (float)i_min_A * ds_ind_di;

            const float x = float(i_min_A) * T_f.x + startVals_f.x;
            const float y = float(j) * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float x_dot_theta = cos_phi * x + sin_phi * y;
            const float v_denom = sqrtf(R * R - u * u) - x_dot_theta;

            const int k = (int)ceilf(z_ind_slope * v_denom + z_ind_offset);

            const float hWeight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float hWeight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C));
            const float hWeight_2 = max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C));

            const float v_phi_x_step = T_f.z / (T_g.y * v_denom);
            const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
            const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
            const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
            const float x_12 = float(i_min_A) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);

            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(j) + 0.5f, float(i_min_A + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i_min_A + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, x_12, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
    }
    if (accum)
        g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += l_phi * g_output;
    else
        g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = l_phi * g_output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void parallelBeamBackprojectorKernel_SF(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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

    if (x*x + y*y > rFOVsq)
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
    
    const float maxWeight = T_f.x * T_f.y / T_g.z;
    const float maxWeight_inv = 1.0f / maxWeight;

    const float T_u_inv = 1.0f / T_g.z;
    const float C_num = 0.5f * T_u_inv * T_f.x;
    const float C_num_T_x = T_f.x * C_num;
    const float x_mult = x * T_u_inv;
    const float y_mult = y * T_u_inv;
    const float s_shift = -startVals_g.z * T_u_inv;

    //*
    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;
    for (int l = 0; l < N_g.x; l++)
    {
        //float sin_phi, cos_phi;
        //sincosf(phis[l], &sin_phi, &cos_phi);
        const float sin_phi = sinf(phis[l]);
        const float cos_phi = cosf(phis[l]);
        const float C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
        float s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
        const float ds = modf(s_arg, &s_arg);
        const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += TEX3D(g, s_ind_A, float(k+k_offset) + 0.5f, float(l) + 0.5f);
    }

    if (accum)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] += vals[k_offset] * maxWeight;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * maxWeight;
        }
    }
    else
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = vals[k_offset] * maxWeight;
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * maxWeight;
        }
    }
    //*/

    /*
    float cos_phi, sin_phi, C, s_arg, ds;
    
    float val = 0.0;
    int l = 0;
    while (l < N_g.x)
    {
        if (l+3 < N_g.x)
        {
            const int l1 = l + 1;
            const int l2 = l + 2;
            const int l3 = l + 3;

			sin_phi = sin(phis[l]);
			cos_phi = cos(phis[l]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l1]);
			cos_phi = cos(phis[l1]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_B = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l2]);
			cos_phi = cos(phis[l2]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_C = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l3]);
			cos_phi = cos(phis[l3]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_D = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            // Do texture mapping
            val += TEX3D(g,s_ind_A, float(k)+0.5f, float(l)+0.5f)
                +  TEX3D(g,s_ind_B, float(k)+0.5f, float(l1)+0.5f)
                +  TEX3D(g,s_ind_C, float(k)+0.5f, float(l2)+0.5f)
                +  TEX3D(g,s_ind_D, float(k)+0.5f, float(l3)+0.5f);
            l += 4;
        }
        else if (l+1 < N_g.x)
        {
            int l1 = l + 1;

			sin_phi = sin(phis[l]);
			cos_phi = cos(phis[l]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l1]);
			cos_phi = cos(phis[l1]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_B = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            val += TEX3D(g,s_ind_A, float(k)+0.5f, float(l)+0.5f)
                +  TEX3D(g,s_ind_B, float(k)+0.5f, float(l1)+0.5f);
            l += 2;
        }
        else //if (l+1 < N_g.x)
        {
			sin_phi = sin(phis[l]);
			cos_phi = cos(phis[l]);
            C = C_num * max(fabsf(cos_phi), fabsf(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            val += TEX3D(g,s_ind_A, float(k)+0.5f, float(l)+0.5f);
            l += 1;
        }
    }

    f[ind] = val * maxWeight;
    //*/
}

__global__ void parallelBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, TEX_DATA f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    //const float v = m * T_g.y + startVals_g.y;
    //const float u = n * T_g.z + startVals_g.z;
    
    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);
    
    const float n_minus_half = (float)n - 0.5f;
    const float n_plus_half = (float)n + 0.5f;
    const float l_phi = T_f.x / max(fabsf(cos_phi), fabsf(sin_phi));
    const float C = T_f.x * T_f.x / (2.0f * T_g.z * l_phi);

    const float ds_ind_di = -T_f.x*sin_phi / T_g.z;
    const float ds_ind_dj = T_f.y*cos_phi / T_g.z;
    const float s_ind_offset = (startVals_f.y*cos_phi - startVals_f.x*sin_phi - startVals_g.z) / T_g.z;
    // s_ind(i,j) = (float)i * ds_ind_di + (float)j * ds_ind_dj + s_ind_offset

    float g_output = 0.0f;
    if (fabsf(cos_phi) > fabsf(sin_phi))
    {
        const float ds_ind_dj_inv = 1.0f / ds_ind_dj;
        float shiftConstant;
        if (ds_ind_dj > 0.0f)
            shiftConstant = (n_minus_half-C) * ds_ind_dj_inv;
        else
            shiftConstant = (n_plus_half+C) * ds_ind_dj_inv;
        for (int i = 0; i < N_f.x; i++)
        {
            const float s_ind_base = (float)i * ds_ind_di +  s_ind_offset;
            const int j_min_A = (int)ceilf(shiftConstant - s_ind_base * ds_ind_dj_inv);
            const float s_ind_A = s_ind_base + (float)j_min_A * ds_ind_dj;

            if (((float)i*T_f.x+startVals_f.x )*((float)i*T_f.x+startVals_f.x) + ((float)j_min_A*T_f.y+startVals_f.y )*((float)j_min_A*T_f.y+startVals_f.y) > rFOVsq)
                continue;

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += (weight_0 + weight_1) * TEX3D(f, float(m)+0.5f, float(j_min_A)+0.5f+ weight_1/(weight_0+ weight_1), float(i)+0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * TEX3D(f, float(m)+0.5f, float(j_min_A + 2) + 0.5f, float(i) + 0.5f);
            }
            else
            {
                g_output += (weight_0 + weight_1) * TEX3D(f, float(i) + 0.5f, float(j_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * TEX3D(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, float(m)+0.5f);
            }
        }
    }
    else
    {
        const float ds_ind_di_inv = 1.0f / ds_ind_di;
        float shiftConstant;
        if (ds_ind_di > 0.0f)
            shiftConstant = (n_minus_half-C) * ds_ind_di_inv;
        else
            shiftConstant = (n_plus_half+C) * ds_ind_di_inv;
        for (int j = 0; j < N_f.y; j++)
        {
            const float s_ind_base = (float)j * ds_ind_dj + s_ind_offset;
            const int i_min_A = (int)ceilf(shiftConstant - s_ind_base * ds_ind_di_inv);
            const float s_ind_A = s_ind_base + (float)i_min_A * ds_ind_di;

            if (((float)i_min_A*T_f.x+startVals_f.x )*((float)i_min_A*T_f.x+startVals_f.x) + ((float)j*T_f.y+startVals_f.y )*((float)j*T_f.y+startVals_f.y) > rFOVsq)
                continue;

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += (weight_0 + weight_1) * TEX3D(f, float(m)+0.5f, float(j)+0.5f, float(i_min_A)+0.5f + weight_1/(weight_0 + weight_1))
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * TEX3D(f, float(m) + 0.5f, float(j) + 0.5f, float(i_min_A + 2)+0.5f);
            }
            else
            {
                g_output += (weight_0 + weight_1) * TEX3D(f, float(i_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(j) + 0.5f, float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * TEX3D(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, float(m) + 0.5f);
            }
        }
    }
    if (accum)
        g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += l_phi * g_output;
    else
        g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = l_phi * g_output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fanBeamBackprojectorKernel_SF(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool doWeight, bool accum)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z)*NUM_SLICES_PER_THREAD;
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

    //*
    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    const float iv = (z - startVals_g.y) / T_g.y + 0.5f;
    const float T_x_over_2 = 0.5f * T_f.x;
    const float Tu_inv = 1.0f / T_g.z;

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
        const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
        const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
        const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabsf(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabsf(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
        const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

        float ind_first = floorf(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float ind_last = ind_first + 2.5f;
        ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            vals[k_offset] += (TEX3D(g, ind_first, iv + float(k_offset), L) * horizontalWeights_0_A
                + TEX3D(g, ind_last, iv + float(k_offset), L) * horizontalWeights_1_A) * bpWeight;
        }
    }

    if (accum)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] += vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset];
        }
    }
    else
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset];
        }
    }
    //*/

    /*
    const float iv = (z - startVals_g.y) / T_g.y + 0.5f;
    const float T_x_over_2 = 0.5f * T_f.x;
    const float Tu_inv = 1.0f / T_g.z;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
        const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
        const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
        const float l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabsf(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabsf(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
        const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

        float ind_first = floorf(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float ind_last = ind_first + 2.5f;
        ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;

        val += (TEX3D(g, ind_first, iv, L) * horizontalWeights_0_A
            + TEX3D(g, ind_last, iv, L) * horizontalWeights_1_A) * bpWeight;
    }

    f[ind] = val;
    //*/
}

__global__ void fanBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, TEX_DATA f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float n_minus_half = (float)n - 0.5f + startVals_g.z / T_g.z;
    const float n_plus_half = (float)n + 0.5f + startVals_g.z / T_g.z;
    //const float m_minus_half = (float)m - 0.5f;
    //const float m_plus_half = (float)m + 0.5f;

    const int iz = int(floorf(0.5 + (v - startVals_f.z) / T_f.z));

    float g_output = 0.0f;

    if (fabsf(u * cos_phi - sin_phi) > fabsf(u * sin_phi + cos_phi))
    {
        const float A_x = fabsf(sin_phi) * 0.5f * T_f.x;
        const float B_x = cos_phi * 0.5f * T_f.x * ((sin_phi < 0.0f) ? -1.0f : 1.0f);
        const float Tx_sin = T_f.x * sin_phi;
        const float Tx_cos = T_g.z * T_f.x * cos_phi;

        float shiftConstant, slopeConstant;
        if (u * cos_phi - sin_phi > 0.0f)
        {
            shiftConstant = (((R + B_x) * (u - 0.5f * T_g.z) - A_x - tau) / (cos_phi * (u - 0.5f * T_g.z) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = (-sin_phi * (u - 0.5f * T_g.z) - cos_phi) / (T_f.x * (cos_phi * (u - 0.5f * T_g.z) - sin_phi));
        }
        else
        {
            shiftConstant = (((R - B_x) * (u + 0.5f * T_g.z) + A_x - tau) / (cos_phi * (u + 0.5f * T_g.z) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = (sin_phi * (u + 0.5f * T_g.z) + cos_phi) / (T_f.x * (-cos_phi * (u + 0.5f * T_g.z) + sin_phi));
        }

        for (int j = 0; j < N_f.y; j++)
        {
            const float y = (float)j * T_f.y + startVals_f.y;
            const int i = (int)ceilf(y * slopeConstant + shiftConstant);
            const float x = (float)i * T_f.x + startVals_f.x;

            if (x * x + y * y > rFOVsq)
                continue;

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float num_low = tau - x * sin_phi + y * cos_phi - A_x;
            const float num_high = num_low + 2.0f * A_x;

            const float denom_low = (R_minus_x_dot_theta - B_x) * T_g.z;
            const float denom_high = (R_minus_x_dot_theta + B_x) * T_g.z;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high - Tx_sin) / (denom_high - Tx_cos), n_plus_half) - max((num_low - Tx_sin) / (denom_low - Tx_cos), n_minus_half));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            if (volumeDimensionOrder == 0)
            {
                g_output += TEX3D(f, float(iz) + 0.5f, float(j) + 0.5f, float(i) + 0.5f + hWeight_1/(hWeight_0 + hWeight_1)) * (hWeight_0 + hWeight_1)
                    + TEX3D(f, float(iz) + 0.5f, float(j) + 0.5f, float(i + 2)+0.5f) * hWeight_2;
            }
            else
            {
                g_output += TEX3D(f, float(i) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1), float(j) + 0.5f, float(iz) + 0.5f) * (hWeight_0 + hWeight_1)
                    + TEX3D(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(iz) + 0.5f) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * cos_phi - sin_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabsf(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float A_y = fabsf(cos_phi) * 0.5f * T_f.x;
        const float B_y = sin_phi * 0.5f * T_f.x * ((cos_phi < 0.0f) ? 1.0f : -1.0f);
        const float Ty_cos = T_f.y * cos_phi;
        const float Ty_sin = T_g.z * T_f.y * sin_phi;

        float shiftConstant, slopeConstant;
        if (u * sin_phi + cos_phi >= 0.0f)
        {
            shiftConstant = (((R + B_y) * (u - 0.5f * T_g.z) - A_y - tau) / (sin_phi * (u - 0.5f * T_g.z) + cos_phi) - startVals_f.y) / T_f.y;
            slopeConstant = (sin_phi - cos_phi * (u - 0.5f * T_g.z)) / (T_f.y * (sin_phi * (u - 0.5f * T_g.z) + cos_phi));
        }
        else
        {
            shiftConstant = (((R - B_y) * (u + 0.5f * T_g.z) + A_y - tau) / (cos_phi + sin_phi * (u + 0.5f * T_g.z)) - startVals_f.y) / T_f.y;
            slopeConstant = (sin_phi - cos_phi * (u + 0.5f * T_g.z)) / (T_f.y * (cos_phi + sin_phi * (u + 0.5f * T_g.z)));
        }
        for (int i = 0; i < N_f.x; i++)
        {
            const float x = (float)i * T_f.x + startVals_f.x;
            const int j = (int)ceilf(x * slopeConstant + shiftConstant);
            const float y = (float)j * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float num_low = tau - x * sin_phi + y * cos_phi - A_y;
            const float num_high = num_low + 2.0f * A_y;

            const float denom_low = (R_minus_x_dot_theta - B_y) * T_g.z;
            const float denom_high = (R_minus_x_dot_theta + B_y) * T_g.z;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high + Ty_cos) / (denom_high - Ty_sin), n_plus_half) - max((num_low + Ty_cos) / (denom_low - Ty_sin), n_minus_half));
            const float hWeight_2 = max(0.0f,1.0f - hWeight_1 - hWeight_0);

            if (volumeDimensionOrder == 0)
            {
                g_output += TEX3D(f, float(iz) + 0.5f, float(j) + 0.5f + hWeight_1/(hWeight_0 + hWeight_1), float(i) + 0.5f) * (hWeight_0 + hWeight_1)
                    + TEX3D(f, float(iz) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * hWeight_2;
            }
            else
            {
                g_output += TEX3D(f, float(i) + 0.5f, float(j) + 0.5f+hWeight_1 / (hWeight_0 + hWeight_1), float(iz) + 0.5f) * (hWeight_0 + hWeight_1)
                    + TEX3D(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(iz)+0.5f) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * sin_phi + cos_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabsf(u * sin_phi + cos_phi) * g_output;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void applyPolarWeight(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= rsqrtf(1.0f + v*v);
}

__global__ void applyInversePolarWeight(float* g, int4 N_g, float4 T_g, float4 startVals_g, int g_rowStride)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z) * uint64(g_rowStride) + uint64(j * N_g.z + k)] *= sqrtf(1.0f + v * v);
}

__global__ void curvedConeBeamHelicalWeightedBackprojectorKernel_SF(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }
    const float z = k * T_f.z + startVals_f.z;

    const float T_x_over_2 = 0.5f * T_f.x;
    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
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
        const float z_source = (phi_cur * T_g.w + startVals_g.w);
        const float z_source_over_T_v = z_source * Tv_inv;
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

        const float dist_from_source_components_x = fabsf(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabsf(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source_inv = rsqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);

        const float v_arg = (z - z_source) * dist_from_source_inv;
        const float centralWeight = helicalConeWeight(v_arg);
        if (centralWeight > 0.0f)
        {
            const float L = (float)l + 0.5f;

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float dist_from_source = 1.0f / dist_from_source_inv;
            const float l_phi = T_f.x * dist_from_source / max(dist_from_source_components_x, dist_from_source_components_y);

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;
            const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
            const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
            float A_x;
            if (x_denom > y_denom)
                A_x = fabsf(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabsf(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = (atan((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x)) - startVals_g.z) * Tu_inv;
            const float tau_high = (atan((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x)) - startVals_g.z) * Tu_inv;

            float ind_first = floorf(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            const float v_phi_x_step_A = Tz_over_Tv * dist_from_source_inv;
            const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * dist_from_source_inv - v0_over_Tv;
            
            const float v_phi_x = v_phi_x_first;

            const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            // Calculate the View Redundancy Weight
            const float phi_cur_conj = phi_cur - 2.0f * atan(u_arg) + atan_term + PI;
            const float cos_phi_conj = cos(phi_cur_conj);
            const float sin_phi_conj = sin(phi_cur_conj);
            const float dist_from_source_components_x_conj = fabsf(R * cos_phi_conj + tau * sin_phi_conj - x);
            const float dist_from_source_components_y_conj = fabsf(R * sin_phi_conj - tau * cos_phi_conj - y);
            const float dist_from_source_conj = sqrtf(dist_from_source_components_x_conj * dist_from_source_components_x_conj + dist_from_source_components_y_conj * dist_from_source_components_y_conj);
            const float dist_from_source_inv_conj = 1.0f / dist_from_source_conj;

            float sumWeights = 0.0f;

            const float v_arg_shift = neg_twoPI_pitch * dist_from_source_inv;

            const float v_bound_A = (v_arg_shift > 0.0f) ? (v_min - v_arg) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_arg) * dist_from_source * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_arg_shift < 0.0f) ? (v_min - v_arg) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_arg) * dist_from_source * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceilf((d_phi_start - phi_cur) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi_cur) * twoPI_inv)), int(floorf(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight(v_arg + iturn * v_arg_shift);
            }

            const float v_arg_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * dist_from_source_inv_conj;
            const float v_arg_shift_conj = neg_twoPI_pitch * dist_from_source_inv_conj;

            const float v_bound_A_conj = (v_arg_shift_conj > 0.0f) ? (v_min - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_arg_shift_conj < 0.0f) ? (v_min - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_arg_conj + iturn * v_arg_shift_conj);

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + (z_high_A - 1.0f)));

                val += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (TEX3D(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + TEX3D(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * extraWeight * centralWeight / (centralWeight + sumWeights);
            }
            else
            {
                const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two));

                val += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * extraWeight * centralWeight / (centralWeight + sumWeights);
            }
        }
    }
    //f[ind] = val;
    if (accum)
        f[ind] += val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
    else
        f[ind] = val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
}

//*
__global__ void coneBeamHelicalWeightedBackprojectorKernel_rot_SF(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float pitchAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }
    const float z = k * T_f.z + startVals_f.z;

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    //const float half_T_z = 0.5f * T_f.z;

    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;
    const float atan_term = atan(2.0f * tau * R / (R * R - tau * tau));
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    const float cos_tilt = cos(tiltAngle);
    const float sin_tilt = sin(tiltAngle);

    const float cos_pitch = cos(pitchAngle);
    const float sin_pitch = sin(pitchAngle);

    const float3 u_vec_0 = make_float3(sin_pitch*sin_tilt, cos_tilt, cos_pitch*sin_tilt);
    const float3 v_vec_0 = make_float3(sin_pitch*cos_tilt, -sin_tilt, cos_pitch*cos_tilt);

    const float v_val_num_shift = R*v_vec_0.x - tau*v_vec_0.y; // approx 0.0

    //float horizontalDetectorShift = 0.5f * float(N_g.z - 1) * T_g.z + startVals_g.z;
	//float verticalDetectorShift = 0.5f * float(N_g.y - 1) * T_g.y + startVals_g.y;

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        const float L = (float)iphi + 0.5f;
        const float phi = phis[iphi];
        const float sin_phi = sin(phi);
        const float cos_phi = cos(phi);

        const float u_0 = startVals_g.z;

        const float3 sourcePosition = make_float3(R*cos_phi + tau*sin_phi, R*sin_phi - tau*cos_phi, phi * T_g.w + startVals_g.w);
        const float3 v_vec = make_float3(cos_phi*v_vec_0.x - sin_phi*v_vec_0.y, sin_phi*v_vec_0.x + cos_phi*v_vec_0.y, v_vec_0.z);
        const float3 u_vec = make_float3(cos_phi*u_vec_0.x - sin_phi*u_vec_0.y, sin_phi*u_vec_0.x + cos_phi*u_vec_0.y, u_vec_0.z);
        //const float3 c_minus_s = make_float3(-cos_phi*cos_pitch*D, -sin_phi*cos_pitch*D, sin_pitch*D);
        const float3 c_minus_s = make_float3(-cos_phi*cos_pitch, -sin_phi*cos_pitch, sin_pitch);

        const float3 n_vec = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
            u_vec.z * v_vec.x - u_vec.x * v_vec.z,
            u_vec.x * v_vec.y - u_vec.y * v_vec.x);

        const float c_minus_s_dot_u = c_minus_s.x * u_vec.x + c_minus_s.y * u_vec.y + c_minus_s.z * u_vec.z;
        const float c_minus_s_dot_v = c_minus_s.x * v_vec.x + c_minus_s.y * v_vec.y + c_minus_s.z * v_vec.z;
        const float c_minus_s_dot_n = c_minus_s.x * n_vec.x + c_minus_s.y * n_vec.y + c_minus_s.z * n_vec.z;

        const float3 x_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

        const float denom = x_minus_s.x * n_vec.x + x_minus_s.y * n_vec.y + x_minus_s.z * n_vec.z;
        const float dist_from_source = -denom;
        const float dist_from_source_inv = 1.0f / dist_from_source;
        const float t_C = c_minus_s_dot_n / denom;
        const float v_C = t_C * (x_minus_s.x * v_vec.x + x_minus_s.y * v_vec.y + x_minus_s.z * v_vec.z) - c_minus_s_dot_v;

        const float centralWeight = helicalConeWeight(v_C);
        if (centralWeight > 0.0f)
        {
            const float u_C = t_C * (x_minus_s.x * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;

            // Start: Calculate the View Redundancy Weight
            const float phi_cur_conj = phi - 2.0f * atan(u_C) + atan_term + PI;
            //const float R_minus_x_dot_theta_conj = R - x * cos(phi_cur_conj) - y * sin(phi_cur_conj);
            //const float R_minus_x_dot_theta_inv_conj = 1.0f / R_minus_x_dot_theta_conj;

            const float cos_phi_conj = cos(phi_cur_conj);
            const float sin_phi_conj = sin(phi_cur_conj);

            const float dist_from_source_conj = (R - x * cos_phi_conj - y * sin_phi_conj)*cos_pitch + sin_pitch*(z-sourcePosition.z);
            const float dist_from_source_inv_conj = 1.0f / dist_from_source_conj;

            float sumWeights = 0.0f;

            const float v_val_shift = neg_twoPI_pitch * dist_from_source_inv;

            const float v_bound_A = (v_val_shift > 0.0f) ? (v_min - v_C) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_C) * dist_from_source * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_val_shift < 0.0f) ? (v_min - v_C) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_C) * dist_from_source * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceilf((d_phi_start - phi) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi) * twoPI_inv)), int(floorf(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight(v_C + iturn * v_val_shift);
            }

            const float z_source_conj = phi_cur_conj * T_g.w + startVals_g.w;

            //const float v_val_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * dist_from_source_inv_conj;
            const float v_vec_dot_x_conj = x * (cos_phi_conj*v_vec_0.x - sin_phi_conj*v_vec_0.y) + y * (sin_phi_conj*v_vec_0.x + cos_phi_conj*v_vec_0.y);
            const float v_val_conj = (v_vec_dot_x_conj + (z-z_source_conj) * v_vec_0.z - v_val_num_shift) * dist_from_source_inv_conj;
            const float v_val_shift_conj = neg_twoPI_pitch * dist_from_source_inv_conj;

            const float v_bound_A_conj = (v_val_shift_conj > 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_val_shift_conj < 0.0f) ? (v_min - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_val_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_val_conj + iturn * v_val_shift_conj);

            sumWeights = centralWeight / (centralWeight + sumWeights);
            // End: Calculate the View Redundancy Weight

            if (fabsf(x - sourcePosition.x) > fabsf(y - sourcePosition.y))
            {
                const float v_phi_x = (v_C - startVals_g.y) * Tv_inv;

                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.y);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.y);

                const float u_arg_A = t_A * (x_minus_s.x * u_vec.x + (x_minus_s.y - half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;
                const float u_arg_B = t_B * (x_minus_s.x * u_vec.x + (x_minus_s.y + half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;

                const float l_phi = sqrtf((x_minus_s.x * x_minus_s.x + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.x * x_minus_s.x);

                // Weights for u
                const float tau_low = (min(u_arg_A, u_arg_B) - u_0) * Tu_inv;
                const float tau_high = (max(u_arg_A, u_arg_B) - u_0) * Tu_inv;

                float u_ind_first = floorf(tau_low + 0.5f); // first detector index

                const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                const float v_phi_x_step_A = t_C * (T_f.z * v_vec.z) * Tv_inv;

                const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                const float extraWeight = sqrtf(1.0f + v_C*v_C) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + max(0.0f, z_high_A - 1.0f)));

                if (z_high_A > 1.0f)
                {
                    val += ((TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (TEX3D(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * extraWeight * sumWeights;
                }
                else
                {
                    val += ((TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * extraWeight * sumWeights;
                }
            }
            else
            {
                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.x);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.x);

                const float u_arg_A = t_A * ((x_minus_s.x - half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;
                const float u_arg_B = t_B * ((x_minus_s.x + half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;

                const float l_phi = sqrtf((x_minus_s.y * x_minus_s.y + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.y * x_minus_s.y);

                // Weights for u
                const float tau_low = (min(u_arg_A, u_arg_B) - u_0) * Tu_inv;
                const float tau_high = (max(u_arg_A, u_arg_B) - u_0) * Tu_inv;

                float u_ind_first = floorf(tau_low + 0.5f); // first detector index

                const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                const float v_phi_x = (v_C - startVals_g.y) * Tv_inv;
                const float v_phi_x_step_A = t_C * (T_f.z * v_vec.z) * Tv_inv;

                const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                const float extraWeight = sqrtf(1.0f + v_C*v_C) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + max(0.0f, z_high_A - 1.0f)));

                if (z_high_A > 1.0f)
                {
                    val += ((TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (TEX3D(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * extraWeight * sumWeights;
                }
                else
                {
                    val += ((TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * extraWeight * sumWeights;
                }
            }
        }
    }

    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}
//*/

__global__ void coneBeamHelicalWeightedBackprojectorKernel_SF(TEX_DATA g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }
    const float z = k * T_f.z + startVals_f.z;

    const float T_x_over_2 = 0.5f * T_f.x;
    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float v_min = 1.0f / d_v_min_inv;
    const float v_max = 1.0f / d_v_max_inv;

    float val = 0.0f;

    const float u_conj_tau_term = 2.0f * tau * R / (R * R - tau * tau);
    const float atan_term = atan(u_conj_tau_term);
    
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    for (int l = 0; l < N_g.x; l++)
    {
        const float phi_cur = phis[l];
        const float z_source = (phi_cur * T_g.w + startVals_g.w);
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

        const float u_0 = startVals_g.z;
        const float u_min = u_0;// - 0.5f*T_g.z;
        const float u_max = T_g.z*(N_g.z-1) + u_0;// + 0.5f*T_g.z;

        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float v_arg = (z - z_source) * R_minus_x_dot_theta_inv;
        const float centralWeight = helicalConeWeight(v_arg);
        if (centralWeight > 0.0f)
        {   
            const float L = (float)l + 0.5f;

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
            const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
            //const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            const float l_phi = T_f.x / max(x_denom, y_denom);
            float A_x;
            if (x_denom > y_denom)
                A_x = fabsf(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabsf(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - u_0) * Tu_inv;
            const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - u_0) * Tu_inv;

            float ind_first = floorf(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            const float v_phi_x = (v_phi_x_start_num - z_source * Tv_inv) * R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            // Calculate the View Redundancy Weight
            float sumWeights = 0.0f;

            const float v_arg_shift = neg_twoPI_pitch * R_minus_x_dot_theta_inv;

            const float v_bound_A = (v_arg_shift > 0.0f) ? (v_min - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_arg_shift < 0.0f) ? (v_min - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceilf((d_phi_start - phi_cur) * twoPI_inv)), int(ceilf(v_bound_A)));
            const int N_turns_above = min(int(floorf((d_phi_end - phi_cur) * twoPI_inv)), int(floorf(v_bound_B)));
            sumWeights = helicalConeWeight_sum(v_arg, v_arg_shift, N_turns_below, N_turns_above);

            const float u_phi_x_conj = (u_conj_tau_term - u_arg) / (1.0 + u_arg*u_conj_tau_term);
            if (u_min <= u_phi_x_conj && u_phi_x_conj <= u_max)
            {
                const float phi_cur_conj = phi_cur - 2.0f * atan(u_arg) + atan_term + PI;
                const float R_minus_x_dot_theta_conj = R - x * cos(phi_cur_conj) - y * sin(phi_cur_conj);
                const float R_minus_x_dot_theta_inv_conj = 1.0f / R_minus_x_dot_theta_conj;

                const float v_arg_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * R_minus_x_dot_theta_inv_conj;
                const float v_arg_shift_conj = neg_twoPI_pitch * R_minus_x_dot_theta_inv_conj;

                const float v_bound_A_conj = (v_arg_shift_conj > 0.0f) ? (v_min - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;
                const float v_bound_B_conj = (v_arg_shift_conj < 0.0f) ? (v_min - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;

                const int N_turns_below_conj = max(int(ceilf((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceilf(v_bound_A_conj)));
                const int N_turns_above_conj = min(int(floorf((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floorf(v_bound_B_conj)));
                sumWeights += helicalConeWeight_sum(v_arg_conj, v_arg_shift_conj, N_turns_below_conj, N_turns_above_conj);
            }
            if (sumWeights > 0.0f)
            {
                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                //const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * R_minus_x_dot_theta_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + max(0.0f, z_high_A - 1.0f)));
                const float extraWeight = R * R_minus_x_dot_theta_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + max(0.0f, z_high_A - 1.0f)));

                val += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (TEX3D(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + TEX3D(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * max(0.0f, z_high_A - 1.0f)) * extraWeight * centralWeight / sumWeights;
            }
        }
    }

    //f[ind] = val;
    if (accum)
        f[ind] += val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
    else
        f[ind] = val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
}

__global__ void curvedConeBeamBackprojectorKernel_SF(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

    const float T_x_over_2 = 0.5f * T_f.x;
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
        const float z_source_over_T_v = (phis[l] * T_g.w + startVals_g.w) * Tv_inv;
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
        const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float dist_from_source_components_x = fabsf(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabsf(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source = sqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);
        const float l_phi = T_f.x * dist_from_source / max(dist_from_source_components_x, dist_from_source_components_y);

        const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
        const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
        //const float l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabsf(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabsf(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = (atan((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x)) - startVals_g.z) * Tu_inv;
        const float tau_high = (atan((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x)) - startVals_g.z) * Tu_inv;

        float ind_first = floorf(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float ind_last = ind_first + 2.5f;
        ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float v_phi_x_step_A = Tz_over_Tv / dist_from_source;
        const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) / dist_from_source - v0_over_Tv;
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            //const float v_phi_x = (v_phi_x_start_num + k_offset * Tz_over_Tv - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x = v_phi_x_first + k_offset * v_phi_x_step_A;

            const float v_arg = (v_phi_x + v0_over_Tv) * T_g.y;
            const float v_weight = sqrtf(1.0f + v_arg * v_arg);

            const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                vals[k_offset] += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (TEX3D(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + TEX3D(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * v_weight;
            }
            else
            {
                vals[k_offset] += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * v_weight;
            }
        }
    }

    if (accum)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] += vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset];
        }
    }
    else
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset];
        }
    }
}

__global__ void curvedConeBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, TEX_DATA f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = tan(n * T_g.z + startVals_g.z);

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float n_minus_half = (float)n - 0.5f; // +startVals_g.z / T_g.z;
    const float n_plus_half = (float)n + 0.5f; // +startVals_g.z / T_g.z;
    const float m_minus_half = (float)m - 0.5f;
    const float m_plus_half = (float)m + 0.5f;

    const float u0_over_Tu = startVals_g.z / T_g.z;
    const float one_over_Tu = 1.0f / T_g.z;

    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v0_over_Tv = startVals_g.y / T_g.y;

    const float z_source = phis[l] * T_g.w + startVals_g.w;

    const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
    const float z_ind_offset = -z0_over_Tz_plus_half + z_source / T_f.z;

    const float z_ind_slope = (v - 0.5f * T_g.y) / T_f.z;

    const float sourcePos_x = R * cos_phi + tau * sin_phi;
    const float sourcePos_y = R * sin_phi - tau * cos_phi;

    float g_output = 0.0f;

    if (fabsf(u * cos_phi - sin_phi) > fabsf(u * sin_phi + cos_phi))
    {
        const float A_x = fabsf(sin_phi) * 0.5f * T_f.x;
        const float B_x = cos_phi * 0.5f * T_f.x * ((sin_phi < 0.0f) ? -1.0f : 1.0f);
        const float Tx_sin = T_f.x * sin_phi;
        //const float Tx_cos = T_g.z * T_f.x * cos_phi; // FIXME?
        const float Tx_cos = T_f.x * cos_phi; // FIXME?

        float shiftConstant, slopeConstant;
        if (u * cos_phi - sin_phi > 0.0f)
        {
            shiftConstant = (((R + B_x) * (u - 0.5f * T_g.z) - A_x - tau) / (cos_phi * (u - 0.5f * T_g.z) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = ((-sin_phi * (u - 0.5f * T_g.z) - cos_phi) / (cos_phi * (u - 0.5f * T_g.z) - sin_phi)) / T_f.x;
        }
        else
        {
            shiftConstant = (((R - B_x) * (u + 0.5f * T_g.z) + A_x - tau) / (cos_phi * (u + 0.5f * T_g.z) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = ((sin_phi * (u + 0.5f * T_g.z) + cos_phi) / (-cos_phi * (u + 0.5f * T_g.z) + sin_phi)) / T_f.x;
        }

        for (int j = 0; j < N_f.y; j++)
        {
            const float y = (float)j * T_f.y + startVals_f.y;
            const int i = (int)ceilf(y * slopeConstant + shiftConstant);
            const float x = (float)i * T_f.x + startVals_f.x;

            if (x * x + y * y > rFOVsq)
                continue;

            const float v_denom_inv = rsqrtf((sourcePos_x - x) * (sourcePos_x - x) + (sourcePos_y - y) * (sourcePos_y - y));

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            //const int k = (int)ceilf(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);
            const int k = (int)ceilf(z_ind_slope / v_denom_inv + z_ind_offset);

            if (k <= -3)
            {
                continue;
            }
            if (k >= N_f.z)
            {
                continue;
            }

            const float num_low = tau - x * sin_phi + y * cos_phi - A_x;
            const float num_high = num_low + 2.0f * A_x;

            const float denom_low = R_minus_x_dot_theta - B_x;
            const float denom_high = R_minus_x_dot_theta + B_x;

            const float hWeight_0 = max(0.0f, min(atan(num_high / denom_high)* one_over_Tu- u0_over_Tu, n_plus_half) - max(atan(num_low / denom_low)* one_over_Tu- u0_over_Tu, n_minus_half));
            const float hWeight_1 = max(0.0f, min(atan((num_high - Tx_sin) / (denom_high - Tx_cos))* one_over_Tu- u0_over_Tu, n_plus_half) - max(atan((num_low - Tx_sin) / (denom_low - Tx_cos))* one_over_Tu- u0_over_Tu, n_minus_half));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float v_phi_x_step = Tz_over_Tv * v_denom_inv;
            const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
            const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
            const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
            const float x_12 = float(i) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);
            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(j) + 0.5f, float(i + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, x_12, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, float(i + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * cos_phi - sin_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabsf(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float A_y = fabsf(cos_phi) * 0.5f * T_f.x;
        const float B_y = sin_phi * 0.5f * T_f.x * ((cos_phi < 0.0f) ? 1.0f : -1.0f);
        const float Ty_cos = T_f.y * cos_phi;
        //const float Ty_sin = T_g.z * T_f.y * sin_phi;
        const float Ty_sin = T_f.y * sin_phi;

        float shiftConstant, slopeConstant;
        if (u * sin_phi + cos_phi >= 0.0f)
        {
            shiftConstant = (((R + B_y) * (u - 0.5f * T_g.z) - A_y - tau) / (sin_phi * (u - 0.5f * T_g.z) + cos_phi) - startVals_f.y) / T_f.y;
            slopeConstant = ((sin_phi - cos_phi * (u - 0.5f * T_g.z)) / (sin_phi * (u - 0.5f * T_g.z) + cos_phi)) / T_f.y;
        }
        else
        {
            shiftConstant = (((R - B_y) * (u + 0.5f * T_g.z) + A_y - tau) / (cos_phi + sin_phi * (u + 0.5f * T_g.z)) - startVals_f.y) / T_f.y;
            slopeConstant = ((sin_phi - cos_phi * (u + 0.5f * T_g.z)) / (cos_phi + sin_phi * (u + 0.5f * T_g.z))) / T_f.y;
        }
        for (int i = 0; i < N_f.x; i++)
        {
            const float x = (float)i * T_f.x + startVals_f.x;
            const int j = (int)ceilf(x * slopeConstant + shiftConstant);
            const float y = (float)j * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float v_denom_inv = rsqrtf((sourcePos_x - x) * (sourcePos_x - x) + (sourcePos_y - y) * (sourcePos_y - y));

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            //const int k = (int)ceilf(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);
            const int k = (int)ceilf(z_ind_slope / v_denom_inv + z_ind_offset);

            if (k <= -3)
            {
                continue;
            }
            if (k >= N_f.z)
            {
                continue;
            }

            const float num_low = tau - x * sin_phi + y * cos_phi - A_y;
            const float num_high = num_low + 2.0f * A_y;

            const float denom_low = R_minus_x_dot_theta - B_y;
            const float denom_high = R_minus_x_dot_theta + B_y;

            const float hWeight_0 = max(0.0f, min(atan(num_high / denom_high)*one_over_Tu- u0_over_Tu, n_plus_half) - max(atan(num_low / denom_low)*one_over_Tu- u0_over_Tu, n_minus_half));
            const float hWeight_1 = max(0.0f, min(atan((num_high + Ty_cos) / (denom_high - Ty_sin))*one_over_Tu- u0_over_Tu, n_plus_half) - max(atan((num_low + Ty_cos) / (denom_low - Ty_sin))*one_over_Tu- u0_over_Tu, n_minus_half));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float v_phi_x_step = Tz_over_Tv * v_denom_inv;
            const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
            const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
            const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
            const float y_12 = float(j) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);
            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(j + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(k + 2) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, float(i) + 0.5f, float(j + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * sin_phi + cos_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabsf(u * sin_phi + cos_phi) * g_output;
    }
}

__global__ void coneBeamBackprojectorKernel_rot_SF(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, int volumeDimensionOrder, bool do_helicalFBP, bool accum)
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

    const float vox_half = 0.5f * T_f.x;
    const float voxz_half = 0.5f * T_f.z;

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;
    const float v_min = (-0.5f-float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;
    const float v_max = (float(N_g.y)-0.5f+float(d_numRowsExtrapolate))*T_g.y + startVals_g.y;

    const float u_length = T_g.z*(N_g.z-1);// + 0.5f*T_g.z;

    float val = 0.0f;
    if (do_helicalFBP)
    {
        const float twoPI_inv = 1.0f / (2.0f * PI);
        const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
        const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;
        const float psi = atan(-tau/R);

        for (int iphi = 0; iphi < N_g.x; iphi++)
        {
            const float iphi_plus_half = float(iphi) + 0.5f;
            const float phi = phis[iphi];
            float cos_phi, sin_phi;
            __sincosf(phi, &sin_phi, &cos_phi);

            const float u_0 = startVals_g.z;
            const float v_0 = startVals_g.y;

            const float u_min = u_0;// - 0.5f*T_g.z;
            const float u_max = T_g.z*(N_g.z-1) + u_0;// + 0.5f*T_g.z;

            const float3 y_phi_minus_x = make_float3(R*cos_phi + tau*sin_phi - x, R*sin_phi - tau*cos_phi - y, phi * T_g.w + startVals_g.w - z);

            const float3 n_vec = make_float3(d_n_vec_0.x*cos_phi - d_n_vec_0.y*sin_phi, d_n_vec_0.x*sin_phi + d_n_vec_0.y*cos_phi, d_n_vec_0.z);
            const float3 u_vec = make_float3(d_u_vec_0.x*cos_phi - d_u_vec_0.y*sin_phi, d_u_vec_0.x*sin_phi + d_u_vec_0.y*cos_phi, d_u_vec_0.z);
            const float3 v_vec = make_float3(d_v_vec_0.x*cos_phi - d_v_vec_0.y*sin_phi, d_v_vec_0.x*sin_phi + d_v_vec_0.y*cos_phi, d_v_vec_0.z);

            const float L = dot3(y_phi_minus_x, n_vec);
            const float L_inv = 1.0f / L;
            const float u_num = -dot3(y_phi_minus_x, u_vec);
            const float u_phi_x = u_num * L_inv;
            const float v_num = -dot3(y_phi_minus_x, v_vec);
            const float v_phi_x = v_num * L_inv;

            const float centralWeight = helicalConeWeight(v_phi_x);
            if (centralWeight > 0.0f)
            {
                const float v_ind = (v_phi_x - v_0) * Tv_inv;

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
                sumWeights = helicalConeWeight_sum(v_phi_x, v_val_shift, N_turns_below, N_turns_above);

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
                    sumWeights += helicalConeWeight_sum(v_phi_conj_x, v_val_shift_conj, N_turns_below_conj, N_turns_above_conj);
                }
                if (sumWeights > 0.0f)
                {
                    const float x_denom = fabsf(u_phi_x * cos_phi - sin_phi);
                    const float y_denom = fabsf(u_phi_x * sin_phi + cos_phi);
                    //const float l_phi = T_f.x * sqrtf(1.0f + u_phi_x * u_phi_x) / max(x_denom, y_denom);
                    const float l_phi = T_f.x / max(x_denom, y_denom);
                    //const float l_phi = T_f.x * sqrtf(1.0f + u_phi_x * u_phi_x) * sqrtf(1.0f + v_phi_x * v_phi_x) / max(x_denom, y_denom);

                    const float v_A = (v_phi_x - voxz_half * L_inv - v_0) * Tv_inv;
                    const float v_B = (v_phi_x + voxz_half * L_inv - v_0) * Tv_inv;

                    //if (x_denom > y_denom)
                    if (fabsf(y_phi_minus_x.x) < fabsf(y_phi_minus_x.y)) // smoothest
                    {
                        //const float z_A = ((v - 0.5f * T_g.y) * rayParam_inv - startVals_f.z) * T_z_inv;
                        //const float z_B = ((v + 0.5f * T_g.y) * rayParam_inv - startVals_f.z) * T_z_inv;
                        const float u_A = ((u_num + sin_phi * vox_half) / (L + vox_half * cos_phi) - u_0) * Tu_inv;
                        const float u_B = ((u_num - sin_phi * vox_half) / (L - vox_half * cos_phi) - u_0) * Tu_inv;

                        const float u_lo = min(u_A, u_B);
                        const float u_hi = max(u_A, u_B);

                        const float helicalWeight = centralWeight / sumWeights * L_inv / (R * l_phi * (u_hi - u_lo) * fabsf(v_B - v_A));
                        const float all_weights = l_phi * helicalWeight;

                        // Start: Calculate Footprint
                        float u_ind_first = floorf(u_lo + 0.5f); // first detector index

                        const float horizontalWeights_0_A = (min(u_hi, u_ind_first + 1.5f) - u_lo) * all_weights;
                        const float horizontalWeights_1_A = all_weights * (u_hi - u_lo) - horizontalWeights_0_A;

                        const float u_ind_last = u_ind_first + 2.5f;
                        u_ind_first = u_ind_first + 0.5f + max(0.0f, min(u_hi - u_ind_first - 0.5f, 1.0f)) * all_weights / horizontalWeights_0_A;

                        const float v_phi_x_step_A = L_inv * (T_f.z * v_vec.z) * Tv_inv;

                        const float row_high_A = floorf(v_ind - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                        const float z_high_A = v_ind + 0.5f * v_phi_x_step_A - row_high_A;

                        const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                        const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                        const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                        const float row_high_plus_two_A = row_high_A + 2.0f;
                        // End: Calculate Footprint

                        if (z_high_A > 1.0f)
                        {
                            val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                                + (TEX3D(g, u_ind_first, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_0_A
                                    + TEX3D(g, u_ind_last, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                        }
                        else
                        {
                            val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                        }
                    }
                    else
                    {
                        // use y_lo, y_hi
                        const float u_A = ((u_num - cos_phi * vox_half) / (L + vox_half * sin_phi) - u_0) * Tu_inv;
                        const float u_B = ((u_num + cos_phi * vox_half) / (L - vox_half * sin_phi) - u_0) * Tu_inv;

                        const float u_lo = min(u_A, u_B);
                        const float u_hi = max(u_A, u_B);

                        const float helicalWeight = centralWeight / sumWeights * L_inv / (R * l_phi * (u_hi - u_lo) * fabsf(v_B - v_A));
                        const float all_weights = l_phi * helicalWeight;

                        // Start: Calculate Footprint
                        float u_ind_first = floorf(u_lo + 0.5f); // first detector index

                        const float horizontalWeights_0_A = (min(u_hi, u_ind_first + 1.5f) - u_lo) * all_weights;
                        const float horizontalWeights_1_A = all_weights * (u_hi - u_lo) - horizontalWeights_0_A;

                        const float u_ind_last = u_ind_first + 2.5f;
                        u_ind_first = u_ind_first + 0.5f + max(0.0f, min(u_hi - u_ind_first - 0.5f, 1.0f)) * all_weights / horizontalWeights_0_A;

                        const float v_phi_x_step_A = L_inv * (T_f.z * v_vec.z) * Tv_inv;

                        const float row_high_A = floorf(v_ind - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                        const float z_high_A = v_ind + 0.5f * v_phi_x_step_A - row_high_A;

                        const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                        const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                        const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                        const float row_high_plus_two_A = row_high_A + 2.0f;
                        // End: Calculate Footprint

                        if (z_high_A > 1.0f)
                        {
                            val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                                + (TEX3D(g, u_ind_first, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_0_A
                                    + TEX3D(g, u_ind_last, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                        }
                        else
                        {
                            val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                        }
                    }

                    //val += TEX3D(g, (u_phi_x - startVals_g.z) * Tu_inv + 0.5f, (v_phi_x - startVals_g.y) * Tv_inv + 0.5f, iphi + 0.5f) * L_inv * L_inv * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
                }
            }
        }
        val *= (T_f.x * T_f.y * T_f.z) * (Tv_inv * Tu_inv);
    }
    else
    {
        const float football_param = (d_R_tau-sqrtf(x * x + y * y))/d_R_tau;
        const float psi = atan(-tau/R);
        bool do_corner_patching = false;
        if (d_zFOV.x != d_zFOV.y && (d_zFOV.y*football_param < z || z < d_zFOV.x*football_param))
            do_corner_patching = true;
        if (d_doDBP)
            do_corner_patching = true;

        for (int iphi = 0; iphi < N_g.x; iphi++)
        {
            const float iphi_plus_half = float(iphi) + 0.5f;
            const float phi = phis[iphi];
            float cos_phi, sin_phi;
            __sincosf(phi, &sin_phi, &cos_phi);

            const float u_0 = startVals_g.z;

            const float3 y_phi_minus_x = make_float3(R*cos_phi + tau*sin_phi - x, R*sin_phi - tau*cos_phi - y, phi * T_g.w + startVals_g.w - z);

            const float3 n_vec = make_float3(d_n_vec_0.x*cos_phi - d_n_vec_0.y*sin_phi, d_n_vec_0.x*sin_phi + d_n_vec_0.y*cos_phi, d_n_vec_0.z);
            const float3 u_vec = make_float3(d_u_vec_0.x*cos_phi - d_u_vec_0.y*sin_phi, d_u_vec_0.x*sin_phi + d_u_vec_0.y*cos_phi, d_u_vec_0.z);
            const float3 v_vec = make_float3(d_v_vec_0.x*cos_phi - d_v_vec_0.y*sin_phi, d_v_vec_0.x*sin_phi + d_v_vec_0.y*cos_phi, d_v_vec_0.z);

            const float L = dot3(y_phi_minus_x, n_vec);
            const float L_inv = 1.0f / L;
            const float u_num = -dot3(y_phi_minus_x, u_vec);
            const float u_phi_x = u_num * L_inv;
            const float v_num = -dot3(y_phi_minus_x, v_vec);
            const float v_phi_x = v_num * L_inv;

            if (v_min <= v_phi_x && v_phi_x <= v_max)
            {
                float parker_weight = 1.0f;
                if (do_corner_patching)
                {
                    const float phi_conj = phis[iphi] + 2.0f*(atan2(-d_n_vec_0.y + u_phi_x*d_u_vec_0.y + v_phi_x*d_v_vec_0.y, -d_n_vec_0.x + u_phi_x*d_u_vec_0.x + v_phi_x*d_v_vec_0.x) - psi) + PI;
                    float cos_phi_conj, sin_phi_conj;
                    __sincosf(phi_conj, &sin_phi_conj, &cos_phi_conj);

                    const float3 y_phi_conj_minus_x = make_float3(R*cos_phi_conj + tau*sin_phi_conj - x, R*sin_phi_conj - tau*cos_phi_conj - y, phi_conj * T_g.w + startVals_g.w - z);
                    const float3 n_vec_conj = make_float3(d_n_vec_0.x*cos_phi_conj - d_n_vec_0.y*sin_phi_conj, d_n_vec_0.x*sin_phi_conj + d_n_vec_0.y*cos_phi_conj, d_n_vec_0.z);
                    const float3 v_vec_conj = make_float3(d_v_vec_0.x*cos_phi_conj - d_v_vec_0.y*sin_phi_conj, d_v_vec_0.x*sin_phi_conj + d_v_vec_0.y*cos_phi_conj, d_v_vec_0.z);
                    const float L_conj = dot3(y_phi_conj_minus_x, n_vec_conj);
                    const float L_conj_inv = 1.0f / L_conj;
                    const float v_phi_conj_x = -dot3(y_phi_conj_minus_x, v_vec_conj) * L_conj_inv;

                    const float3 u_vec_conj = make_float3(d_u_vec_0.x*cos_phi_conj - d_u_vec_0.y*sin_phi_conj, d_u_vec_0.x*sin_phi_conj + d_u_vec_0.y*cos_phi_conj, d_u_vec_0.z);
                    const float u_phi_conj_x = -dot3(y_phi_conj_minus_x, u_vec_conj) * L_conj_inv;

                    //if (v_min <= v_phi_conj_x && v_phi_conj_x <= v_max)
                    //    parker_weight = (1.0f + L * L_conj_inv) * 20.0f;
                    //*
                    if (v_min > v_phi_conj_x || v_phi_conj_x > v_max || u_0 > u_phi_conj_x | u_phi_conj_x > u_0 + u_length)
                        parker_weight = 1.0f + L * L_conj_inv;
                    //*/
                }

                const float x_denom = fabsf(u_phi_x * cos_phi - sin_phi);
                const float y_denom = fabsf(u_phi_x * sin_phi + cos_phi);
                const float l_phi = T_f.x * parker_weight * sqrtf(1.0f + u_phi_x * u_phi_x) * sqrtf(1.0f + v_phi_x * v_phi_x) / max(x_denom, y_denom);

                //const float v_A = (v_phi_x - voxz_half * L_inv - startVals_g.y) * Tv_inv;
                //const float v_B = (v_phi_x + voxz_half * L_inv - startVals_g.y) * Tv_inv;
                const float v_ind = (v_phi_x - startVals_g.y) * Tv_inv;

                //if (x_denom > y_denom)
                if (fabsf(y_phi_minus_x.x) < fabsf(y_phi_minus_x.y))
                {
                    //const float z_A = ((v - 0.5f * T_g.y) * raystartVals_g.zParam_inv - startVals_f.z) * T_z_inv;
                    //const float z_B = ((v + 0.5f * T_g.y) * rayParam_inv - startVals_f.z) * T_z_inv;
                    const float u_A = ((u_num + sin_phi * vox_half) / (L + vox_half * cos_phi) - u_0) * Tu_inv;
                    const float u_B = ((u_num - sin_phi * vox_half) / (L - vox_half * cos_phi) - u_0) * Tu_inv;

                    const float u_lo = min(u_A, u_B);
                    const float u_hi = max(u_A, u_B);

                    // Start: Calculate Footprint
                    float u_ind_first = floorf(u_lo + 0.5f); // first detector index

                    const float horizontalWeights_0_A = (min(u_hi, u_ind_first + 1.5f) - u_lo) * l_phi;
                    const float horizontalWeights_1_A = l_phi * (u_hi - u_lo) - horizontalWeights_0_A;

                    const float u_ind_last = u_ind_first + 2.5f;
                    u_ind_first = u_ind_first + 0.5f + max(0.0f, min(u_hi - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                    const float v_phi_x_step_A = L_inv * (T_f.z * v_vec.z) * Tv_inv;
                    //const float v_phi_x_step_A = L_inv * (T_f.z) * Tv_inv;

                    const float row_high_A = floorf(v_ind - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                    const float z_high_A = v_ind + 0.5f * v_phi_x_step_A - row_high_A;

                    const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                    const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                    const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                    const float row_high_plus_two_A = row_high_A + 2.0f;
                    // End: Calculate Footprint

                    if (z_high_A > 1.0f)
                    {
                        val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                            + (TEX3D(g, u_ind_first, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                    }
                    else
                    {
                        val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                    }
                }
                else
                {
                    // use y_lo, y_hi
                    const float u_A = ((u_num - cos_phi * vox_half) / (L + vox_half * sin_phi) - u_0) * Tu_inv;
                    const float u_B = ((u_num + cos_phi * vox_half) / (L - vox_half * sin_phi) - u_0) * Tu_inv;

                    const float u_lo = min(u_A, u_B);
                    const float u_hi = max(u_A, u_B);

                    // Start: Calculate Footprint
                    float u_ind_first = floorf(u_lo + 0.5f); // first detector index

                    const float horizontalWeights_0_A = (min(u_hi, u_ind_first + 1.5f) - u_lo) * l_phi;
                    const float horizontalWeights_1_A = l_phi * (u_hi - u_lo) - horizontalWeights_0_A;

                    const float u_ind_last = u_ind_first + 2.5f;
                    u_ind_first = u_ind_first + 0.5f + max(0.0f, min(u_hi - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                    const float v_phi_x_step_A = L_inv * (T_f.z * v_vec.z) * Tv_inv;
                    //const float v_phi_x_step_A = L_inv * (T_f.z) * Tv_inv;

                    const float row_high_A = floorf(v_ind - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                    const float z_high_A = v_ind + 0.5f * v_phi_x_step_A - row_high_A;

                    const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                    const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                    const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                    const float row_high_plus_two_A = row_high_A + 2.0f;
                    // End: Calculate Footprint

                    if (z_high_A > 1.0f)
                    {
                        val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                            + (TEX3D(g, u_ind_first, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_0_A
                                + TEX3D(g, u_ind_last, row_high_plus_two_A, iphi_plus_half) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                    }
                    else
                    {
                        val += (TEX3D(g, u_ind_first, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_0_A
                            + TEX3D(g, u_ind_last, row_high_A + v_oneAndTwo, iphi_plus_half) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                    }
                }

                //val += TEX3D(g, (u_phi_x - startVals_g.z) * Tu_inv + 0.5f, (v_phi_x - startVals_g.y) * Tv_inv + 0.5f, iphi + 0.5f) * L_inv * L_inv * sqrtf(1.0f + u_phi_x * u_phi_x + v_phi_x * v_phi_x);
            }
        }
    }
    if (accum)
        f[ind] += val;
    else
        f[ind] = val;
}

__global__ void coneBeamBackprojectorKernel_SF(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = (threadIdx.z + blockIdx.z * blockDim.z)*NUM_SLICES_PER_THREAD;
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

    const float T_x_over_2 = 0.5f * T_f.x;
    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    //*
    const float v_min_ind = -0.5f-float(d_numRowsExtrapolate);
    const float v_max_ind = float(N_g.y)-0.5f+float(d_numRowsExtrapolate);
    const float v_min = v_min_ind*T_g.y + startVals_g.y;
    const float v_max = v_max_ind*T_g.y + startVals_g.y;
    //*/

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    if (tiltAngle == 0.0f)
    {
        const float z = k * T_f.z + startVals_f.z;
        const float v_phi_x_start_num = z / T_g.y;
        for (int l = 0; l < N_g.x; l++)
        {
            const float L = (float)l + 0.5f;
            const float z_source_over_T_v = (phis[l] * T_g.w + startVals_g.w) * Tv_inv;
            const float sin_phi = sin(phis[l]);
            const float cos_phi = cos(phis[l]);

            const float u_0 = startVals_g.z;

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
            const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
            const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);

            float A_x;
            if (x_denom > y_denom)
                A_x = fabsf(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabsf(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - u_0) * Tu_inv;
            const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - u_0) * Tu_inv;

            float ind_first = floorf(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;
            const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv;
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                //const float v_phi_x = (v_phi_x_start_num + k_offset * Tz_over_Tv - z_source_over_T_v) * R_minus_x_dot_theta_inv - v0_over_Tv;
                const float v_phi_x = v_phi_x_first + k_offset * v_phi_x_step_A;
                const float v_arg = (v_phi_x + v0_over_Tv) * T_g.y;

                if (v_arg < v_min || v_max < v_arg)
                    continue;

                const float v_weight = sqrtf(1.0f + v_arg * v_arg);

                const float row_high_A = floorf(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                if (z_high_A > 1.0f)
                {
                    vals[k_offset] += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (TEX3D(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + TEX3D(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * v_weight;
                }
                else
                {
                    vals[k_offset] += ((TEX3D(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + TEX3D(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * v_weight;
                }
            }
        }
    }
    else
    {
        const float voxz_half = 0.5f * T_f.z;
        const float cos_tilt = cos(tiltAngle);
        const float sin_tilt = sin(tiltAngle);
        for (int l = 0; l < N_g.x; l++)
        {
            const float L = (float)l + 0.5f;
            const float z_source = phis[l] * T_g.w + startVals_g.w;
            const float sin_phi = sin(phis[l]);
            const float cos_phi = cos(phis[l]);

            const float u_0 = startVals_g.z;

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = (k+k_offset) * T_f.z + startVals_f.z;

                const float u_num = x_dot_theta_perp * cos_tilt + (z-z_source) * sin_tilt;
                const float v_num = (z-z_source) * cos_tilt - x_dot_theta_perp * sin_tilt;

                const float u_arg = u_num * R_minus_x_dot_theta_inv;
                const float v_arg = v_num * R_minus_x_dot_theta_inv;

                if (v_arg < v_min || v_max < v_arg)
                    continue;

                const float x_denom = fabsf(u_arg * cos_phi - sin_phi);
                const float y_denom = fabsf(u_arg * sin_phi + cos_phi);
                const float l_phi = T_f.x * sqrtf((1.0f + u_arg * u_arg) * (1.0f + v_arg * v_arg)) / max(x_denom, y_denom);

                // Calculate footprint along columns
                float A_x;
                if (x_denom > y_denom)
                    A_x = fabsf(sin_phi) * T_x_over_2;
                else
                {
                    A_x = fabsf(cos_phi) * T_x_over_2;
                    B_x = B_y;
                }
                const float tau_low = ((u_num - A_x) / (R_minus_x_dot_theta - B_x) - u_0) * Tu_inv;
                const float tau_high = ((u_num + A_x) / (R_minus_x_dot_theta + B_x) - u_0) * Tu_inv;

                float u_ind_first = floorf(tau_low + 0.5f); // first detector index

                const float uWeights_0 = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float uWeights_1 = l_phi * (tau_high - tau_low) - uWeights_0;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / uWeights_0;

                // Calculate footprint along rows
                const float v_A = (v_arg - voxz_half * R_minus_x_dot_theta_inv - startVals_g.y) * Tv_inv;
                const float v_B = (v_arg + voxz_half * R_minus_x_dot_theta_inv - startVals_g.y) * Tv_inv;

                float v_ind_first = floorf(v_A + 0.5f); // first detector index

                const float vWeights_0 = (min(v_B, v_ind_first + 1.5f) - v_A);
                const float vWeights_1 = (v_B - v_A) - vWeights_0;

                const float v_ind_last = v_ind_first + 2.5f;
                v_ind_first = v_ind_first + 0.5f + max(0.0f, min(v_B - v_ind_first - 0.5f, 1.0f)) / vWeights_0;

                if (vWeights_1 > 0.0f)
                {
                    vals[k_offset] += (TEX3D(g, u_ind_first, v_ind_first, L) * uWeights_0
                        + TEX3D(g, u_ind_last, v_ind_first, L) * uWeights_1) * vWeights_0
                        + (TEX3D(g, u_ind_first, v_ind_last, L) * uWeights_0
                            + TEX3D(g, u_ind_last, v_ind_last, L) * uWeights_1) * vWeights_1;
                }
                else
                {
                    vals[k_offset] += (TEX3D(g, u_ind_first, v_ind_first, L) * uWeights_0
                        + TEX3D(g, u_ind_last, v_ind_first, L) * uWeights_1) * vWeights_0;
                }
            }
        }
    }

    if (accum)
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] += vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset];
        }
    }
    else
    {
        if (volumeDimensionOrder == 0)
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset)] = vals[k_offset];
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset];
        }
    }
}

__global__ void coneBeamProjectorKernel_pitch_SF(float* g, const int4 N_g, const float4 T_g, const float4 startVals_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float sdd, const float tau, const float tiltAngle, const float pitchAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    //if (l == 0 && m == N_g.y/2 && n == N_g.z/2)
    //    printf("hello from coneBeamProjectorKernel_pitch_SF\n");

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float cos_tilt = cos(tiltAngle);
    const float sin_tilt = sin(tiltAngle);

    const float cos_pitch = cos(pitchAngle);
    const float sin_pitch = sin(pitchAngle);

    const float3 u_vec_0 = make_float3(sin_pitch*sin_tilt, cos_tilt, cos_pitch*sin_tilt);
    const float3 v_vec_0 = make_float3(sin_pitch*cos_tilt, -sin_tilt, cos_pitch*cos_tilt);

    const float3 p = make_float3(R*cos_phi + tau*sin_phi, R*sin_phi - tau*cos_phi, phis[l] * T_g.w + startVals_g.w);
    const float3 v_vec = make_float3(cos_phi*v_vec_0.x - sin_phi*v_vec_0.y, sin_phi*v_vec_0.x + cos_phi*v_vec_0.y, v_vec_0.z);
    const float3 u_vec = make_float3(cos_phi*u_vec_0.x - sin_phi*u_vec_0.y, sin_phi*u_vec_0.x + cos_phi*u_vec_0.y, u_vec_0.z);
    const float3 moduleCenter = make_float3(p.x - cos_phi*cos_pitch*sdd, p.y - sin_phi*cos_pitch*sdd, p.z + sin_pitch*sdd);

    //const float3 detNormal = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
    //    u_vec.z * v_vec.x - u_vec.x * v_vec.z,
    //    u_vec.x * v_vec.y - u_vec.y * v_vec.x);

    // These are just needed to calculate the footprint width but not the location
    const float u_vec_flat_normalizer = rsqrtf(u_vec.x * u_vec.x + u_vec.y * u_vec.y);
    const float3 u_vec_flat = make_float3(u_vec.x* u_vec_flat_normalizer, u_vec.y* u_vec_flat_normalizer, 0.0f);

    const float t = m * T_g.y + startVals_g.y; // row
    const float s = n * T_g.z + startVals_g.z; // column

    const float m_pos = float(m) + 0.5f;
    const float m_neg = float(m) - 0.5f;
    const float n_pos = float(n) + 0.5f;
    const float n_neg = float(n) - 0.5f;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;
    const float T_x_inv = 1.0f / T_f.x;
    const float T_y_inv = 1.0f / T_f.y;
    const float T_z_inv = 1.0f / T_f.z;

    const float3 detPos = make_float3(moduleCenter.x + u_vec.x * s + v_vec.x * t, moduleCenter.y + u_vec.y * s + v_vec.y * t, moduleCenter.z + u_vec.z * s + v_vec.z * t);
    const float3 r = make_float3(detPos.x - p.x, detPos.y - p.y, detPos.z - p.z);
    const float D = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

    const float3 p_minus_c = make_float3(p.x - moduleCenter.x, p.y - moduleCenter.y, p.z - moduleCenter.z);

    const float p_minus_c_dot_u = p_minus_c.x * u_vec.x + p_minus_c.y * u_vec.y + p_minus_c.z * u_vec.z;
    const float p_minus_c_dot_v = p_minus_c.x * v_vec.x + p_minus_c.y * v_vec.y + p_minus_c.z * v_vec.z;

    float g_output = 0.0f;

    // Line Integral: p + t*r
    if (fabsf(r.y) > fabsf(r.x))
    {
        const float r_y_inv = 1.0f / r.y;
        for (int j = 0; j < N_f.y; j++)
        {
            const float y = (float)j * T_f.y + startVals_f.y;
            const float x = p.x + (y - p.y) * r_y_inv * r.x;
            if (x * x + y * y > rFOVsq)
                continue;
            const float z = p.z + (y - p.y) * r_y_inv * r.z;

            // Calculate the index and position of central voxel
            const int ix = int(0.5f + (x - startVals_f.x) * T_x_inv);
            const int iz = int(0.5f + (z - startVals_f.z) * T_z_inv);
            const float x_c = ix * T_f.x + startVals_f.x;
            const float z_c = iz * T_f.z + startVals_f.z;

            // consider: three x positions and three z positions
            const float vox_dist_inv = rsqrtf((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z));
            const float t = D * vox_dist_inv;

            const float iu_c = (p_minus_c_dot_u + t * ((x_c - p.x) * u_vec.x + (y - p.y) * u_vec.y + (z_c - p.z) * u_vec.z) - startVals_g.z) * T_u_inv;
            const float iv_c = (p_minus_c_dot_v + t * ((x_c - p.x) * v_vec.x + (y - p.y) * v_vec.y + (z_c - p.z) * v_vec.z) - startVals_g.y) * T_v_inv;

            const float horizontal_footprint_half_width = 0.5f * T_f.x * t * fabsf(u_vec_flat.x) * T_u_inv;
            const float vertical_footprint_half_width = 0.5f * T_f.z * t * T_v_inv;

            float hWeight_0;
            if (u_vec_flat.x > 0.0f)
                hWeight_0 = max(0.0f, min(n_pos, iu_c - horizontal_footprint_half_width) - max(n_neg, iu_c - 2.0f * horizontal_footprint_half_width));
            else
                hWeight_0 = max(0.0f, min(n_pos, iu_c + 2.0f * horizontal_footprint_half_width) - max(n_neg, iu_c + horizontal_footprint_half_width));
            const float hWeight_1 = max(0.0f, min(n_pos, iu_c + horizontal_footprint_half_width) - max(n_neg, iu_c - horizontal_footprint_half_width));
            //const float hWeight_2 = max(0.0f, min(n_pos, iu_c + 2.0f*horizontal_footprint_half_width) - max(n_neg, iu_c + horizontal_footprint_half_width));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float vWeight_0 = max(0.0f, min(m_pos, iv_c - vertical_footprint_half_width) - max(m_neg, iv_c - 2.0f * vertical_footprint_half_width));
            const float vWeight_1 = max(0.0f, min(m_pos, iv_c + vertical_footprint_half_width) - max(m_neg, iv_c - vertical_footprint_half_width));
            //const float vWeight_2 = max(0.0f, min(m_pos, iv_c + 2.0f * vertical_footprint_half_width) - max(m_neg, iv_c + vertical_footprint_half_width));
            const float vWeight_2 = max(0.0f, 1.0f - vWeight_1 - vWeight_0);

            const float x_12 = float(ix-1) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(iz-1) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);

            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(iz + 1) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(j) + 0.5f, float(ix + 1) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(iz + 1) + 0.5f, float(j) + 0.5f, float(ix + 1) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                if (hWeight_2 == 0.0f)
                {
                    g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, x_12, float(j) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1);
                }
                else
                {
                    g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, x_12, float(j) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                             + (TEX3D(f, float(ix + 1) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, float(ix + 1) + 0.5f, float(j) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * hWeight_2;
                }
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf((r.y*r.y + r.x*r.x)*(r.y*r.y + r.z*r.z)) * fabsf(r_y_inv*r_y_inv) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf((r.y*r.y + r.x*r.x)*(r.y*r.y + r.z*r.z)) * fabsf(r_y_inv*r_y_inv) * g_output;
    }
    else
    {
        const float r_x_inv = 1.0f / r.x;
        for (int i = 0; i < N_f.x; i++)
        {
            const float x = (float)i * T_f.x + startVals_f.x;
            const float y = p.y + (x - p.x) * r_x_inv * r.y;
            if (x * x + y * y > rFOVsq)
                continue;

            const float z = p.z + (x - p.x) * r_x_inv * r.z;

            // Calculate the index and position of central voxel
            const int iy = int(0.5f + (y - startVals_f.y) * T_y_inv);
            const int iz = int(0.5f + (z - startVals_f.z) * T_z_inv);
            const float y_c = iy * T_f.y + startVals_f.y;
            const float z_c = iz * T_f.z + startVals_f.z;

            // consider: three x positions and three z positions
            const float vox_dist_inv = rsqrtf((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z));
            const float t = D * vox_dist_inv;

            const float iu_c = (p_minus_c_dot_u + t * ((x - p.x) * u_vec.x + (y_c - p.y) * u_vec.y + (z_c - p.z) * u_vec.z) - startVals_g.z) * T_u_inv;
            const float iv_c = (p_minus_c_dot_v + t * ((x - p.x) * v_vec.x + (y_c - p.y) * v_vec.y + (z_c - p.z) * v_vec.z) - startVals_g.y) * T_v_inv;

            const float horizontal_footprint_half_width = 0.5f * T_f.y * t * fabsf(u_vec_flat.y) * T_u_inv;
            const float vertical_footprint_half_width = 0.5f * T_f.z * t * T_v_inv;

            //const float hWeight_0 = max(0.0f, min(n_pos, iu_c - horizontal_footprint_half_width) - max(n_neg, iu_c - 2.0f * horizontal_footprint_half_width));
            float hWeight_0;
            if (u_vec_flat.y > 0.0f)
                hWeight_0 = max(0.0f, min(n_pos, iu_c - horizontal_footprint_half_width) - max(n_neg, iu_c - 2.0f * horizontal_footprint_half_width));
            else
                hWeight_0 = max(0.0f, min(n_pos, iu_c + 2.0f * horizontal_footprint_half_width) - max(n_neg, iu_c + horizontal_footprint_half_width));
            const float hWeight_1 = max(0.0f, min(n_pos, iu_c + horizontal_footprint_half_width) - max(n_neg, iu_c - horizontal_footprint_half_width));
            //const float hWeight_2 = max(0.0f, min(n_pos, iu_c + 2.0f * horizontal_footprint_half_width) - max(n_neg, iu_c + horizontal_footprint_half_width));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float vWeight_0 = max(0.0f, min(m_pos, iv_c - vertical_footprint_half_width) - max(m_neg, iv_c - 2.0f * vertical_footprint_half_width));
            const float vWeight_1 = max(0.0f, min(m_pos, iv_c + vertical_footprint_half_width) - max(m_neg, iv_c - vertical_footprint_half_width));
            //const float vWeight_2 = max(0.0f, min(m_pos, iv_c + 2.0f * vertical_footprint_half_width) - max(m_neg, iv_c + vertical_footprint_half_width));
            const float vWeight_2 = max(0.0f, 1.0f - vWeight_1 - vWeight_0);

            const float y_12 = float(iy-1) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
            const float z_12 = float(iz-1) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);

            if (volumeDimensionOrder == 0)
            {
                g_output += (TEX3D(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + TEX3D(f, float(iz + 1) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (TEX3D(f, z_12, float(iy + 1) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(iz + 1) + 0.5f, float(iy + 1) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                if (hWeight_2 == 0.0f)
                {
                    g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, float(i) + 0.5f, y_12, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1);
                }
                else
                {
                    g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, float(i) + 0.5f, y_12, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                             + (TEX3D(f, float(i) + 0.5f, float(iy + 1) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, float(i) + 0.5f, float(iy + 1) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * hWeight_2;
                }
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf((r.x*r.x + r.y*r.y)*(r.x*r.x + r.z*r.z)) * fabsf(r_x_inv*r_x_inv) * g_output;
        else
            g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf((r.x*r.x + r.y*r.y)*(r.x*r.x + r.z*r.z)) * fabsf(r_x_inv*r_x_inv) * g_output;
    }
}

__global__ void coneBeamProjectorKernel_SF(float* g, const int4 N_g, const float4 T_g, const float4 startVals_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum, int g_rowStride)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float cos_tilt = cos(tiltAngle);
    const float sin_tilt = sin(tiltAngle);

    /*
    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const float n_minus_half = (float)n - 0.5f + startVals_g.z / T_g.z;
    const float n_plus_half = (float)n + 0.5f + startVals_g.z / T_g.z;
    const float m_minus_half = (float)m - 0.5f;
    const float m_plus_half = (float)m + 0.5f;
    //*/

    //*
    const float v_no_tilt = m * T_g.y + startVals_g.y;
    const float u_no_tilt = n * T_g.z + startVals_g.z;
    
    const float u = cos_tilt * u_no_tilt - sin_tilt * v_no_tilt;
    const float v = sin_tilt * u_no_tilt + cos_tilt * v_no_tilt;

    //const float n_tilt = (u - startVals_g.z) / T_g.z;
    //const float m_tilt = (v - startVals_g.y) / T_g.y;

    const float n_minus_half = u/T_g.z - 0.5f;
    const float n_plus_half = u/T_g.z + 0.5f;
    const float m_minus_half = (v - startVals_g.y) / T_g.y - 0.5f;
    const float m_plus_half = (v - startVals_g.y) / T_g.y + 0.5f;
    //*/

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float v0_over_Tv = startVals_g.y / T_g.y;

    const float z_source = phis[l] * T_g.w + startVals_g.w;

    const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
    const float z_ind_offset = -z0_over_Tz_plus_half + z_source/T_f.z;

    const float z_ind_slope = (v - 0.5f*T_g.y) / T_f.z;

    float g_output = 0.0f;
    
     if (fabsf(u*cos_phi-sin_phi) > fabsf(u*sin_phi+cos_phi))
     {
         const float A_x = fabsf(sin_phi) * 0.5f*T_f.x;
         const float B_x = cos_phi * 0.5f*T_f.x * ((sin_phi < 0.0f) ? -1.0f : 1.0f);
         const float Tx_sin = T_f.x*sin_phi;
         const float Tx_cos = T_g.z*T_f.x*cos_phi;

         float shiftConstant, slopeConstant;
         if (u*cos_phi - sin_phi > 0.0f)
         {
             shiftConstant = (((R + B_x)*(u - 0.5f*T_g.z) - A_x - tau) / (cos_phi*(u-0.5f*T_g.z)-sin_phi) - startVals_f.x) / T_f.x;
             slopeConstant = (-sin_phi*(u-0.5f*T_g.z)-cos_phi) / (T_f.x*(cos_phi*(u-0.5f*T_g.z)-sin_phi) );
         }
         else
         {
             shiftConstant = (((R - B_x)*(u + 0.5f*T_g.z) + A_x - tau) / (cos_phi*(u+0.5f*T_g.z)-sin_phi) - startVals_f.x) / T_f.x;
             slopeConstant = (sin_phi*(u+0.5f*T_g.z)+cos_phi) / (T_f.x*(-cos_phi*(u+0.5f*T_g.z)+sin_phi) );
         }

         for (int j = 0; j < N_f.y; j++)
         {
             const float y = (float)j * T_f.y + startVals_f.y;
             const int i = (int)ceilf(y * slopeConstant +  shiftConstant);
             const float x = (float)i * T_f.x + startVals_f.x;

            if (x*x + y*y > rFOVsq)
                continue;

             const float R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
             const int k = (int)ceilf(  z_ind_slope*R_minus_x_dot_theta + z_ind_offset  );

             if (k <= -3)
             {
                 if (z_ind_slope*sin_phi > 0.0f)
                     break;
                 else
                     continue;
             }
             if (k >= N_f.z)
             {
                 if (z_ind_slope*sin_phi < 0.0f)
                     break;
                 else
                     continue;
             }

             const float num_low = tau - x*sin_phi + y*cos_phi - A_x;
             const float num_high = num_low + 2.0f*A_x;

             const float denom_low = (R_minus_x_dot_theta - B_x) * T_g.z;
             const float denom_high = (R_minus_x_dot_theta + B_x) * T_g.z;

             const float hWeight_0 = max(0.0f, min( num_high/denom_high, n_plus_half ) - max( num_low/denom_low, n_minus_half ) );
             const float hWeight_1 = max(0.0f, min( (num_high-Tx_sin)/(denom_high-Tx_cos), n_plus_half ) - max( (num_low-Tx_sin)/(denom_low-Tx_cos), n_minus_half ) );
             const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

             const float v_phi_x_step = T_f.z / (T_g.y*R_minus_x_dot_theta);
             const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

             const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
             const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
             const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             const float x_12 = float(i) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
             const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);
             if (volumeDimensionOrder == 0)
             {
                 g_output += (TEX3D(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                     + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                     + (TEX3D(f, z_12, float(j) + 0.5f, float(i + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                         + TEX3D(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i + 2) + 0.5f) * vWeight_2) * hWeight_2;
             }
             else
             {
                if (hWeight_2 == 0.0f)
                {
                    g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, x_12, float(j) + 0.5f, float(k+2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1);
                }
                else
                {
                    g_output += (TEX3D(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, x_12, float(j) + 0.5f, float(k+2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                             + (TEX3D(f, float(i + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                             + TEX3D(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
                }
             }
         }
         if (accum)
             g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * cos_phi - sin_phi) * g_output;
         else
             g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f+u*u) / fabsf(u*cos_phi-sin_phi) * g_output;
     }
     else
     {
         const float A_y = fabsf(cos_phi) * 0.5f*T_f.x;
         const float B_y = sin_phi * 0.5f*T_f.x * ((cos_phi < 0.0f) ? 1.0f : -1.0f);
         const float Ty_cos = T_f.y*cos_phi;
         const float Ty_sin = T_g.z*T_f.y*sin_phi;

         float shiftConstant, slopeConstant;
         if (u*sin_phi + cos_phi >= 0.0f)
         {
             shiftConstant = (((R + B_y)*(u - 0.5f*T_g.z) - A_y - tau) / (sin_phi*(u-0.5f*T_g.z)+cos_phi) - startVals_f.y) / T_f.y;
             slopeConstant = (sin_phi-cos_phi*(u-0.5f*T_g.z)) / (T_f.y*(sin_phi*(u-0.5f*T_g.z)+cos_phi) );
         }
         else
         {
             shiftConstant = (((R - B_y)*(u + 0.5f*T_g.z) + A_y - tau) / (cos_phi+sin_phi*(u+0.5f*T_g.z)) - startVals_f.y) / T_f.y;
             slopeConstant = (sin_phi-cos_phi*(u+0.5f*T_g.z)) / (T_f.y*(cos_phi+sin_phi*(u+0.5f*T_g.z)) );
         }
         for (int i = 0; i < N_f.x; i++)
         {
             const float x = (float)i * T_f.x + startVals_f.x;
             const int j = (int)ceilf( x * slopeConstant + shiftConstant);
             const float y = (float)j * T_f.y + startVals_f.y;

            if (x*x + y*y > rFOVsq)
                continue;

             const float R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
             const int k = (int)ceilf(  z_ind_slope*R_minus_x_dot_theta + z_ind_offset  );

             if (k <= -3)
             {
                 if (z_ind_slope*cos_phi > 0.0f)
                     break;
                 else
                     continue;
             }
             if (k >= N_f.z)
             {
                 if (z_ind_slope*cos_phi < 0.0f)
                     break;
                 else
                     continue;
             }

             const float num_low = tau - x*sin_phi + y*cos_phi - A_y;
             const float num_high = num_low + 2.0f*A_y;

             const float denom_low = (R_minus_x_dot_theta - B_y) * T_g.z;
             const float denom_high = (R_minus_x_dot_theta + B_y) * T_g.z;

             const float hWeight_0 = max(0.0f, min( num_high/denom_high, n_plus_half ) - max( num_low/denom_low, n_minus_half ) );
             const float hWeight_1 = max(0.0f, min( (num_high+Ty_cos)/(denom_high-Ty_sin), n_plus_half ) - max( (num_low+Ty_cos)/(denom_low-Ty_sin), n_minus_half ) );
             const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

             const float v_phi_x_step = T_f.z / (T_g.y*R_minus_x_dot_theta);
             const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

             const float vWeight_0 = (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f);
             const float vWeight_1 = max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f);
             const float vWeight_2 = max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             const float y_12 = float(j) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1);
             const float z_12 = float(k) + 0.5f + vWeight_1 / (vWeight_0 + vWeight_1);
             if (volumeDimensionOrder == 0)
             {
                 g_output += (TEX3D(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                     + TEX3D(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                     + (TEX3D(f, z_12, float(j + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                         + TEX3D(f, float(k + 2) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
             }
             else
             {
                if (hWeight_2 == 0.0f)
                {
                    g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1);
                }
                else
                {
                    g_output += (TEX3D(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                        + (TEX3D(f, float(i) + 0.5f, float(j + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + TEX3D(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
                }
             }
         }
         if (accum)
             g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabsf(u * sin_phi + cos_phi) * g_output;
         else
             g[uint64(l) * uint64(N_g.z) * uint64(g_rowStride) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f+u*u) / fabsf(u*sin_phi+cos_phi) * g_output;
     }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool project_SF(float*& g, float* f, parameters* params, bool data_on_cpu)
{
    return project_SF(g, f, params, data_on_cpu, data_on_cpu);
}

bool backproject_SF(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_SF(g, f, params, data_on_cpu, data_on_cpu);
}

bool project_SF(float *&g, float *f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
{
    if (params->voxelSizeWorksForFastSF(1) == false)
    {
        //printf("using extended\n");
        if (params->voxelWidth < params->default_voxelWidth() || params->voxelHeight < params->default_voxelHeight())
            return project_eSF(g, f, params, data_on_cpu, volume_on_cpu, accum);
    }
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    bool doNormalize = true;
    if (fabs(params->pitchAngle) > 1.0e-6)
        doNormalize = false;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, doNormalize);
    
    float rFOVsq = params->rFOV()*params->rFOV();
    
    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

    // For !data_on_cpu, caller may pass a row-strided GPU buffer; pre-offset and pass
    // the true row stride. data_on_cpu uses contiguous temp dev_g (stride = numRows).
    float* dev_g_out = dev_g;
    int g_rowStride = params->numRows;
    if (!data_on_cpu)
    {
        g_rowStride = params->projectionDataStride > 0 ? params->projectionDataStride : params->numRows;
        dev_g_out = dev_g + (size_t)params->projectionDataFirstRow * params->numCols;
    }

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = NULL;
    /*
    if (data_on_cpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;
    d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));
    //*/
    //*
    if (volume_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, f, N_f, false, true, bool(params->volumeDimensionOrder == 1));
    else
        d_data_array = loadTexture(d_data_txt, f, N_f, false, true, bool(params->volumeDimensionOrder == 1));
    //*/

    if (d_data_array == nullptr)
    {
        fprintf(stderr, "failed to load texture!\n");
        if (data_on_cpu && dev_g != 0)
            cudaFree(dev_g);
        if (volume_on_cpu && dev_f != 0)
            cudaFree(dev_f);
        freeTexture(d_data_array, d_data_txt);
        cudaFree(dev_phis);

        return false;
    }

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->geometry == parameters::CONE)
    {
        if (params->detectorType == parameters::FLAT)
        {
            if (doNormalize)
                coneBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
            else
                coneBeamProjectorKernel_pitch_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, params->pitchAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
        }
        else
        {
            curvedConeBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
        }
        if (doNormalize)
            applyInversePolarWeight <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, g_rowStride);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        coneParallelProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g_out, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum, g_rowStride);
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
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

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

bool backproject_SF(float *g, float *&f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;
    if (params->voxelSizeWorksForFastSF(-1) == false)
    {
        //printf("using extended\n");
        if (params->voxelWidth > params->default_voxelWidth() || params->voxelHeight > params->default_voxelHeight())
            return backproject_eSF(g, f, params, data_on_cpu, volume_on_cpu, accum);
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_g; float4 T_g; float4 startVal_g;
    bool doNormalize = true;
    //if (fabs(params->pitchAngle) > 1.0e-6)
    //    doNormalize = false;
    //if (params->doWeightedBackprojection && params->helicalPitch != 0.0)
    //    doNormalize = true;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, doNormalize);
    
    bool doExtrapolation = params->doExtrapolation;

    float rFOVsq = params->rFOV()*params->rFOV();
    if (params->doDBP)
    {
        rFOVsq = params->furthestFromCenter() + params->voxelWidth;
        rFOVsq *= rFOVsq;
        doExtrapolation = false;
    }
    //printf("rFOV = %f, numCols = %d, u[0] = %f, u[-1] = %f\n", params->rFOV(), params->numCols, params->u(0), params->u(params->numCols - 1));

    bool rotated_detector = setConstantMemoryGeometryParameters_SF(params);

    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = NULL;
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, params, doExtrapolation, true);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, doExtrapolation, true);

    if (d_data_array == nullptr)
    {
        // Texture load failed (bounds error or true OOM even after CUDA pool reclaim); do not launch kernels with an invalid texture.
        cudaFree(dev_phis);
        if (data_on_cpu && dev_g != 0)
            cudaFree(dev_g);
        return false;
    }

    if (volume_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
        }
    }
    else
        dev_f = f;

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z)/float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);
    if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
    }
    else if (params->geometry == parameters::CONE)
    {
        if (params->doWeightedBackprojection && params->helicalPitch != 0.0)
        {
            if (params->detectorType == parameters::FLAT)
            {
                if (params->tiltAngle == 0.0 && params->pitchAngle == 0.0)
                    coneBeamHelicalWeightedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
                else
                {
                    //coneBeamHelicalWeightedBackprojectorKernel_rot_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, params->pitchAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
                    coneBeamBackprojectorKernel_rot_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, true, accum);
                }
            }
            else
                curvedConeBeamHelicalWeightedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
        else
        {
            if (params->detectorType == parameters::FLAT)
            {
                if (rotated_detector || params->doDBP)
                    coneBeamBackprojectorKernel_rot_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, false, accum);
                else
                    coneBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
            }
            else
                curvedConeBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        if (params->doWeightedBackprojection == true && params->helicalPitch != 0.0)
        {
            coneParallelWeightedHelicalBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
        else
            coneParallelBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
    }
    else
        return false;

    // pull result off GPU
    bool retVal = true;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }
    if (volume_on_cpu)
    {
        // on failure the volume was never copied back to f; returning true here
        // would let the caller consume stale/zeroed CPU data
        if (!pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU))
            retVal = false;
    }
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

    return retVal;
}

bool setConstantMemoryGeometryParameters_SF(parameters* params)
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
    if (sin_pitch == 0.0 && sin_tilt == 0.0 && zFOV.x == zFOV.y)
        return false;
    else
        return true;
    //*/
    //return true;
}
