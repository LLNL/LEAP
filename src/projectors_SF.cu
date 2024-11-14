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
__constant__ float d_v_min_inv;
__constant__ float d_v_max_inv;
__constant__ float d_weightFcnTransition;
__constant__ float d_weightFcnParameter;
__constant__ float d_phi_start;
__constant__ float d_phi_end;

__device__ float helicalConeWeight(float v)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void coneParallelWeightedHelicalBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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
        const float C = C_num * max(fabs(cos_phi), fabs(sin_phi));

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

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
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

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_arg_conj + iturn * v_arg_shift_conj);
            //sumWeights = 0.0f;

            const float v_phi_x_step = Tz_over_Tv * v_denom_inv;
            const float bpWeight = v_denom * rsqrtf(R * R + v_arg * v_arg) * centralWeight / (centralWeight + sumWeights);

            const float row_high = floor(v_phi_x - 0.5f * v_phi_x_step + 0.5f) + 0.5f;
            const float z_high = v_phi_x + 0.5f * v_phi_x_step - row_high;

            const float v_weight_one = min(v_phi_x_step, v_phi_x_step - z_high);
            const float v_weight_two = max(0.0f, min(z_high, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two = row_high + 2.0f;

            if (z_high > 1.0f)
            {
                val += (tex3D<float>(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two)
                    + tex3D<float>(g, s_ind_A, row_high_plus_two, L) * (z_high - 1.0f)) * bpWeight;
            }
            else
            {
                val += tex3D<float>(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two) * bpWeight;
            }
        }
    }
    if (accum)
        f[ind] += val * maxWeight;
    else
        f[ind] = val * maxWeight;
}

__global__ void coneParallelBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool doWeight, bool accum)
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
        const float C = C_num * max(fabs(cos_phi), fabs(sin_phi));

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

            const float row_high = floor(v_phi_x - 0.5f * v_phi_x_step + 0.5f) + 0.5f;
            const float z_high = v_phi_x + 0.5f * v_phi_x_step - row_high;

            const float v_weight_one = min(v_phi_x_step, v_phi_x_step - z_high);
            const float v_weight_two = max(0.0f, min(z_high, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two = row_high + 2.0f;

            if (z_high > 1.0f)
            {
                vals[k_offset] += (tex3D<float>(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two)
                    + tex3D<float>(g, s_ind_A, row_high_plus_two, L) * (z_high - 1.0f)) * bpWeight;
            }
            else
            {
                vals[k_offset] += tex3D<float>(g, s_ind_A, row_high + v_oneAndTwo, L) * (v_weight_one + v_weight_two) * bpWeight;
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

__global__ void coneParallelProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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
    const float l_phi = T_f.x / max(fabs(cos_phi), fabs(sin_phi));
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
    if (fabs(cos_phi) > fabs(sin_phi))
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
            const int j_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_dj_inv);
            const float s_ind_A = s_ind_base + (float)j_min_A * ds_ind_dj;

            const float x = float(i) * T_f.x + startVals_f.x;
            const float y = float(j_min_A) * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float x_dot_theta = cos_phi * x + sin_phi * y;
            const float v_denom = sqrtf(R * R - u * u) - x_dot_theta;

            const int k = (int)ceil(z_ind_slope * v_denom + z_ind_offset);

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
                g_output += (tex3D<float>(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(j_min_A + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(k + 2) + 0.5f, float(j_min_A + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
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
            const int i_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_di_inv);
            const float s_ind_A = s_ind_base + (float)i_min_A * ds_ind_di;

            const float x = float(i_min_A) * T_f.x + startVals_f.x;
            const float y = float(j) * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float x_dot_theta = cos_phi * x + sin_phi * y;
            const float v_denom = sqrtf(R * R - u * u) - x_dot_theta;

            const int k = (int)ceil(z_ind_slope * v_denom + z_ind_offset);

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
                g_output += (tex3D<float>(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(j) + 0.5f, float(i_min_A + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i_min_A + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, x_12, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
    }
    if (accum)
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += l_phi * g_output;
    else
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = l_phi * g_output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void parallelBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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
        const float C = C_num * max(fabs(cos_phi), fabs(sin_phi));
        float s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
        const float ds = modf(s_arg, &s_arg);
        const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
            vals[k_offset] += tex3D<float>(g, s_ind_A, float(k+k_offset) + 0.5f, float(l) + 0.5f);
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
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l1]);
			cos_phi = cos(phis[l1]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_B = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l2]);
			cos_phi = cos(phis[l2]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_C = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l3]);
			cos_phi = cos(phis[l3]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_D = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            // Do texture mapping
            val += tex3D<float>(g,s_ind_A, float(k)+0.5f, float(l)+0.5f)
                +  tex3D<float>(g,s_ind_B, float(k)+0.5f, float(l1)+0.5f)
                +  tex3D<float>(g,s_ind_C, float(k)+0.5f, float(l2)+0.5f)
                +  tex3D<float>(g,s_ind_D, float(k)+0.5f, float(l3)+0.5f);
            l += 4;
        }
        else if (l+1 < N_g.x)
        {
            int l1 = l + 1;

			sin_phi = sin(phis[l]);
			cos_phi = cos(phis[l]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

			sin_phi = sin(phis[l1]);
			cos_phi = cos(phis[l1]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_B = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            val += tex3D<float>(g,s_ind_A, float(k)+0.5f, float(l)+0.5f)
                +  tex3D<float>(g,s_ind_B, float(k)+0.5f, float(l1)+0.5f);
            l += 2;
        }
        else //if (l+1 < N_g.x)
        {
			sin_phi = sin(phis[l]);
			cos_phi = cos(phis[l]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg,&s_arg);
            const float s_ind_A = s_arg-(C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds) )))*maxWeight_inv+1.5f;

            val += tex3D<float>(g,s_ind_A, float(k)+0.5f, float(l)+0.5f);
            l += 1;
        }
    }

    f[ind] = val * maxWeight;
    //*/
}

__global__ void parallelBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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
    const float l_phi = T_f.x / max(fabs(cos_phi), fabs(sin_phi));
    const float C = T_f.x * T_f.x / (2.0f * T_g.z * l_phi);

    const float ds_ind_di = -T_f.x*sin_phi / T_g.z;
    const float ds_ind_dj = T_f.y*cos_phi / T_g.z;
    const float s_ind_offset = (startVals_f.y*cos_phi - startVals_f.x*sin_phi - startVals_g.z) / T_g.z;
    // s_ind(i,j) = (float)i * ds_ind_di + (float)j * ds_ind_dj + s_ind_offset

    float g_output = 0.0f;
    if (fabs(cos_phi) > fabs(sin_phi))
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
            const int j_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_dj_inv);
            const float s_ind_A = s_ind_base + (float)j_min_A * ds_ind_dj;

            if (((float)i*T_f.x+startVals_f.x )*((float)i*T_f.x+startVals_f.x) + ((float)j_min_A*T_f.y+startVals_f.y )*((float)j_min_A*T_f.y+startVals_f.y) > rFOVsq)
                continue;

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += (weight_0 + weight_1) * tex3D<float>(f, float(m)+0.5f, float(j_min_A)+0.5f+ weight_1/(weight_0+ weight_1), float(i)+0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, float(m)+0.5f, float(j_min_A + 2) + 0.5f, float(i) + 0.5f);
            }
            else
            {
                g_output += (weight_0 + weight_1) * tex3D<float>(f, float(i) + 0.5f, float(j_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, float(m)+0.5f);
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
            const int i_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_di_inv);
            const float s_ind_A = s_ind_base + (float)i_min_A * ds_ind_di;

            if (((float)i_min_A*T_f.x+startVals_f.x )*((float)i_min_A*T_f.x+startVals_f.x) + ((float)j*T_f.y+startVals_f.y )*((float)j*T_f.y+startVals_f.y) > rFOVsq)
                continue;

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += (weight_0 + weight_1) * tex3D<float>(f, float(m)+0.5f, float(j)+0.5f, float(i_min_A)+0.5f + weight_1/(weight_0 + weight_1))
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, float(m) + 0.5f, float(j) + 0.5f, float(i_min_A + 2)+0.5f);
            }
            else
            {
                g_output += (weight_0 + weight_1) * tex3D<float>(f, float(i_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(j) + 0.5f, float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, float(m) + 0.5f);
            }
        }
    }
    if (accum)
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += l_phi * g_output;
    else
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = l_phi * g_output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fanBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool doWeight, bool accum)
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
        const float x_denom = fabs(u_arg * cos_phi - sin_phi);
        const float y_denom = fabs(u_arg * sin_phi + cos_phi);
        const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabs(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabs(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
        const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

        float ind_first = floor(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float ind_last = ind_first + 2.5f;
        ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;

        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            vals[k_offset] += (tex3D<float>(g, ind_first, iv + float(k_offset), L) * horizontalWeights_0_A
                + tex3D<float>(g, ind_last, iv + float(k_offset), L) * horizontalWeights_1_A) * bpWeight;
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
        const float x_denom = fabs(u_arg * cos_phi - sin_phi);
        const float y_denom = fabs(u_arg * sin_phi + cos_phi);
        const float l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabs(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabs(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
        const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

        float ind_first = floor(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float ind_last = ind_first + 2.5f;
        ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float bpWeight = doWeight ? R * R_minus_x_dot_theta_inv : 1.0f;

        val += (tex3D<float>(g, ind_first, iv, L) * horizontalWeights_0_A
            + tex3D<float>(g, ind_last, iv, L) * horizontalWeights_1_A) * bpWeight;
    }

    f[ind] = val;
    //*/
}

__global__ void fanBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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

    const int iz = int(floor(0.5 + (v - startVals_f.z) / T_f.z));

    float g_output = 0.0f;

    if (fabs(u * cos_phi - sin_phi) > fabs(u * sin_phi + cos_phi))
    {
        const float A_x = fabs(sin_phi) * 0.5f * T_f.x;
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
            const int i = (int)ceil(y * slopeConstant + shiftConstant);
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
                g_output += tex3D<float>(f, float(iz) + 0.5f, float(j) + 0.5f, float(i) + 0.5f + hWeight_1/(hWeight_0 + hWeight_1)) * (hWeight_0 + hWeight_1)
                    + tex3D<float>(f, float(iz) + 0.5f, float(j) + 0.5f, float(i + 2)+0.5f) * hWeight_2;
            }
            else
            {
                g_output += tex3D<float>(f, float(i) + 0.5f + hWeight_1 / (hWeight_0 + hWeight_1), float(j) + 0.5f, float(iz) + 0.5f) * (hWeight_0 + hWeight_1)
                    + tex3D<float>(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(iz) + 0.5f) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float A_y = fabs(cos_phi) * 0.5f * T_f.x;
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
            const int j = (int)ceil(x * slopeConstant + shiftConstant);
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
                g_output += tex3D<float>(f, float(iz) + 0.5f, float(j) + 0.5f + hWeight_1/(hWeight_0 + hWeight_1), float(i) + 0.5f) * (hWeight_0 + hWeight_1)
                    + tex3D<float>(f, float(iz) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * hWeight_2;
            }
            else
            {
                g_output += tex3D<float>(f, float(i) + 0.5f, float(j) + 0.5f+hWeight_1 / (hWeight_0 + hWeight_1), float(iz) + 0.5f) * (hWeight_0 + hWeight_1)
                    + tex3D<float>(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(iz)+0.5f) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
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

__global__ void applyInversePolarWeight(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[uint64(i) * uint64(N_g.z * N_g.y) + uint64(j * N_g.z + k)] *= sqrtf(1.0f + v * v);
}

__global__ void curvedConeBeamHelicalWeightedBackprojectorKernel_SF(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

        const float dist_from_source_components_x = fabs(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabs(R * sin_phi - tau * cos_phi - y);
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
            const float x_denom = fabs(u_arg * cos_phi - sin_phi);
            const float y_denom = fabs(u_arg * sin_phi + cos_phi);
            float A_x;
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = (atan((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x)) - startVals_g.z) * Tu_inv;
            const float tau_high = (atan((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x)) - startVals_g.z) * Tu_inv;

            float ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            const float v_phi_x_step_A = Tz_over_Tv * dist_from_source_inv;
            const float v_phi_x_first = (v_phi_x_start_num - z_source_over_T_v) * dist_from_source_inv - v0_over_Tv;
            
            const float v_phi_x = v_phi_x_first;

            const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            // Calculate the View Redundancy Weight
            const float phi_cur_conj = phi_cur - 2.0f * atan(u_arg) + atan_term + PI;
            const float cos_phi_conj = cos(phi_cur_conj);
            const float sin_phi_conj = sin(phi_cur_conj);
            const float dist_from_source_components_x_conj = fabs(R * cos_phi_conj + tau * sin_phi_conj - x);
            const float dist_from_source_components_y_conj = fabs(R * sin_phi_conj - tau * cos_phi_conj - y);
            const float dist_from_source_conj = sqrtf(dist_from_source_components_x_conj * dist_from_source_components_x_conj + dist_from_source_components_y_conj * dist_from_source_components_y_conj);
            const float dist_from_source_inv_conj = 1.0f / dist_from_source_conj;

            float sumWeights = 0.0f;

            const float v_arg_shift = neg_twoPI_pitch * dist_from_source_inv;

            const float v_bound_A = (v_arg_shift > 0.0f) ? (v_min - v_arg) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_arg) * dist_from_source * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_arg_shift < 0.0f) ? (v_min - v_arg) * dist_from_source * neg_twoPI_pitch_inv : (v_max - v_arg) * dist_from_source * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight(v_arg + iturn * v_arg_shift);
            }

            const float v_arg_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * dist_from_source_inv_conj;
            const float v_arg_shift_conj = neg_twoPI_pitch * dist_from_source_inv_conj;

            const float v_bound_A_conj = (v_arg_shift_conj > 0.0f) ? (v_min - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_arg_shift_conj < 0.0f) ? (v_min - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * dist_from_source_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_arg_conj + iturn * v_arg_shift_conj);

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + (z_high_A - 1.0f)));

                val += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (tex3D<float>(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + tex3D<float>(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * extraWeight * centralWeight / (centralWeight + sumWeights);
            }
            else
            {
                const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * dist_from_source_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two));

                val += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * extraWeight * centralWeight / (centralWeight + sumWeights);
            }
        }
    }
    //f[ind] = val;
    if (accum)
        f[ind] += val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
    else
        f[ind] = val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
}

__global__ void coneBeamHelicalWeightedBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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

    const float atan_term = atan(2.0f * tau * R / (R * R - tau * tau));
    
    const float twoPI_inv = 1.0f / (2.0f * PI);
    const float neg_twoPI_pitch = -2.0f * PI * T_g.w;
    const float neg_twoPI_pitch_inv = 1.0f / neg_twoPI_pitch;

    for (int l = 0; l < N_g.x; l++)
    {
        const float phi_cur = phis[l];
        const float z_source = (phi_cur * T_g.w + startVals_g.w);
        const float sin_phi = sin(phi_cur);
        const float cos_phi = cos(phi_cur);

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
            const float x_denom = fabs(u_arg * cos_phi - sin_phi);
            const float y_denom = fabs(u_arg * sin_phi + cos_phi);
            const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            float A_x;
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            float ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            const float v_phi_x = (v_phi_x_start_num - z_source * Tv_inv) * R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            // Calculate the View Redundancy Weight
            const float phi_cur_conj = phi_cur - 2.0f * atan(u_arg) + atan_term + PI;
            const float R_minus_x_dot_theta_conj = R - x * cos(phi_cur_conj) - y * sin(phi_cur_conj);
            const float R_minus_x_dot_theta_inv_conj = 1.0f / R_minus_x_dot_theta_conj;
            float sumWeights = 0.0f;

            const float v_arg_shift = neg_twoPI_pitch * R_minus_x_dot_theta_inv;

            const float v_bound_A = (v_arg_shift > 0.0f) ? (v_min - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;
            const float v_bound_B = (v_arg_shift < 0.0f) ? (v_min - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv : (v_max - v_arg) * R_minus_x_dot_theta * neg_twoPI_pitch_inv;

            const int N_turns_below = max(int(ceil((d_phi_start - phi_cur) * twoPI_inv)), int(ceil(v_bound_A)));
            const int N_turns_above = min(int(floor((d_phi_end - phi_cur) * twoPI_inv)), int(floor(v_bound_B)));
            for (int iturn = N_turns_below; iturn <= N_turns_above; iturn++)
            {
                if (iturn != 0)
                    sumWeights += helicalConeWeight(v_arg + iturn * v_arg_shift);
            }

            const float v_arg_conj = (z - (phi_cur_conj * T_g.w + startVals_g.w)) * R_minus_x_dot_theta_inv_conj;
            const float v_arg_shift_conj = neg_twoPI_pitch * R_minus_x_dot_theta_inv_conj;

            const float v_bound_A_conj = (v_arg_shift_conj > 0.0f) ? (v_min - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;
            const float v_bound_B_conj = (v_arg_shift_conj < 0.0f) ? (v_min - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv : (v_max - v_arg_conj) * R_minus_x_dot_theta_conj * neg_twoPI_pitch_inv;

            const int N_turns_below_conj = max(int(ceil((d_phi_start - phi_cur_conj) * twoPI_inv)), int(ceil(v_bound_A_conj)));
            const int N_turns_above_conj = min(int(floor((d_phi_end - phi_cur_conj) * twoPI_inv)), int(floor(v_bound_B_conj)));
            for (int iturn = N_turns_below_conj; iturn <= N_turns_above_conj; iturn++)
                sumWeights += helicalConeWeight(v_arg_conj + iturn * v_arg_shift_conj);

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            const float extraWeight = sqrtf(1.0f + v_arg*v_arg) * R * R_minus_x_dot_theta_inv / (l_phi * (tau_high - tau_low) * (v_weight_one + v_weight_two + max(0.0f, z_high_A - 1.0f)));

            //*
            val += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                + (tex3D<float>(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * max(0.0f, z_high_A - 1.0f)) * extraWeight * centralWeight / (centralWeight + sumWeights);
            //*/
            //val += centralWeight + sumWeights;
            //val += 1.0f + sumWeights;
            //val += centralWeight / (centralWeight + sumWeights) * R_minus_x_dot_theta;
        }
    }

    //f[ind] = val;
    if (accum)
        f[ind] += val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
    else
        f[ind] = val * (T_f.x * T_f.y * T_f.z) / (R * R * T_g.y * T_g.z);
}

__global__ void curvedConeBeamBackprojectorKernel_SF(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

        const float dist_from_source_components_x = fabs(R * cos_phi + tau * sin_phi - x);
        const float dist_from_source_components_y = fabs(R * sin_phi - tau * cos_phi - y);
        const float dist_from_source = sqrtf(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);
        const float l_phi = T_f.x * dist_from_source / max(dist_from_source_components_x, dist_from_source_components_y);

        const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabs(u_arg * cos_phi - sin_phi);
        const float y_denom = fabs(u_arg * sin_phi + cos_phi);
        //const float l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
        float A_x;
        if (x_denom > y_denom)
            A_x = fabs(sin_phi) * T_x_over_2;
        else
        {
            A_x = fabs(cos_phi) * T_x_over_2;
            B_x = B_y;
        }
        const float tau_low = (atan((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x)) - startVals_g.z) * Tu_inv;
        const float tau_high = (atan((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x)) - startVals_g.z) * Tu_inv;

        float ind_first = floor(tau_low + 0.5f); // first detector index

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

            const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                vals[k_offset] += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (tex3D<float>(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + tex3D<float>(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * v_weight;
            }
            else
            {
                vals[k_offset] += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * v_weight;
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

__global__ void curvedConeBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder, bool accum)
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

    if (fabs(u * cos_phi - sin_phi) > fabs(u * sin_phi + cos_phi))
    {
        const float A_x = fabs(sin_phi) * 0.5f * T_f.x;
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
            const int i = (int)ceil(y * slopeConstant + shiftConstant);
            const float x = (float)i * T_f.x + startVals_f.x;

            if (x * x + y * y > rFOVsq)
                continue;

            const float v_denom_inv = rsqrtf((sourcePos_x - x) * (sourcePos_x - x) + (sourcePos_y - y) * (sourcePos_y - y));

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            //const int k = (int)ceil(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);
            const int k = (int)ceil(z_ind_slope / v_denom_inv + z_ind_offset);

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
                g_output += (tex3D<float>(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(j) + 0.5f, float(i + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, x_12, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(i + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float A_y = fabs(cos_phi) * 0.5f * T_f.x;
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
            const int j = (int)ceil(x * slopeConstant + shiftConstant);
            const float y = (float)j * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float v_denom_inv = rsqrtf((sourcePos_x - x) * (sourcePos_x - x) + (sourcePos_y - y) * (sourcePos_y - y));

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            //const int k = (int)ceil(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);
            const int k = (int)ceil(z_ind_slope / v_denom_inv + z_ind_offset);

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
                g_output += (tex3D<float>(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(j + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(k + 2) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(i) + 0.5f, float(j + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
    }
}

__global__ void coneBeamBackprojectorKernel_SF(cudaTextureObject_t g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            const float u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            const float x_denom = fabs(u_arg * cos_phi - sin_phi);
            const float y_denom = fabs(u_arg * sin_phi + cos_phi);
            const float l_phi = T_f.x * sqrtf(1.0f + u_arg * u_arg) / max(x_denom, y_denom);

            float A_x;
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            const float tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            const float tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            float ind_first = floor(tau_low + 0.5f); // first detector index

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
                const float v_weight = sqrtf(1.0f + v_arg * v_arg);

                const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                if (z_high_A > 1.0f)
                {
                    vals[k_offset] += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (tex3D<float>(g, ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + tex3D<float>(g, ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f)) * v_weight;
                }
                else
                {
                    vals[k_offset] += ((tex3D<float>(g, ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)) * v_weight;
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

            float B_x = (sin_phi < 0.0f) ? -cos_phi * T_x_over_2 : cos_phi * T_x_over_2;
            const float B_y = (cos_phi < 0.0f) ? sin_phi * T_x_over_2 : -sin_phi * T_x_over_2;

            const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = (k+k_offset) * T_f.z + startVals_f.z;

                const float u_num = x_dot_theta_perp * cos_tilt + z * sin_tilt;
                const float v_num = z * cos_tilt - x_dot_theta_perp * sin_tilt - z_source;

                const float u_arg = u_num * R_minus_x_dot_theta_inv;
                const float v_arg = v_num * R_minus_x_dot_theta_inv;
                const float x_denom = fabs(u_arg * cos_phi - sin_phi);
                const float y_denom = fabs(u_arg * sin_phi + cos_phi);
                const float l_phi = T_f.x * sqrtf((1.0f + u_arg * u_arg) * (1.0f + v_arg * v_arg)) / max(x_denom, y_denom);

                // Calculate footprint along columns
                float A_x;
                if (x_denom > y_denom)
                    A_x = fabs(sin_phi) * T_x_over_2;
                else
                {
                    A_x = fabs(cos_phi) * T_x_over_2;
                    B_x = B_y;
                }
                const float tau_low = ((u_num - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
                const float tau_high = ((u_num + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

                float u_ind_first = floor(tau_low + 0.5f); // first detector index

                const float uWeights_0 = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float uWeights_1 = l_phi * (tau_high - tau_low) - uWeights_0;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / uWeights_0;

                // Calculate footprint along rows
                const float v_A = (v_arg - voxz_half * R_minus_x_dot_theta_inv - startVals_g.y) * Tv_inv;
                const float v_B = (v_arg + voxz_half * R_minus_x_dot_theta_inv - startVals_g.y) * Tv_inv;

                float v_ind_first = floor(v_A + 0.5f); // first detector index

                const float vWeights_0 = (min(v_B, v_ind_first + 1.5f) - v_A);
                const float vWeights_1 = (v_B - v_A) - vWeights_0;

                const float v_ind_last = v_ind_first + 2.5f;
                v_ind_first = v_ind_first + 0.5f + max(0.0f, min(v_B - v_ind_first - 0.5f, 1.0f)) / vWeights_0;

                if (vWeights_1 > 0.0f)
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, v_ind_first, L) * uWeights_0
                        + tex3D<float>(g, u_ind_last, v_ind_first, L) * uWeights_1) * vWeights_0
                        + (tex3D<float>(g, u_ind_first, v_ind_last, L) * uWeights_0
                            + tex3D<float>(g, u_ind_last, v_ind_last, L) * uWeights_1) * vWeights_1;
                }
                else
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, v_ind_first, L) * uWeights_0
                        + tex3D<float>(g, u_ind_last, v_ind_first, L) * uWeights_1) * vWeights_0;
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

__global__ void coneBeamProjectorKernel_SF(float* g, const int4 N_g, const float4 T_g, const float4 startVals_g, cudaTextureObject_t f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float rFOVsq, const float* phis, const int volumeDimensionOrder, bool accum)
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
    
     if (fabs(u*cos_phi-sin_phi) > fabs(u*sin_phi+cos_phi))
     {
         const float A_x = fabs(sin_phi) * 0.5f*T_f.x;
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
             const int i = (int)ceil(y * slopeConstant +  shiftConstant);
             const float x = (float)i * T_f.x + startVals_f.x;

            if (x*x + y*y > rFOVsq)
                continue;

             const float R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
             const int k = (int)ceil(  z_ind_slope*R_minus_x_dot_theta + z_ind_offset  );

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
                 g_output += (tex3D<float>(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                     + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                     + (tex3D<float>(f, z_12, float(j) + 0.5f, float(i + 2) + 0.5f) * (vWeight_0 + vWeight_1)
                         + tex3D<float>(f, float(k + 2) + 0.5f, float(j) + 0.5f, float(i + 2) + 0.5f) * vWeight_2) * hWeight_2;
             }
             else
             {
                 g_output += (tex3D<float>(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                 + tex3D<float>(f, x_12, float(j) + 0.5f, float(k+2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                 + (tex3D<float>(f, float(i + 2) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                 + tex3D<float>(f, float(i + 2) + 0.5f, float(j) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
             }
         }
         if (accum)
             g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
         else
             g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f+u*u) / fabs(u*cos_phi-sin_phi) * g_output;
     }
     else
     {
         const float A_y = fabs(cos_phi) * 0.5f*T_f.x;
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
             const int j = (int)ceil( x * slopeConstant + shiftConstant);
             const float y = (float)j * T_f.y + startVals_f.y;

            if (x*x + y*y > rFOVsq)
                continue;

             const float R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
             const int k = (int)ceil(  z_ind_slope*R_minus_x_dot_theta + z_ind_offset  );

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
                 g_output += (tex3D<float>(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                     + tex3D<float>(f, float(k + 2) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                     + (tex3D<float>(f, z_12, float(j + 2) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                         + tex3D<float>(f, float(k + 2) + 0.5f, float(j + 2) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
             }
             else
             {
                 g_output += (tex3D<float>(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                     + tex3D<float>(f, float(i) + 0.5f, y_12, float(k + 2) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                     + (tex3D<float>(f, float(i) + 0.5f, float(j + 2) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                     + tex3D<float>(f, float(i) + 0.5f, float(j + 2) + 0.5f, float(k + 2) + 0.5f) * vWeight_2) * hWeight_2;
             }
         }
         if (accum)
             g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
         else
             g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf(1.0f+u*u) / fabs(u*sin_phi+cos_phi) * g_output;
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
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);
    
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

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = NULL;
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

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->geometry == parameters::CONE)
    {
        if (params->detectorType == parameters::FLAT)
            coneBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        else
        {
            curvedConeBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
        applyInversePolarWeight <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        coneParallelProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
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
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);
    
    float rFOVsq = params->rFOV()*params->rFOV();
    //printf("rFOV = %f, numCols = %d, u[0] = %f, u[-1] = %f\n", params->rFOV(), params->numCols, params->u(0), params->u(params->numCols - 1));

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = NULL;
    /*
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    if (params->geometry == parameters::CONE)
    {
        applyInversePolarWeight <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
    }

    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, true);
    //*/
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, N_g, params->doExtrapolation, true);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, true);

    if (volume_on_cpu)
    {
        //if (dev_g != 0) // not needed anymore
        //    cudaFree(dev_g);
        //dev_g = 0;
        //printf("mallocing %.0f elements\n", float(params->volumeData_numberOfElements()));
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
            float q_helical = float(params->helicalFBPWeight);
            //float q_helical = float(0.99);
            float weightFcnParameter = float(-2.0 / ((1.0 - q_helical) * (1.0 - q_helical)));
            float weightFcnTransition = float((q_helical + 1.0) / 2.0);
            float v_min_inv = float((params->v(0) - 0.5 * params->pixelHeight) / params->sdd);
            v_min_inv = float(1.0 / v_min_inv);
            float v_max_inv = float((params->v(params->numRows - 1) + 0.5 * params->pixelHeight) / params->sdd);
            v_max_inv = float(1.0 / v_max_inv);
            float phi_start = params->get_phi_start();
            float phi_end = params->get_phi_end();

            /*
            float* v_weights = new float[params->numRows];
            for (int i = 0; i < params->numRows; i++)
            {
                float v = startVal_g.y + i * T_g.y;

                float abs_v_hat = (v >= 0.0f) ? v * v_max_inv : v * v_min_inv;

                if (abs_v_hat <= q_helical)
                    v_weights[i] = 1.0;
                else if (abs_v_hat > 1.0)
                    v_weights[i] = 0.0;
                else if (abs_v_hat <= weightFcnTransition)
                    v_weights[i] = weightFcnParameter * (abs_v_hat - q_helical) * (abs_v_hat - q_helical) + 1.0;
                else
                    v_weights[i] = -1.0 * weightFcnParameter * (abs_v_hat - 1.0) * (abs_v_hat - 1.0);
                //printf("cpu %f: %f\n", v, v_weights[i]);
            }
            cudaTextureObject_t d_v_weights_txt = NULL;
            cudaArray* d_v_weights_array = loadTexture1D(d_v_weights_txt, v_weights, params->numRows, false, true);
            //*/

            //printf("v_min/max = %f, %f\n", v_min, v_max);
            //printf("weight params: %f, %f\n", weightFcnParameter, weightFcnTransition);

            cudaMemcpyToSymbol(d_q_helical, &q_helical, sizeof(float));
            cudaMemcpyToSymbol(d_v_min_inv, &v_min_inv, sizeof(float));
            cudaMemcpyToSymbol(d_v_max_inv, &v_max_inv, sizeof(float));
            cudaMemcpyToSymbol(d_weightFcnTransition, &weightFcnTransition, sizeof(float));
            cudaMemcpyToSymbol(d_weightFcnParameter, &weightFcnParameter, sizeof(float));
            cudaMemcpyToSymbol(d_phi_start, &phi_start, sizeof(float));
            cudaMemcpyToSymbol(d_phi_end, &phi_end, sizeof(float));

            if (params->detectorType == parameters::FLAT)
                coneBeamHelicalWeightedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
            else
                curvedConeBeamHelicalWeightedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);

            //cudaFreeArray(d_v_weights_array);
            //cudaDestroyTextureObject(d_v_weights_txt);
            //delete[] v_weights;
        }
        else
        {
            if (params->detectorType == parameters::FLAT)
                coneBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
            else
                curvedConeBeamBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        if (params->doWeightedBackprojection == true && params->helicalPitch != 0.0)
        {
            float q_helical = float(params->helicalFBPWeight);
            //float q_helical = float(0.99);
            float weightFcnParameter = float(-2.0 / ((1.0 - q_helical) * (1.0 - q_helical)));
            float weightFcnTransition = float((q_helical + 1.0) / 2.0);
            float v_min_inv = float((params->v(0) - 0.5 * params->pixelHeight) / params->sdd);
            v_min_inv = float(1.0 / v_min_inv);
            float v_max_inv = float((params->v(params->numRows - 1) + 0.5 * params->pixelHeight) / params->sdd);
            v_max_inv = float(1.0 / v_max_inv);
            float phi_start = params->get_phi_start();
            float phi_end = params->get_phi_end();

            //printf("v_min/max = %f, %f\n", v_min, v_max);
            //printf("weight params: %f, %f\n", weightFcnParameter, weightFcnTransition);

            cudaMemcpyToSymbol(d_q_helical, &q_helical, sizeof(float));
            cudaMemcpyToSymbol(d_v_min_inv, &v_min_inv, sizeof(float));
            cudaMemcpyToSymbol(d_v_max_inv, &v_max_inv, sizeof(float));
            cudaMemcpyToSymbol(d_weightFcnTransition, &weightFcnTransition, sizeof(float));
            cudaMemcpyToSymbol(d_weightFcnParameter, &weightFcnParameter, sizeof(float));
            cudaMemcpyToSymbol(d_phi_start, &phi_start, sizeof(float));
            cudaMemcpyToSymbol(d_phi_end, &phi_end, sizeof(float));

            coneParallelWeightedHelicalBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, accum);
        }
        else
            coneParallelBackprojectorKernel_SF <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder, params->doWeightedBackprojection, accum);
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
    //*
    if (volume_on_cpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;
    //*/

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
    else if (params->geometry == parameters::CONE)
    {
        applyPolarWeight <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
        cudaStatus = cudaDeviceSynchronize();
    }
    //*/

    return true;
}
