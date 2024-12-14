////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for Joseph projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors.h"
#include "projectors_Joseph.cuh"
#include "cuda_utils.h"
//#include "ray_weighting.cuh"
//#include "ray_weighting_cpu.h"

//#define NUM_SLICES_PER_THREAD 1
#define NUM_SLICES_PER_THREAD 8

//*
__global__ void modularBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOVsq, const bool accum)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float3 moduleCenter = make_float3(moduleCenters[3 * l + 0], moduleCenters[3 * l + 1], moduleCenters[3 * l + 2]);
    const float3 p = make_float3(sourcePositions[3 * l + 0], sourcePositions[3 * l + 1], sourcePositions[3 * l + 2]);
    const float3 u_vec = make_float3(colVectors[3 * l + 0], colVectors[3 * l + 1], colVectors[3 * l + 2]);
    const float3 v_vec = make_float3(rowVectors[3 * l + 0], rowVectors[3 * l + 1], rowVectors[3 * l + 2]);

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
    if (fabs(r.y) > fabs(r.x))
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

            const float horizontal_footprint_half_width = 0.5f * T_f.x * t * fabs(u_vec_flat.x) * T_u_inv;
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
                g_output += (tex3D<float>(f, z_12, float(j) + 0.5f, x_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(iz + 1) + 0.5f, float(j) + 0.5f, x_12) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(j) + 0.5f, float(ix + 1) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(iz + 1) + 0.5f, float(j) + 0.5f, float(ix + 1) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, x_12, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, x_12, float(j) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(ix + 1) + 0.5f, float(j) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(ix + 1) + 0.5f, float(j) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf((r.y*r.y + r.x*r.x)*(r.y*r.y + r.z*r.z)) * fabs(r_y_inv*r_y_inv) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf((r.y*r.y + r.x*r.x)*(r.y*r.y + r.z*r.z)) * fabs(r_y_inv*r_y_inv) * g_output;
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

            const float horizontal_footprint_half_width = 0.5f * T_f.y * t * fabs(u_vec_flat.y) * T_u_inv;
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
                g_output += (tex3D<float>(f, z_12, y_12, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(iz + 1) + 0.5f, y_12, float(i) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, z_12, float(iy + 1) + 0.5f, float(i) + 0.5f) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(iz + 1) + 0.5f, float(iy + 1) + 0.5f, float(i) + 0.5f) * vWeight_2) * hWeight_2;
            }
            else
            {
                g_output += (tex3D<float>(f, float(i) + 0.5f, y_12, z_12) * (vWeight_0 + vWeight_1)
                    + tex3D<float>(f, float(i) + 0.5f, y_12, float(iz + 1) + 0.5f) * vWeight_2) * (hWeight_0 + hWeight_1)
                    + (tex3D<float>(f, float(i) + 0.5f, float(iy + 1) + 0.5f, z_12) * (vWeight_0 + vWeight_1)
                        + tex3D<float>(f, float(i) + 0.5f, float(iy + 1) + 0.5f, float(iz + 1) + 0.5f) * vWeight_2) * hWeight_2;
            }
        }
        if (accum)
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] += T_f.x * sqrtf((r.x*r.x + r.y*r.y)*(r.x*r.x + r.z*r.z)) * fabs(r_x_inv*r_x_inv) * g_output;
        else
            g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrtf((r.x*r.x + r.y*r.y)*(r.x*r.x + r.z*r.z)) * fabs(r_x_inv*r_x_inv) * g_output;
    }
}
//*/

/*
__global__ void modularBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOVsq)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    float* sourcePosition = &sourcePositions[3 * l];
    float* moduleCenter = &moduleCenters[3 * l];
    float* v_vec = &rowVectors[3 * l];
    float* u_vec = &colVectors[3 * l];

    const float t = m * T_g.y + startVals_g.y;
    const float s = n * T_g.z + startVals_g.z;

    const float3 s_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
    const float D = sqrtf(s_minus_c.x * s_minus_c.x + s_minus_c.y * s_minus_c.y);

    const float phi = atan2(s_minus_c.y, s_minus_c.x);
    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

    const float R = sourcePosition[0] * cos_phi + sourcePosition[1] * sin_phi;
    const float tau = sourcePosition[0] * sin_phi - sourcePosition[1] * cos_phi;

    const float u = (s * (-u_vec[0] * sin_phi + u_vec[1] * cos_phi) + t*(-v_vec[0]*sin_phi + v_vec[1]*cos_phi)) / D;
    const float v = (-s_minus_c.z + s * u_vec[2] + t * v_vec[2]) / D;
    //const float v = -t / D;

    const float n_minus_half = (float)n - 0.5f + startVals_g.z / T_g.z;
    const float n_plus_half = (float)n + 0.5f + startVals_g.z / T_g.z;
    const float m_minus_half = (float)m - 0.5f -s_minus_c.z / (T_g.y);
    const float m_plus_half = (float)m + 0.5f -s_minus_c.z / (T_g.y);

    const float T_v = T_g.y / D;
    const float T_u = T_g.z / D;

    const float v0_over_Tv = startVals_g.y / T_g.y;

    const float z_source = sourcePosition[2];

    const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
    const float z_ind_offset = -z0_over_Tz_plus_half + z_source / T_f.z;

    const float z_ind_slope = (v - 0.5f * T_v) / T_f.z;

    float g_output = 0.0f;

    if (fabs(u * cos_phi - sin_phi) > fabs(u * sin_phi + cos_phi))
    {
        const float A_x = fabs(sin_phi) * 0.5f * T_f.x;
        const float B_x = cos_phi * 0.5f * T_f.x * ((sin_phi < 0.0f) ? -1.0f : 1.0f);
        const float Tx_sin = T_f.x * sin_phi;
        const float Tx_cos = T_u * T_f.x * cos_phi;

        float shiftConstant, slopeConstant;
        if (u * cos_phi - sin_phi > 0.0f)
        {
            shiftConstant = (((R + B_x) * (u - 0.5f * T_u) - A_x - tau) / (cos_phi * (u - 0.5f * T_u) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = (-sin_phi * (u - 0.5f * T_u) - cos_phi) / (T_f.x * (cos_phi * (u - 0.5f * T_u) - sin_phi));
        }
        else
        {
            shiftConstant = (((R - B_x) * (u + 0.5f * T_u) + A_x - tau) / (cos_phi * (u + 0.5f * T_u) - sin_phi) - startVals_f.x) / T_f.x;
            slopeConstant = (sin_phi * (u + 0.5f * T_u) + cos_phi) / (T_f.x * (-cos_phi * (u + 0.5f * T_u) + sin_phi));
        }

        for (int j = 0; j < N_f.y; j++)
        {
            const float y = (float)j * T_f.y + startVals_f.y;
            const int i = (int)ceil(y * slopeConstant + shiftConstant);
            const float x = (float)i * T_f.x + startVals_f.x;

            if (x * x + y * y > rFOVsq)
                continue;

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const int k = (int)ceil(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);

            if (k <= -3)
            {
                if (z_ind_slope * sin_phi > 0.0f)
                    break;
                else
                    continue;
            }
            if (k >= N_f.z)
            {
                if (z_ind_slope * sin_phi < 0.0f)
                    break;
                else
                    continue;
            }

            const float num_low = tau - x * sin_phi + y * cos_phi - A_x;
            const float num_high = num_low + 2.0f * A_x;

            const float denom_low = (R_minus_x_dot_theta - B_x) * T_u;
            const float denom_high = (R_minus_x_dot_theta + B_x) * T_u;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high - Tx_sin) / (denom_high - Tx_cos), n_plus_half) - max((num_low - Tx_sin) / (denom_low - Tx_cos), n_minus_half));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float v_phi_x_step = T_f.z / (T_v * R_minus_x_dot_theta);
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
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrt(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float A_y = fabs(cos_phi) * 0.5f * T_f.x;
        const float B_y = sin_phi * 0.5f * T_f.x * ((cos_phi < 0.0f) ? 1.0f : -1.0f);
        const float Ty_cos = T_f.y * cos_phi;
        const float Ty_sin = T_u * T_f.y * sin_phi;

        float shiftConstant, slopeConstant;
        if (u * sin_phi + cos_phi >= 0.0f)
        {
            shiftConstant = (((R + B_y) * (u - 0.5f * T_u) - A_y - tau) / (sin_phi * (u - 0.5f * T_u) + cos_phi) - startVals_f.y) / T_f.y;
            slopeConstant = (sin_phi - cos_phi * (u - 0.5f * T_u)) / (T_f.y * (sin_phi * (u - 0.5f * T_u) + cos_phi));
        }
        else
        {
            shiftConstant = (((R - B_y) * (u + 0.5f * T_u) + A_y - tau) / (cos_phi + sin_phi * (u + 0.5f * T_u)) - startVals_f.y) / T_f.y;
            slopeConstant = (sin_phi - cos_phi * (u + 0.5f * T_u)) / (T_f.y * (cos_phi + sin_phi * (u + 0.5f * T_u)));
        }
        for (int i = 0; i < N_f.x; i++)
        {
            const float x = (float)i * T_f.x + startVals_f.x;
            const int j = (int)ceil(x * slopeConstant + shiftConstant);
            const float y = (float)j * T_f.y + startVals_f.y;

            if (x * x + y * y > rFOVsq)
                continue;

            const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            const int k = (int)ceil(z_ind_slope * R_minus_x_dot_theta + z_ind_offset);

            if (k <= -3)
            {
                if (z_ind_slope * cos_phi > 0.0f)
                    break;
                else
                    continue;
            }
            if (k >= N_f.z)
            {
                if (z_ind_slope * cos_phi < 0.0f)
                    break;
                else
                    continue;
            }

            const float num_low = tau - x * sin_phi + y * cos_phi - A_y;
            const float num_high = num_low + 2.0f * A_y;

            const float denom_low = (R_minus_x_dot_theta - B_y) * T_u;
            const float denom_high = (R_minus_x_dot_theta + B_y) * T_u;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high + Ty_cos) / (denom_high - Ty_sin), n_plus_half) - max((num_low + Ty_cos) / (denom_low - Ty_sin), n_minus_half));
            const float hWeight_2 = max(0.0f, 1.0f - hWeight_1 - hWeight_0);

            const float v_phi_x_step = T_f.z / (T_v * R_minus_x_dot_theta);
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
        g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = T_f.x * sqrt(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
    }
}
//*/

__global__ void modularBeamBackprojectorKernel_SF_stack(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    //const float z = float(k) * T_f.z + startVals_f.z;

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    //const float half_T_z = 0.5f * T_f.z;

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        const float L = (float)iphi + 0.5f;

        const float3 sourcePosition = make_float3(sourcePositions[3 * iphi + 0], sourcePositions[3 * iphi + 1], sourcePositions[3 * iphi + 2]);
        const float3 moduleCenter = make_float3(moduleCenters[3 * iphi + 0], moduleCenters[3 * iphi + 1], moduleCenters[3 * iphi + 2]);
        const float3 v_vec = make_float3(rowVectors[3 * iphi + 0], rowVectors[3 * iphi + 1], rowVectors[3 * iphi + 2]);
        const float3 u_vec = make_float3(colVectors[3 * iphi + 0], colVectors[3 * iphi + 1], colVectors[3 * iphi + 2]);

        const float3 n_vec = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
            u_vec.z * v_vec.x - u_vec.x * v_vec.z,
            u_vec.x * v_vec.y - u_vec.y * v_vec.x);

        const float c_minus_s_dot_u = (moduleCenter.x - sourcePosition.x) * u_vec.x + (moduleCenter.y - sourcePosition.y) * u_vec.y + (moduleCenter.z - sourcePosition.z) * u_vec.z;
        const float c_minus_s_dot_v = (moduleCenter.x - sourcePosition.x) * v_vec.x + (moduleCenter.y - sourcePosition.y) * v_vec.y + (moduleCenter.z - sourcePosition.z) * v_vec.z;
        const float c_minus_s_dot_n = (moduleCenter.x - sourcePosition.x) * n_vec.x + (moduleCenter.y - sourcePosition.y) * n_vec.y + (moduleCenter.z - sourcePosition.z) * n_vec.z;
        if (fabs(x - sourcePosition.x) > fabs(y - sourcePosition.y))
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = float(k+k_offset) * T_f.z + startVals_f.z;
                const float3 x_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

                const float denom = x_minus_s.x * n_vec.x + x_minus_s.y * n_vec.y + x_minus_s.z * n_vec.z;
                const float t_C = c_minus_s_dot_n / denom;
                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.y);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.y);

                const float u_arg_A = t_A * (x_minus_s.x * u_vec.x + (x_minus_s.y - half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;
                const float u_arg_B = t_B * (x_minus_s.x * u_vec.x + (x_minus_s.y + half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;

                //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(x - sourcePosition[0]);
                const float l_phi = sqrtf((x_minus_s.x * x_minus_s.x + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.x * x_minus_s.x);
                //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) / fabs(x - sourcePosition[0]);

                // Weights for u
                const float tau_low = (min(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;
                const float tau_high = (max(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;

                float u_ind_first = floor(tau_low + 0.5f); // first detector index

                const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                //const float v_val = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;
                //const float vWeight = sqrtf(1.0f + v_val*v_val);

                //const float v_phi_x = (v_val - startVals_g.y) * Tv_inv;
                const float v_phi_x = (t_C * (x_minus_s.x * v_vec.x + x_minus_s.y * v_vec.y + x_minus_s.z * v_vec.z) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;
                const float v_phi_x_step_A = t_C * (T_f.z * v_vec.z) * Tv_inv;

                const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                if (z_high_A > 1.0f)
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (tex3D<float>(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + tex3D<float>(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                }
                else
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                }
            }
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = float(k + k_offset) * T_f.z + startVals_f.z;
                const float3 x_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

                const float denom = x_minus_s.x * n_vec.x + x_minus_s.y * n_vec.y + x_minus_s.z * n_vec.z;
                const float t_C = c_minus_s_dot_n / denom;
                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.x);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.x);

                const float u_arg_A = t_A * ((x_minus_s.x - half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;
                const float u_arg_B = t_B * ((x_minus_s.x + half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u;

                //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(y - sourcePosition[1]);
                const float l_phi = sqrtf((x_minus_s.y * x_minus_s.y + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.y * x_minus_s.y);

                // Weights for u
                const float tau_low = (min(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;
                const float tau_high = (max(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;

                float u_ind_first = floor(tau_low + 0.5f); // first detector index

                const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
                const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

                const float u_ind_last = u_ind_first + 2.5f;
                u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

                //const float v_val = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;
                //const float vWeight = sqrtf(1.0f + v_val*v_val);

                //const float v_phi_x = (v_val - startVals_g.y) * Tv_inv;
                const float v_phi_x = (t_C * (x_minus_s.x * v_vec.x + x_minus_s.y * v_vec.y + x_minus_s.z * v_vec.z) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;
                const float v_phi_x_step_A = t_C * (T_f.z * v_vec.z) * Tv_inv;

                const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
                const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

                const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
                const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
                const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
                const float row_high_plus_two_A = row_high_A + 2.0f;

                if (z_high_A > 1.0f)
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                        + (tex3D<float>(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                            + tex3D<float>(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f);
                }
                else
                {
                    vals[k_offset] += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
                }
            }
        }
    }

    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * T_f.x;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * T_f.x;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * T_f.x;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * T_f.x;
        }
    }
}

//*
__global__ void modularBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float z = float(k) * T_f.z + startVals_f.z;

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    //const float half_T_z = 0.5f * T_f.z;

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        const float L = (float)iphi + 0.5f;

        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        const float c_minus_s_dot_u = (moduleCenter[0] - sourcePosition[0]) * u_vec[0] + (moduleCenter[1] - sourcePosition[1]) * u_vec[1] + (moduleCenter[2] - sourcePosition[2]) * u_vec[2];
        const float c_minus_s_dot_v = (moduleCenter[0] - sourcePosition[0]) * v_vec[0] + (moduleCenter[1] - sourcePosition[1]) * v_vec[1] + (moduleCenter[2] - sourcePosition[2]) * v_vec[2];
        const float c_minus_s_dot_n = (moduleCenter[0] - sourcePosition[0]) * detNormal.x + (moduleCenter[1] - sourcePosition[1]) * detNormal.y + (moduleCenter[2] - sourcePosition[2]) * detNormal.z;
        if (fabs(x - sourcePosition[0]) > fabs(y - sourcePosition[1]))
        {
            const float denom = (x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z;
            const float t_C = c_minus_s_dot_n / denom;
            const float t_A = c_minus_s_dot_n / (denom - half_T_x * detNormal.y);
            const float t_B = c_minus_s_dot_n / (denom + half_T_x * detNormal.y);

            const float u_arg_A = t_A * ((x - sourcePosition[0]) * u_vec[0] + (y - half_T_x - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;
            const float u_arg_B = t_B * ((x - sourcePosition[0]) * u_vec[0] + (y + half_T_x - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;

            //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(x - sourcePosition[0]);
            const float l_phi = sqrtf(((x - sourcePosition[0]) * (x - sourcePosition[0]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) *((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((x - sourcePosition[0])* (x - sourcePosition[0]));
            //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) / fabs(x - sourcePosition[0]);

            // Weights for u
            const float tau_low = (min(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;
            const float tau_high = (max(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;

            float u_ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float u_ind_last = u_ind_first + 2.5f;
            u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            //const float v_val = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;
            //const float vWeight = sqrtf(1.0f + v_val*v_val);

            //const float v_phi_x = (v_val - startVals_g.y) * Tv_inv;
            const float v_phi_x = (t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;
            const float v_phi_x_step_A = t_C * (T_f.z * v_vec[2]) * Tv_inv;

            const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (tex3D<float>(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f);
            }
            else
            {
                val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
            }
        }
        else
        {
            const float denom = (x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z;
            const float t_C = c_minus_s_dot_n / denom;
            const float t_A = c_minus_s_dot_n / (denom - half_T_x * detNormal.x);
            const float t_B = c_minus_s_dot_n / (denom + half_T_x * detNormal.x);

            const float u_arg_A = t_A * ((x - half_T_x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;
            const float u_arg_B = t_B * ((x + half_T_x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;

            //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(y - sourcePosition[1]);
            const float l_phi = sqrtf(((y - sourcePosition[1]) * (y - sourcePosition[1]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) * ((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((y - sourcePosition[1]) * (y - sourcePosition[1]));

            // Weights for u
            const float tau_low = (min(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;
            const float tau_high = (max(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;

            float u_ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            const float u_ind_last = u_ind_first + 2.5f;
            u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            //const float v_val = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;
            //const float vWeight = sqrtf(1.0f + v_val*v_val);

            //const float v_phi_x = (v_val - startVals_g.y) * Tv_inv;
            const float v_phi_x = (t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;
            const float v_phi_x_step_A = t_C * (T_f.z * v_vec[2]) * Tv_inv;

            const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
            const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
            const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
            const float row_high_plus_two_A = row_high_A + 2.0f;

            if (z_high_A > 1.0f)
            {
                val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                    + (tex3D<float>(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                        + tex3D<float>(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f);
            }
            else
            {
                val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                    + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
            }
        }
    }
    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}
//*/

/*
__global__ void modularBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq)
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

    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float z = float(k) * T_f.z + startVals_f.z;

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    //const float half_T_z = 0.5f * T_f.z;

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        const float L = (float)iphi + 0.5f;

        const float3 sourcePosition = make_float3(sourcePositions[3 * iphi + 0], sourcePositions[3 * iphi + 1], sourcePositions[3 * iphi + 2]);
        const float3 c_minus_s = make_float3(moduleCenters[3 * iphi + 0] - sourcePosition.x, moduleCenters[3 * iphi + 1] - sourcePosition.y, moduleCenters[3 * iphi + 2] - sourcePosition.z);
        const float3 v_vec = make_float3(rowVectors[3 * iphi + 0], rowVectors[3 * iphi + 1], rowVectors[3 * iphi + 2]);
        const float3 u_vec = make_float3(colVectors[3 * iphi + 0], colVectors[3 * iphi + 1], colVectors[3 * iphi + 2]);

        //const float* sourcePosition = &sourcePositions[3 * iphi];
        //const float* moduleCenter = &moduleCenters[3 * iphi];
        //const float* v_vec = &rowVectors[3 * iphi];
        //const float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
            u_vec.z * v_vec.x - u_vec.x * v_vec.z,
            u_vec.x * v_vec.y - u_vec.y * v_vec.x); // 9 ops

        const float3 vox_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

        const float c_minus_s_dot_u = c_minus_s.x * u_vec.x + c_minus_s.y * u_vec.y + c_minus_s.z * u_vec.z; // 5 ops
        const float c_minus_s_dot_v = c_minus_s.x * v_vec.x + c_minus_s.y * v_vec.y + c_minus_s.z * v_vec.z; // 5 ops
        const float c_minus_s_dot_n = c_minus_s.x * detNormal.x + c_minus_s.y * detNormal.y + c_minus_s.z * detNormal.z; // 5 ops

        const float vox_minus_s_max = max(fabs(vox_minus_s.x), fabs(vox_minus_s.y));
        float denom_shift, num_shift;
        if (fabs(vox_minus_s.x) > fabs(vox_minus_s.y))
        {
            denom_shift = half_T_x * detNormal.y;
            num_shift = half_T_x * u_vec.y;
        }
        else
        {
            denom_shift = half_T_x * detNormal.x;
            num_shift = half_T_x * u_vec.x;
        }

        const float denom = vox_minus_s.x * detNormal.x + vox_minus_s.y * detNormal.y + vox_minus_s.z * detNormal.z;
        const float t_C = c_minus_s_dot_n / denom;
        const float t_A = c_minus_s_dot_n / (denom - denom_shift);
        const float t_B = c_minus_s_dot_n / (denom + denom_shift);

        const float vox_minus_s_dot_u = vox_minus_s.x * u_vec.x + vox_minus_s.y * u_vec.y + vox_minus_s.z * u_vec.z;

        //const float u_arg_A = t_A * (vox_minus_s.x * u_vec.x + vox_minus_s.y * u_vec.y + vox_minus_s.z * u_vec.z - num_shift) - c_minus_s_dot_u;
        //const float u_arg_B = t_B * (vox_minus_s.x * u_vec.x + vox_minus_s.y * u_vec.y + vox_minus_s.z * u_vec.z + num_shift) - c_minus_s_dot_u;
        const float u_arg_A = t_A * (vox_minus_s_dot_u - num_shift) - c_minus_s_dot_u;
        const float u_arg_B = t_B * (vox_minus_s_dot_u + num_shift) - c_minus_s_dot_u;

        const float l_phi = 1.0f / (rsqrtf(vox_minus_s.x * vox_minus_s.x + vox_minus_s.y * vox_minus_s.y) * vox_minus_s_max);

        // Weights for u
        const float tau_low = (min(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;
        const float tau_high = (max(u_arg_A, u_arg_B) - startVals_g.z) * Tu_inv;

        float u_ind_first = floor(tau_low + 0.5f); // first detector index

        const float horizontalWeights_0_A = (min(tau_high, u_ind_first + 1.5f) - tau_low) * l_phi;
        const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

        const float u_ind_last = u_ind_first + 2.5f;
        u_ind_first = u_ind_first + 0.5f + max(0.0f, min(tau_high - u_ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

        const float v_phi_x = (t_C * (vox_minus_s.x * v_vec.x + vox_minus_s.y * v_vec.y + vox_minus_s.z * v_vec.z) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;
        const float v_phi_x_step_A = t_C * (T_f.z * v_vec.z) * Tv_inv;

        const float row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
        const float z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

        const float v_weight_one = min(v_phi_x_step_A, v_phi_x_step_A - z_high_A);
        const float v_weight_two = max(0.0f, min(z_high_A, 1.0f));
        const float v_oneAndTwo = v_weight_two / (v_weight_one + v_weight_two);
        const float row_high_plus_two_A = row_high_A + 2.0f;

        if (z_high_A > 1.0f)
        {
            val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two)
                + (tex3D<float>(g, u_ind_first, row_high_plus_two_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, u_ind_last, row_high_plus_two_A, L) * horizontalWeights_1_A) * (z_high_A - 1.0f);
        }
        else
        {
            val += (tex3D<float>(g, u_ind_first, row_high_A + v_oneAndTwo, L) * horizontalWeights_0_A
                + tex3D<float>(g, u_ind_last, row_high_A + v_oneAndTwo, L) * horizontalWeights_1_A) * (v_weight_one + v_weight_two);
        }
    }
    f[ind] = val * T_f.x;
}
//*/

__global__ void modularBeamBackprojectorKernel_eSF_stack(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    const float half_T_z = 0.5f * T_f.z;

    float vals[NUM_SLICES_PER_THREAD];
    for (int k_offset = 0; k_offset < numZ; k_offset++)
        vals[k_offset] = 0.0f;

    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        //const float L = (float)iphi + 0.5f;

        const float3 sourcePosition = make_float3(sourcePositions[3 * iphi + 0], sourcePositions[3 * iphi + 1], sourcePositions[3 * iphi + 2]);
        const float3 moduleCenter = make_float3(moduleCenters[3 * iphi + 0], moduleCenters[3 * iphi + 1], moduleCenters[3 * iphi + 2]);
        const float3 v_vec = make_float3(rowVectors[3 * iphi + 0], rowVectors[3 * iphi + 1], rowVectors[3 * iphi + 2]);
        const float3 u_vec = make_float3(colVectors[3 * iphi + 0], colVectors[3 * iphi + 1], colVectors[3 * iphi + 2]);

        const float3 n_vec = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
            u_vec.z * v_vec.x - u_vec.x * v_vec.z,
            u_vec.x * v_vec.y - u_vec.y * v_vec.x);

        const float c_minus_s_dot_u = (moduleCenter.x - sourcePosition.x) * u_vec.x + (moduleCenter.y - sourcePosition.y) * u_vec.y + (moduleCenter.z - sourcePosition.z) * u_vec.z;
        const float c_minus_s_dot_v = (moduleCenter.x - sourcePosition.x) * v_vec.x + (moduleCenter.y - sourcePosition.y) * v_vec.y + (moduleCenter.z - sourcePosition.z) * v_vec.z;
        const float c_minus_s_dot_n = (moduleCenter.x - sourcePosition.x) * n_vec.x + (moduleCenter.y - sourcePosition.y) * n_vec.y + (moduleCenter.z - sourcePosition.z) * n_vec.z;
        if (fabs(x - sourcePosition.x) > fabs(y - sourcePosition.y))
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = float(k + k_offset) * T_f.z + startVals_f.z;
                const float3 x_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

                const float denom = x_minus_s.x * n_vec.x + x_minus_s.y * n_vec.y + x_minus_s.z * n_vec.z;
                const float t_C = c_minus_s_dot_n / denom;
                const float v_c = t_C * (x_minus_s.x * v_vec.x + x_minus_s.y * v_vec.y + x_minus_s.z * v_vec.z) - c_minus_s_dot_v;
                const int div = max(1, int(ceil(half_T_z * v_vec.z * t_C * Tv_inv)));

                const float v_A = (v_c - half_T_z * v_vec.z * t_C - startVals_g.y) * Tv_inv;
                const float v_B = (v_c + half_T_z * v_vec.z * t_C - startVals_g.y) * Tv_inv;
                const int iv_min = int(ceil(v_A - 0.5f));
                const int iv_max = int(floor(v_B + 0.5f));

                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.y);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.y);

                const float u_A = (t_A * (x_minus_s.x * u_vec.x + (x_minus_s.y - half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;
                const float u_B = (t_B * (x_minus_s.x * u_vec.x + (x_minus_s.y + half_T_x) * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;

                const float u_min = min(u_A, u_B);
                const float u_max = max(u_A, u_B);

                //const float diu = max(1, int(ceil(half_T_x * u_vec[1] * t_C * Tu_inv)));
                const int iu_min = int(ceil(u_min - 0.5f));
                const int iu_max = int(floor(u_max + 0.5f));

                //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(x - sourcePosition[0]);
                //const float l_phi = sqrtf(((x - sourcePosition[0]) * (x - sourcePosition[0]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) * ((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((x - sourcePosition[0]) * (x - sourcePosition[0]));
                const float l_phi = sqrtf((x_minus_s.x * x_minus_s.x + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.x * x_minus_s.x);

                for (int iu = iu_min; iu <= iu_max; iu += 2)
                {
                    const float uWeight = l_phi * max(0.0f, min(float(iu) + 0.5f, u_max) - max(float(iu) - 0.5f, u_min));
                    const float uWeight_2 = l_phi * max(0.0f, min(float(iu + 1) + 0.5f, u_max) - max(float(iu + 1) - 0.5f, u_min));
                    if (uWeight + uWeight_2 > 0.0f)
                    {
                        const float ushift_12 = uWeight_2 / (uWeight + uWeight_2);
                        for (int iv = iv_min; iv <= iv_max; iv += 2)
                        {
                            // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                            //const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));
                            //const float vWeight = max(0.0, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                            const float vWeight = max(0.0f, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                            const float vWeight_2 = max(0.0f, min(float(iv + 1) + 0.5f, v_B) - max(float(iv + 1) - 0.5f, v_A));

                            if (vWeight + vWeight_2 > 0.0f)
                            {
                                const float vshift_12 = vWeight_2 / (vWeight + vWeight_2);
                                vals[k_offset] += tex3D<float>(g, iu + ushift_12 + 0.5f, iv + vshift_12 + 0.5f, iphi + 0.5f) * (uWeight + uWeight_2) * (vWeight + vWeight_2);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (int k_offset = 0; k_offset < numZ; k_offset++)
            {
                const float z = float(k + k_offset) * T_f.z + startVals_f.z;
                const float3 x_minus_s = make_float3(x - sourcePosition.x, y - sourcePosition.y, z - sourcePosition.z);

                const float denom = x_minus_s.x * n_vec.x + x_minus_s.y * n_vec.y + x_minus_s.z * n_vec.z;
                const float t_C = c_minus_s_dot_n / denom;

                const float v_c = t_C * (x_minus_s.x * v_vec.x + x_minus_s.y * v_vec.y + x_minus_s.z * v_vec.z) - c_minus_s_dot_v;
                const int div = max(1, int(ceil(half_T_z * v_vec.z * t_C * Tv_inv)));

                const float v_A = (v_c - half_T_z * v_vec.z * t_C - startVals_g.y) * Tv_inv;
                const float v_B = (v_c + half_T_z * v_vec.z * t_C - startVals_g.y) * Tv_inv;
                const int iv_min = int(ceil(v_A - 0.5f));
                const int iv_max = int(floor(v_B + 0.5f));

                const float t_A = c_minus_s_dot_n / (denom - half_T_x * n_vec.x);
                const float t_B = c_minus_s_dot_n / (denom + half_T_x * n_vec.x);

                const float u_A = (t_A * ((x_minus_s.x - half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;
                const float u_B = (t_B * ((x_minus_s.x + half_T_x) * u_vec.x + x_minus_s.y * u_vec.y + x_minus_s.z * u_vec.z) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;

                const float u_min = min(u_A, u_B);
                const float u_max = max(u_A, u_B);

                //const float diu = max(1, int(ceil(half_T_x * u_vec[0] * t_C * Tu_inv)));
                const int iu_min = int(ceil(u_min - 0.5f));
                const int iu_max = int(floor(u_max + 0.5f));

                //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(y - sourcePosition[1]);
                //const float l_phi = sqrtf(((y - sourcePosition[1]) * (y - sourcePosition[1]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) * ((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((y - sourcePosition[1]) * (y - sourcePosition[1]));
                const float l_phi = sqrtf((x_minus_s.y * x_minus_s.y + x_minus_s.z * x_minus_s.z) * (x_minus_s.x * x_minus_s.x + x_minus_s.y * x_minus_s.y)) / (x_minus_s.y * x_minus_s.y);

                for (int iu = iu_min; iu <= iu_max; iu += 2)
                {
                    const float uWeight = l_phi * max(0.0f, min(float(iu) + 0.5f, u_max) - max(float(iu) - 0.5f, u_min));
                    const float uWeight_2 = l_phi * max(0.0f, min(float(iu + 1) + 0.5f, u_max) - max(float(iu + 1) - 0.5f, u_min));
                    if (uWeight + uWeight_2 > 0.0f)
                    {
                        const float ushift_12 = uWeight_2 / (uWeight + uWeight_2);
                        for (int iv = iv_min; iv <= iv_max; iv += 2)
                        {
                            // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                            //const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));
                            //const float vWeight = max(0.0, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                            const float vWeight = max(0.0f, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                            const float vWeight_2 = max(0.0f, min(float(iv + 1) + 0.5f, v_B) - max(float(iv + 1) - 0.5f, v_A));

                            if (vWeight + vWeight_2 > 0.0f)
                            {
                                const float vshift_12 = vWeight_2 / (vWeight + vWeight_2);
                                vals[k_offset] += tex3D<float>(g, iu + ushift_12 + 0.5f, iv + vshift_12 + 0.5f, iphi + 0.5f) * (uWeight + uWeight_2) * (vWeight + vWeight_2);
                            }
                        }
                    }
                }
            }
        }
    }

    if (volumeDimensionOrder == 0)
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset)] += vals[k_offset] * T_f.x;
            else
                f[ind + uint64(k_offset)] = vals[k_offset] * T_f.x;
        }
    }
    else
    {
        for (int k_offset = 0; k_offset < numZ; k_offset++)
        {
            if (accum)
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] += vals[k_offset] * T_f.x;
            else
                f[ind + uint64(k_offset) * uint64(N_f.y * N_f.x)] = vals[k_offset] * T_f.x;
        }
    }
}

__global__ void modularBeamBackprojectorKernel_eSF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float z = float(k) * T_f.z + startVals_f.z;

    //const float T_x_inv = 1.0f / T_f.x;
    const float Tu_inv = 1.0f / T_g.z;
    const float Tv_inv = 1.0f / T_g.y;
    const float half_T_x = 0.5f * T_f.x;
    const float half_T_z = 0.5f * T_f.z;

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        //const float L = (float)iphi + 0.5f;

        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        const float c_minus_s_dot_u = (moduleCenter[0] - sourcePosition[0]) * u_vec[0] + (moduleCenter[1] - sourcePosition[1]) * u_vec[1] + (moduleCenter[2] - sourcePosition[2]) * u_vec[2];
        const float c_minus_s_dot_v = (moduleCenter[0] - sourcePosition[0]) * v_vec[0] + (moduleCenter[1] - sourcePosition[1]) * v_vec[1] + (moduleCenter[2] - sourcePosition[2]) * v_vec[2];
        const float c_minus_s_dot_n = (moduleCenter[0] - sourcePosition[0]) * detNormal.x + (moduleCenter[1] - sourcePosition[1]) * detNormal.y + (moduleCenter[2] - sourcePosition[2]) * detNormal.z;

        const float denom = (x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z;
        const float t_C = c_minus_s_dot_n / denom;

        const float v_c = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;
        //const int iv_c = int(0.5f+(v_c - startVals_g.y) * Tv_inv);
        const int div = max(1, int(ceil(half_T_z*v_vec[2]*t_C * Tv_inv)));

        //const float coneWeight = sqrtf(1.0f + v_c*v_c);

        const float v_A = (v_c - half_T_z * v_vec[2] * t_C - startVals_g.y) * Tv_inv;
        const float v_B = (v_c + half_T_z * v_vec[2] * t_C - startVals_g.y) * Tv_inv;
        const int iv_min = int(ceil(v_A - 0.5f));
        const int iv_max = int(floor(v_B + 0.5f));

        //const int iu_c = int(0.5f+(t_C * ((x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u - startVals_g.z) * Tu_inv);

        if (fabs(x - sourcePosition[0]) > fabs(y - sourcePosition[1]))
        {
            const float t_A = c_minus_s_dot_n / (denom - half_T_x * detNormal.y);
            const float t_B = c_minus_s_dot_n / (denom + half_T_x * detNormal.y);

            const float u_A = (t_A * ((x - sourcePosition[0]) * u_vec[0] + (y - half_T_x - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;
            const float u_B = (t_B * ((x - sourcePosition[0]) * u_vec[0] + (y + half_T_x - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;

            const float u_min = min(u_A, u_B);
            const float u_max = max(u_A, u_B);

            //const float diu = max(1, int(ceil(half_T_x * u_vec[1] * t_C * Tu_inv)));
            const int iu_min = int(ceil(u_min-0.5f));
            const int iu_max = int(floor(u_max+0.5f));

            //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(x - sourcePosition[0]);
            const float l_phi = sqrtf(((x - sourcePosition[0]) * (x - sourcePosition[0]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) * ((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((x - sourcePosition[0]) * (x - sourcePosition[0]));

            for (int iu = iu_min; iu <= iu_max; iu += 2)
            {
                const float uWeight = l_phi * max(0.0f, min(float(iu) + 0.5f, u_max) - max(float(iu) - 0.5f, u_min));
                const float uWeight_2 = l_phi * max(0.0f, min(float(iu + 1) + 0.5f, u_max) - max(float(iu + 1) - 0.5f, u_min));
                if (uWeight + uWeight_2 > 0.0f)
                {
                    const float ushift_12 = uWeight_2 / (uWeight + uWeight_2);
                    for (int iv = iv_min; iv <= iv_max; iv += 2)
                    {
                        // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                        //const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));
                        //const float vWeight = max(0.0, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                        const float vWeight = max(0.0f, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                        const float vWeight_2 = max(0.0f, min(float(iv + 1) + 0.5f, v_B) - max(float(iv + 1) - 0.5f, v_A));

                        if (vWeight + vWeight_2 > 0.0f)
                        {
                            const float vshift_12 = vWeight_2 / (vWeight + vWeight_2);
                            val += tex3D<float>(g, iu + ushift_12 + 0.5f, iv + vshift_12 + 0.5f, iphi + 0.5f) * (uWeight + uWeight_2) * (vWeight + vWeight_2);
                        }
                    }
                }
            }
        }
        else
        {
            const float t_A = c_minus_s_dot_n / (denom - half_T_x * detNormal.x);
            const float t_B = c_minus_s_dot_n / (denom + half_T_x * detNormal.x);

            const float u_A = (t_A * ((x - half_T_x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;
            const float u_B = (t_B * ((x + half_T_x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u - startVals_g.z) * Tu_inv;

            const float u_min = min(u_A, u_B);
            const float u_max = max(u_A, u_B);

            //const float diu = max(1, int(ceil(half_T_x * u_vec[0] * t_C * Tu_inv)));
            const int iu_min = int(ceil(u_min - 0.5f));
            const int iu_max = int(floor(u_max + 0.5f));

            //const float l_phi = sqrtf((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1])) / fabs(y - sourcePosition[1]);
            const float l_phi = sqrtf(((y - sourcePosition[1]) * (y - sourcePosition[1]) + (z - sourcePosition[2]) * (z - sourcePosition[2])) * ((x - sourcePosition[0]) * (x - sourcePosition[0]) + (y - sourcePosition[1]) * (y - sourcePosition[1]))) / ((y - sourcePosition[1]) * (y - sourcePosition[1]));

            for (int iu = iu_min; iu <= iu_max; iu += 2)
            {
                const float uWeight = l_phi * max(0.0f, min(float(iu) + 0.5f, u_max) - max(float(iu) - 0.5f, u_min));
                const float uWeight_2 = l_phi * max(0.0f, min(float(iu + 1) + 0.5f, u_max) - max(float(iu + 1) - 0.5f, u_min));
                if (uWeight + uWeight_2 > 0.0f)
                {
                    const float ushift_12 = uWeight_2 / (uWeight + uWeight_2);
                    for (int iv = iv_min; iv <= iv_max; iv += 2)
                    {
                        // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                        //const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));
                        //const float vWeight = max(0.0, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                        const float vWeight = max(0.0f, min(float(iv) + 0.5f, v_B) - max(float(iv) - 0.5f, v_A));
                        const float vWeight_2 = max(0.0f, min(float(iv + 1) + 0.5f, v_B) - max(float(iv + 1) - 0.5f, v_A));

                        if (vWeight + vWeight_2 > 0.0f)
                        {
                            const float vshift_12 = vWeight_2 / (vWeight + vWeight_2);
                            val += tex3D<float>(g, iu + ushift_12 + 0.5f, iv + vshift_12 + 0.5f, iphi + 0.5f) * (uWeight + uWeight_2) * (vWeight + vWeight_2);
                        }
                    }
                }
            }
        }
    }
    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}

__device__ float lineIntegral_Joseph_ZYX(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return 0.0f;
            int ix_max = min(N.x - 1, int(ceil((dst.x - startVal.x) / T.x)));

            val = tex3D<float>(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(ix_start) - 0.5f) - max(-0.5f, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;

            for (int ix = ix_start; ix <= ix_max; ix++)
                val += tex3D<float>(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix));
        }
        else
        {
            if (ip.x <= -0.5f) return 0.0f;
            int ix_min = max(0, int(floor((dst.x - startVal.x) / T.x)));

            val = tex3D<float>(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.x) - 0.5f), ip.x) - (float(ix_start) + 0.5f));

            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix >= ix_min; ix--)
                val += tex3D<float>(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix));
        }
        return val * sqrt(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return 0.0f;
            int iy_max = min(N.y - 1, int(ceil((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, ix_start + 0.5f, float(iy_start) + 0.5f, iz_start + 0.5f) *
                ((float(iy_start) - 0.5f) - max(-0.5f, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            //printf("p = %f, %f, %f to dst = %f, %f, %f\n", p.x, p.y, p.z, dst.x, dst.y, dst.z);
            //printf("istarts = %f, %d, %f\n", ix_start, iy_start, iz_start);
            for (int iy = iy_start; iy <= iy_max; iy++)
            {
                //printf("iy = %d: indices = %f, %f, %f; update: %f\n", iy, ix_offset + ir.x * float(iy)-0.5f, float(iy) + 0.5f - 0.5f, iz_offset + ir.z * float(iy) - 0.5f, tex3D<float>(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy)));
                val += tex3D<float>(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
            }
        }
        else
        {
            if (ip.y <= -0.5f) return 0.0f;
            int iy_min = max(0, int(floor((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.y) - 0.5f), ip.y) - (float(iy_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy >= iy_min; iy--)
                val += tex3D<float>(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy));
        }
        //printf("left edge = %f\n", tex3D<float>(mu, -0.9f+0.5f, 0.5f, 0.5f));
        //printf("forward project rayWeight = sqrt(1.0 + (%f)^2 + (%f)^2) = %f\n", ir.x, ir.z, sqrt(1.0f + ir.x * ir.x + ir.z * ir.z));
        return val * sqrt(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
    }
    else
    {
        // integral in z direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return 0.0f;
            int iz_max = min(N.z - 1, int(ceil((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(iz_start) - 0.5f) - max(-0.5f, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz <= iz_max; iz++)
                val += tex3D<float>(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f);
        }
        else
        {
            if (ip.z <= -0.5f) return 0.0f;
            int iz_min = max(0, int(floor((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, float(iz_start) + 0.5f) *
                (min((float(N.z) - 0.5f), ip.z) - (float(iz_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz >= iz_min; iz--)
                val += tex3D<float>(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f);
        }
        return val * sqrt(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
    }
}

__device__ float lineIntegral_Joseph_XYZ(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
        (p.z - startVal.z) / T.z);
    const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction

        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return 0.0f;
            int ix_max = min(N.x - 1, int(ceil((dst.x - startVal.x) / T.x)));

            val = tex3D<float>(mu, iz_start + 0.5f, iy_start + 0.5f, float(ix_start) + 0.5f) *
                ((float(ix_start) - 0.5f) - max(-0.5f, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;

            for (int ix = ix_start; ix <= ix_max; ix++)
                val += tex3D<float>(mu, iz_offset + ir.z * float(ix), iy_offset + ir.y * float(ix), float(ix) + 0.5f);
        }
        else
        {
            if (ip.x <= -0.5f) return 0.0f;
            int ix_min = max(0, int(floor((dst.x - startVal.x) / T.x)));
            val = tex3D<float>(mu, iz_start + 0.5f, iy_start + 0.5f, float(ix_start) + 0.5f) *
                (min((float(N.x) - 0.5f), ip.x) - (float(ix_start) + 0.5f));

            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix >= ix_min; ix--)
                val += tex3D<float>(mu, iz_offset - ir.z * float(ix), iy_offset - ir.y * float(ix), float(ix) + 0.5f);
        }
        return val * sqrt(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return 0.0f;
            int iy_max = min(N.y - 1, int(ceil((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, iz_start + 0.5f, float(iy_start) + 0.5f, ix_start + 0.5f) *
                ((float(iy_start) - 0.5f) - max(-0.5f, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy <= iy_max; iy++)
                val += tex3D<float>(mu, iz_offset + ir.z * float(iy), float(iy) + 0.5f, ix_offset + ir.x * float(iy));
        }
        else
        {
            if (ip.y <= -0.5f) return 0.0f;
            int iy_min = max(0, int(floor((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, iz_start + 0.5f, iy_start + 0.5f, ix_start + 0.5f) *
                (min((float(N.y) - 0.5f), ip.y) - (float(iy_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy >= iy_min; iy--)
                val += tex3D<float>(mu, iz_offset - ir.z * float(iy), float(iy) + 0.5f, ix_offset - ir.x * float(iy));
        }
        return val * sqrt(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
    }
    else
    {
        // integral in z direction
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return 0.0f;
            int iz_max = min(N.z - 1, int(ceil((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, iz_start + 0.5f, iy_start + 0.5f, ix_start + 0.5f) *
                ((float(iz_start) - 0.5f) - max(-0.5f, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz <= iz_max; iz++)
                val += tex3D<float>(mu, float(iz) + 0.5f, iy_offset + ir.y * float(iz), ix_offset + ir.x * float(iz));
        }
        else
        {
            if (ip.z <= -0.5f) return 0.0f;
            int iz_min = max(0, int(floor((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, float(iz_start) + 0.5f, iy_start + 0.5f, ix_start + 0.5f) *
                (min((float(N.z) - 0.5f), ip.z) - (float(iz_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz >= iz_min; iz--)
                val += tex3D<float>(mu, float(iz) + 0.5f, iy_offset - ir.y * float(iz), ix_offset - ir.x * float(iz));
        }
        return val * sqrt(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
    }
}

__global__ void modularBeamParallelJosephBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    if (x * x + y * y > rFOV_sq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float z = float(k) * T_f.z + startVals_f.z;

    const float T_x_inv = 1.0f / T_f.x;
    //const float T_x_over2 = 0.5f * T_f.x;

    const float T_v_inv = 1.0f / T_g.y;
    const float T_u_inv = 1.0f / T_g.z;

    const float footprintSize_u = T_f.x * T_u_inv;
    const float footprintSize_v = T_f.z * T_v_inv;
    const int searchWidth_u = 1 + int(ceil(0.5f * footprintSize_u));
    const int searchWidth_v = 1 + int(ceil(0.5f * footprintSize_v));

    float val = 0.0f;
    for (int iphi = 0; iphi < N_g.x; iphi++)
    {
        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        float3 r = make_float3(moduleCenter[0] - sourcePosition[0], moduleCenter[1] - sourcePosition[1], moduleCenter[2] - sourcePosition[2]);
        const float D = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

        const float odd = ((moduleCenter[0] - x) * detNormal.x + (moduleCenter[1] - y) * detNormal.y + (moduleCenter[2] - z) * detNormal.z) / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
        const float u_arg = (x + odd * r.x - moduleCenter[0]) * u_vec[0] + (y + odd * r.y - moduleCenter[1]) * u_vec[1] + (z + odd * r.z - moduleCenter[2]) * u_vec[2];
        const float v_arg = (x + odd * r.x - moduleCenter[0]) * v_vec[0] + (y + odd * r.y - moduleCenter[1]) * v_vec[1] + (z + odd * r.z - moduleCenter[2]) * v_vec[2];

        const int u_ind = int(floor(0.5f + (u_arg - startVals_g.z) * T_u_inv));
        const int v_ind = int(floor(0.5f + (v_arg - startVals_g.y) * T_v_inv));

        //const int searchWidth_u = 1 + int(ceil(0.5f * T_f.x / T_g.z));
        //const int searchWidth_v = 1 + int(ceil(0.5f * T_f.z / T_g.y));

        float val_local = 0.0;
        float sum_weights = 0.0;

        if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
        {
            const float r_x_inv = 1.0f / r.x;
            //const float rayWeight = D * fabs(r_x_inv);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float dx = (moduleCenter[0] - x + u * u_vec[0] + v * v_vec[0]) * r_x_inv;
                    const float yy = moduleCenter[1] + u * u_vec[1] + v * v_vec[1] - dx * r.y;
                    const float zz = moduleCenter[2] + u * u_vec[2] + v * v_vec[2] - dx * r.z;

                    const float dy = max(0.0f, 1.0f - fabs(y - yy) * T_x_inv);
                    const float dz = max(0.0f, 1.0f - fabs(z - zz) * T_x_inv);

                    if (dy > 0.0f && dz > 0.0f)
                    {
                        sum_weights += dy * dz;
                        val_local += dy * dz * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += dy * dz * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }
        else if (fabs(r.y) >= fabs(r.z))
        {
            const float r_y_inv = 1.0f / r.y;
            //const float rayWeight = D * fabs(r_y_inv);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float dy = (moduleCenter[1] - y + u * u_vec[1] + v * v_vec[1]) * r_y_inv;
                    const float xx = moduleCenter[0] + u * u_vec[0] + v * v_vec[0] - dy * r.x;
                    const float zz = moduleCenter[2] + u * u_vec[2] + v * v_vec[2] - dy * r.z;

                    const float dx = max(0.0f, 1.0f - fabs(x - xx) * T_x_inv);
                    const float dz = max(0.0f, 1.0f - fabs(z - zz) * T_x_inv);

                    if (dx > 0.0f && dz > 0.0f)
                    {
                        sum_weights += dx * dz;
                        val_local += dx * dz * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += dx * dz * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }
        else
        {
            const float r_z_inv = 1.0f / r.z;
            //const float rayWeight = D * fabs(r_z_inv);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float dz = (moduleCenter[2] - z + u * u_vec[2] + v * v_vec[2]) * r_z_inv;
                    const float xx = moduleCenter[0] + u * u_vec[0] + v * v_vec[0] - dz * r.x;
                    const float yy = moduleCenter[1] + u * u_vec[1] + v * v_vec[1] - dz * r.y;

                    const float dx = max(0.0f, 1.0f - fabs(x - xx) * T_x_inv);
                    const float dy = max(0.0f, 1.0f - fabs(y - yy) * T_x_inv);

                    if (dx > 0.0f && dy > 0.0f)
                    {
                        sum_weights += dx * dy;
                        val_local += dx * dy * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += dx * dy * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }

        if (sum_weights > 0.0f)
            val += val_local * footprintSize_u * footprintSize_v / sum_weights;

    }
    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}

__global__ void modularBeamJosephBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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
        float* sourcePosition = &sourcePositions[3 * iphi];
        float* moduleCenter = &moduleCenters[3 * iphi];
        float* v_vec = &rowVectors[3 * iphi];
        float* u_vec = &colVectors[3 * iphi];
        const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
                                             u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
                                             u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

        float3 r = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);
        const float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float D = -p_minus_c_dot_n / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);

        //<p_minus_c + lineLength*r, u>
        //<p_minus_c + lineLength*r, v>
        const float u_arg = (p_minus_c.x + D * r.x) * u_vec[0] + (p_minus_c.y + D * r.y) * u_vec[1] + (p_minus_c.z + D * r.z) * u_vec[2];
        const float v_arg = (p_minus_c.x + D * r.x) * v_vec[0] + (p_minus_c.y + D * r.y) * v_vec[1] + (p_minus_c.z + D * r.z) * v_vec[2];

        const int u_ind = int(floor(0.5f + (u_arg - startVals_g.z) * T_u_inv));
        const int v_ind = int(floor(0.5f + (v_arg - startVals_g.y) * T_v_inv));

        // D is not necessarily the distance from the source to detector, the distance is R*D
        //const int searchWidth_u = 1 + int(0.5f * T_f.x / (R / D * T_g.z));
        //const int searchWidth_v = 1 + int(0.5f * T_f.z / (R / D * T_g.y));
        const float footprintSize_u = T_f.x * T_u_inv * fabs(D);
        const float footprintSize_v = T_f.z * T_v_inv * fabs(D);
        const int searchWidth_u = 1 + int(ceil(0.5f * footprintSize_u));
        const int searchWidth_v = 1 + int(ceil(0.5f * footprintSize_v));

        float val_local = 0.0;
        float sum_weights = 0.0;

        //*
        r.x *= T_x_inv;
        r.y *= T_x_inv;
        r.z *= T_x_inv;
        if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
        {
            /*
            // Footprint spanned by [y-0.5f*T_f.y, y+0.5*T_f.y] and [z-0.5f*T_f.z, z+0.5f*T_f.z]
            const float du_y = fabs(T_x_over2 * D * u_vec[1] * T_u_inv);
            const float du_z = fabs(T_x_over2 * D * u_vec[2] * T_u_inv);

            const float dv_y = fabs(T_x_over2 * D * v_vec[1] * T_v_inv);
            const float dv_z = fabs(T_x_over2 * D * v_vec[2] * T_v_inv);
            //*/

            //const float rayWeight = R / fabs(r.x) * T_x_inv;
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const float trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const float trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const float trueRay_x_inv = 1.0f / trueRay_x;

                    const float dy = max(0.0f, 1.0f - fabs(r.y - r.x * trueRay_y * trueRay_x_inv));
                    const float dz = max(0.0f, 1.0f - fabs(r.z - r.x * trueRay_z * trueRay_x_inv));

                    if (dy > 0.0f && dz > 0.0f)
                    {
                        sum_weights += dy * dz;
                        val_local += dy * dz * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += tex3D<float>(g, iu, iv, iphi);
                        //val += 1.0f;
                    }
                }
            }
        }
        else if (fabs(r.y) >= fabs(r.z))
        {
            //const float rayWeight = R / fabs(r.y) * T_x_inv;
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const float trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const float trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const float trueRay_y_inv = 1.0f / trueRay_y;

                    const float dx = max(0.0f, 1.0f - fabs(r.x - r.y * trueRay_x * trueRay_y_inv));
                    const float dz = max(0.0f, 1.0f - fabs(r.z - r.y * trueRay_z * trueRay_y_inv));

                    if (dx > 0.0f && dz > 0.0f)
                    {
                        sum_weights += dx * dz;
                        val_local += dx * dz * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += tex3D<float>(g, iu, iv, iphi);
                        //val += 1.0f;
                    }
                }
            }
        }
        else
        {
            //const float rayWeight = R / fabs(r.z) * T_x_inv;
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    // trueRay is the vector from the source to the detector pixel in the loop
                    const float trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const float trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const float trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const float trueRay_z_inv = 1.0f / trueRay_z;

                    const float dy = max(0.0f, 1.0f - fabs(r.y - r.z * trueRay_y * trueRay_z_inv));
                    const float dx = max(0.0f, 1.0f - fabs(r.x - r.z * trueRay_x * trueRay_z_inv));

                    if (dx > 0.0f && dy > 0.0f)
                    {
                        sum_weights += dx * dy;
                        val_local += dx * dy * tex3D<float>(g, iu, iv, iphi);// *rayWeight;
                        //val += tex3D<float>(g, iu, iv, iphi);
                        //val += 1.0f;
                    }
                }
            }
        }
        if (sum_weights > 0.0f)
            val += val_local * footprintSize_u * footprintSize_v / sum_weights;
        else
            val += footprintSize_u * footprintSize_v * tex3D<float>(g, u_ind, v_ind, iphi);
        //*/

        /*
        //const double3 rd = make_double3((double(x) - double(sourcePosition[0])) * double(T_x_inv), (double(y) - double(sourcePosition[1])) * double(T_x_inv), (double(z) - double(sourcePosition[2])) * double(T_x_inv));
        const double3 rd = make_double3(double(r.x) * double(T_x_inv), double(r.y) * double(T_x_inv), double(r.z) * double(T_x_inv));
        if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
        {
            const float rayWeight = R / fabs(r.x);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const double v = iv * T_g.y + startVals_g.y;
                //const float v = iv * T_v + v_0;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const double u = iu * T_g.z + startVals_g.z;
                    //const float u = iu * T_u + u_0;

                    const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const double trueRay_x_inv = 1.0 / trueRay_x;

                    const float dy = max(0.0f, 1.0f - fabs(rd.y - rd.x * trueRay_y * trueRay_x_inv));
                    const float dz = max(0.0f, 1.0f - fabs(rd.z - rd.x * trueRay_z * trueRay_x_inv));

                    if (dy > 0.0f && dz > 0.0f)
                    {
                        //val += sqrtf(trueRay_x * trueRay_x + trueRay_y * trueRay_y + trueRay_z * trueRay_z) * fabs(trueRay_x_inv) * dy * dz * tex3D<float>(g, iu, iv, iphi);
                        val += rayWeight * dy * dz * tex3D<float>(g, iu, iv, iphi);
                        //val += dy * dz * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }
        else if (fabs(r.y) >= fabs(r.z))
        {
            const float rayWeight = R / fabs(r.y);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const double v = iv * T_g.y + startVals_g.y;
                //const float v = iv * T_v + v_0;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const double u = iu * T_g.z + startVals_g.z;
                    //const float u = iu * T_u + u_0;

                    const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const double trueRay_y_inv = 1.0 / trueRay_y;

                    const float dx = max(0.0f, 1.0f - fabs(rd.x - rd.y * trueRay_x * trueRay_y_inv));
                    const float dz = max(0.0f, 1.0f - fabs(rd.z - rd.y * trueRay_z * trueRay_y_inv));

                    if (dx > 0.0f && dz > 0.0f)
                    {
                        //val += sqrtf(trueRay_y * trueRay_y + trueRay_x * trueRay_x + trueRay_z * trueRay_z) * fabs(trueRay_y_inv) * dx * dz * tex3D<float>(g, iu, iv, iphi);
                        val += rayWeight * dx * dz * tex3D<float>(g, iu, iv, iphi);
                        //val += dx * dz * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }
        else
        {
            const float rayWeight = R / fabs(r.z);
            for (int iv = max(0, v_ind - searchWidth_v); iv <= v_ind + searchWidth_v; iv++)
            {
                const double v = iv * T_g.y + startVals_g.y;
                //const float v = iv * T_v + v_0;
                for (int iu = max(0, u_ind - searchWidth_u); iu <= u_ind + searchWidth_u; iu++)
                {
                    const double u = iu * T_g.z + startVals_g.z;
                    //const float u = iu * T_u + u_0;

                    const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                    const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                    const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                    const double trueRay_z_inv = 1.0 / trueRay_z;

                    const float dy = max(0.0f, 1.0f - fabs(rd.y - rd.z * trueRay_y * trueRay_z_inv));
                    const float dx = max(0.0f, 1.0f - fabs(rd.x - rd.z * trueRay_x * trueRay_z_inv));

                    if (dx > 0.0f && dy > 0.0f)
                    {
                        //val += sqrtf(trueRay_z * trueRay_z + trueRay_x * trueRay_x + trueRay_y * trueRay_y) * fabs(trueRay_z_inv) * dx * dy * tex3D<float>(g, iu, iv, iphi);
                        val += rayWeight * dx * dy * tex3D<float>(g, iu, iv, iphi);
                        //val += dx * dy * tex3D<float>(g, iu, iv, iphi);
                    }
                }
            }
        }
        //*/
    }
    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}

//#####################################################################################################################
__global__ void modularBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const float rFOV_sq, const bool accum)
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

    const float T_x_inv = 1.0f / T_f.x;
    const float T_x_over2 = 0.5f * T_f.x;

    const float T_v_inv = 1.0f / T_g.y;
    const float T_u_inv = 1.0f / T_g.z;

    float val = 0.0f;
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
        const float R = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float D = -p_minus_c_dot_n / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);

        //<p_minus_c + lineLength*r, u>
        //<p_minus_c + lineLength*r, v>
        const float u_arg = (p_minus_c.x + D * r.x) * u_vec[0] + (p_minus_c.y + D * r.y) * u_vec[1] + (p_minus_c.z + D * r.z) * u_vec[2];
        const float v_arg = (p_minus_c.x + D * r.x) * v_vec[0] + (p_minus_c.y + D * r.y) * v_vec[1] + (p_minus_c.z + D * r.z) * v_vec[2];

        const float u_ind = (u_arg - startVals_g.z) * T_u_inv;
        const float v_ind = (v_arg - startVals_g.y) * T_v_inv;

        // D is not necessarily the distance from the source to detector, the distance is R*D
        //const int searchWidth_u = 1 + int(0.5f * T_f.x / (R / D * T_g.z));
        //const int searchWidth_v = 1 + int(0.5f * T_f.z / (R / D * T_g.y));
        //const int searchWidth_u = 1 + int(ceil(0.5f * T_f.x / T_g.z * fabs(D)));
        //const int searchWidth_v = 1 + int(ceil(0.5f * T_f.z / T_g.y * fabs(D)));

        r.x *= T_x_inv;
        r.y *= T_x_inv;
        r.z *= T_x_inv;
        if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
        {
            // Footprint spanned by [y-0.5f*T_f.y, y+0.5*T_f.y] and [z-0.5f*T_f.z, z+0.5f*T_f.z]
            const float du_y = fabs(T_x_over2 * D * u_vec[1] * T_u_inv);
            const float du_z = fabs(T_x_over2 * D * u_vec[2] * T_u_inv);

            const float dv_y = fabs(T_x_over2 * D * v_vec[1] * T_v_inv);
            const float dv_z = fabs(T_x_over2 * D * v_vec[2] * T_v_inv);

            float alpha;
            if (fabs(u_vec[1]) >= fabs(v_vec[1]))
                alpha = atan2(u_vec[2], u_vec[1]);
            else
                alpha = atan2(v_vec[2], v_vec[1]);
            const float cos_alpha = cos(alpha);
            const float sin_alpha = sin(alpha);

            if (du_y >= dv_y)
            {
                // after rotation u is aligned with y and v with z
                const int iu_min = int(ceil(u_ind - du_y-du_z - 0.5f)) - 1;
                const int iu_max = int(floor(u_ind + du_y+du_z + 0.5f)) + 1;

                const int iv_min = int(ceil(v_ind - dv_z-dv_y - 0.5f)) - 1;
                const int iv_max = int(floor(v_ind + dv_z+dv_y + 0.5f)) + 1;

                for (int iu = iu_min; iu <= iu_max; iu++)
                {
                    for (int iv = iv_min; iv <= iv_max; iv++)
                    {

                    }
                }
            }
            else
            {
                // after rotation u is aligned with z and v with y
            }

            //const int iu_min = int(ceil(u_A - 0.5f));
            //const int iu_max = int(floor(u_B + 0.5f));
        }
        else if (fabs(r.y) >= fabs(r.z))
        {
            
        }
        else
        {
            
        }
        
    }
    if (accum)
        f[ind] += val * T_f.x;
    else
        f[ind] = val * T_f.x;
}
//#####################################################################################################################

__global__ void modularBeamJosephProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder, const bool accum)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const double v = double(j) * double(T_g.y) + double(startVals_g.y);
    const double u = double(k) * double(T_g.z) + double(startVals_g.z);

    float3 sourcePos;
    sourcePos.x = sourcePositions[3 * i + 0];
    sourcePos.y = sourcePositions[3 * i + 1];
    sourcePos.z = sourcePositions[3 * i + 2];

    float3 dst;
    dst.x = float(double(moduleCenters[3 * i + 0]) + v * double(rowVectors[3 * i + 0]) + u * double(colVectors[3 * i + 0]));
    dst.y = float(double(moduleCenters[3 * i + 1]) + v * double(rowVectors[3 * i + 1]) + u * double(colVectors[3 * i + 1]));
    dst.z = float(double(moduleCenters[3 * i + 2]) + v * double(rowVectors[3 * i + 2]) + u * double(colVectors[3 * i + 2]));

    const double3 r = make_double3(double(dst.x) - double(sourcePos.x), double(dst.y) - double(sourcePos.y), double(dst.z) - double(sourcePos.z));
    double t = 0.0f;
    if (fabs(r.x) > max(fabs(r.y), fabs(r.z)))
    {
        if (r.x < 0.0f)
            t = (double(N_f.x - 1) * double(T_f.x) + double(startVals_f.x) - double(sourcePos.x)) / r.x;
        else
            t = (double(startVals_f.x) - double(sourcePos.x)) / r.x;
    }
    else if (fabs(r.y) > fabs(r.z))
    {
        if (r.y < 0.0f)
            t = (double(N_f.y - 1) * double(T_f.y) + double(startVals_f.y) - double(sourcePos.y)) / r.y;
        else
            t = double(double(startVals_f.y) - double(sourcePos.y)) / r.y;
    }
    else
    {
        if (r.z < 0.0f)
            t = (double(N_f.z - 1) * double(T_f.z) + double(startVals_f.z) - double(sourcePos.z)) / r.z;
        else
            t = (double(startVals_f.z) - double(sourcePos.z)) / r.z;
    }
    const float3 edgePos = make_float3(float(double(sourcePos.x) + t * r.x), float(double(sourcePos.y) + t * r.y), float(double(sourcePos.z) + t * r.z));

    if (accum)
    {
        if (volumeDimensionOrder == 0)
            g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] += lineIntegral_Joseph_XYZ(f, N_f, T_f, startVals_f, edgePos, dst);
        else
            g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] += lineIntegral_Joseph_ZYX(f, N_f, T_f, startVals_f, edgePos, dst);
    }
    else
    {
        if (volumeDimensionOrder == 0)
            g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] = lineIntegral_Joseph_XYZ(f, N_f, T_f, startVals_f, edgePos, dst);
        else
            g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] = lineIntegral_Joseph_ZYX(f, N_f, T_f, startVals_f, edgePos, dst);
    }
}

bool project_Joseph_modular(float*& g, float* f, parameters* params, bool data_on_cpu)
{
    return project_Joseph_modular(g, f, params, data_on_cpu, data_on_cpu);
}

bool backproject_Joseph_modular(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    return backproject_Joseph_modular(g, f, params, data_on_cpu, data_on_cpu);
}

bool project_Joseph_modular(float*& g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);
    
    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

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

    bool useSF = true;
    if (params->voxelSizeWorksForFastSF() == false)
    {
        if (params->voxelWidth < params->default_voxelWidth() || params->voxelHeight < params->default_voxelHeight())
            useSF = false;
    }

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->modularbeamIsAxiallyAligned() == true && useSF == true)
    {
        //printf("s = %f, %f, %f\n", params->sourcePositions[0], params->sourcePositions[1], params->sourcePositions[2]);
        //printf("c = %f, %f, %f\n", params->moduleCenters[0], params->moduleCenters[1], params->moduleCenters[2]);
        //printf("u = %f, %f, %f\n", params->colVectors[0], params->colVectors[1], params->colVectors[2]);
        //printf("v = %f, %f, %f\n", params->rowVectors[0], params->rowVectors[1], params->rowVectors[2]);

        float rFOV_sq = params->rFOV() * params->rFOV();
        modularBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
        //applyViewDependentPolarWeights_gpu(dev_g, params, NULL, true, false);
        //applyViewDependentPolarWeights_gpu(dev_g, params, NULL, false, false); // was this
    }
    else
    {
        modularBeamJosephProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, accum);
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

    return true;
}

bool backproject_Joseph_modular(float* g, float*& f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accum)
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

    double alpha = max(atan(0.5 * (params->numCols - 1) * params->pixelWidth / params->sdd), atan(0.5 * (params->numRows - 1) * params->pixelHeight / params->sdd));
    double rFOV = max(params->furthestFromCenter(), max(fabs(params->z_0()), fabs(params->z_samples(params->numZ - 1))));
    double maxDivergence = tan(alpha) * (params->sdd + rFOV) - tan(alpha) * (params->sdd - rFOV);
    //double maxTravel = maxDivergence / min(params->pixelWidth, params->pixelHeight);
    double maxTravel = maxDivergence / params->voxelWidth;
    //printf("maxTravel = %f\n", maxTravel);
    bool isParallel = false;
    if (maxTravel < 0.25 && params->truncatedScan == false)
        isParallel = true;

    bool doLinearInterpolation = false;

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);
    float* w_polar = NULL;
    bool modularbeamIsAxiallyAligned = params->modularbeamIsAxiallyAligned();
    if (isParallel)
        modularbeamIsAxiallyAligned = false;
    if (modularbeamIsAxiallyAligned == true)
        doLinearInterpolation = true;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = NULL;

    /*
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;
    if (modularbeamIsAxiallyAligned == true)
    {
        w_polar = setViewDependentPolarWeights(params);
        //applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, true, false);
        applyViewDependentPolarWeights_gpu(dev_g, params, w_polar, false, false);
    }
    d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, doLinearInterpolation);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        dev_g = 0;
    }
    //*/

    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, N_g, params->doExtrapolation, doLinearInterpolation);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, doLinearInterpolation);

    if (volume_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
        }
    }
    else
        dev_f = f;

    float rFOV_sq = params->rFOV() * params->rFOV();

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    int4 N_f_mod = make_int4(N_f.x, N_f.y, int(ceil(float(N_f.z) / float(NUM_SLICES_PER_THREAD))), N_f.w);
    dim3 dimBlock_slab = setBlockSize(N_f_mod);
    dim3 dimGrid_slab = setGridSize(N_f_mod, dimBlock_slab);

    if (isParallel)
    {
        //printf("executing parallel Joseph backprojector\n");
        modularBeamParallelJosephBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
    }
    else if (modularbeamIsAxiallyAligned == true)
    {
        //printf("SF backproject\n");
        if (params->voxelSizeWorksForFastSF() == true)
        {
            //modularBeamBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
            modularBeamBackprojectorKernel_SF_stack <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
        }
        else
        {
            //modularBeamBackprojectorKernel_eSF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
            modularBeamBackprojectorKernel_eSF_stack <<< dimGrid_slab, dimBlock_slab >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
        }
    }
    else
    {
        //printf("executing cone Joseph backprojector\n");
        modularBeamJosephBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder, rFOV_sq, accum);
    }

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
