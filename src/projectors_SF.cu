////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "projectors.h"
#include "projectors_SF.h"
#include "cuda_utils.h"

#include "projectors_extendedSF.h"

__global__ void parallelBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;
        
    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    
    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k;
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i;

    if (x*x + y*y > rFOVsq)
    {
        f[ind] = 0.0;
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
    float cos_phi, sin_phi, C, s_arg, ds;

    float val = 0.0;
    // loop over projection angles
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
}

__global__ void parallelBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
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

            if (volumeDimensionOrder == 0)
            {
                g_output += max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, m, j_min_A, i)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C)) * tex3D<float>(f, m, j_min_A + 1, i)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, m, j_min_A + 2, i);
            }
            else
            {
                g_output += max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, i, j_min_A, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C)) * tex3D<float>(f, i, j_min_A + 1, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, i, j_min_A + 2, m);
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

            if (volumeDimensionOrder == 0)
            {
                g_output += max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, m, j, i_min_A)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C)) * tex3D<float>(f, m, j, i_min_A + 1)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, m, j, i_min_A + 2);
            }
            else
            {
                g_output += max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, i_min_A, j, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C)) * tex3D<float>(f, i_min_A + 1, j, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, i_min_A + 2, j, m);
            }
        }
    }
    //g[l * N_g.z * N_g.y + m * N_g.y + n] = l_phi * g_output;
    g[l * N_g.z * N_g.y + m * N_g.z + n] = l_phi * g_output;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fanBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k;
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    float val = 0.0;
    //float sin_phi, cos_phi;
    const float T_x_over_2 = 0.5f * T_f.x;

    float tau_low, tau_high;
    float ind_first;
    float ind_last;
    float u_arg;
    float A_x;
    float B_x, B_y;
    float x_dot_theta_perp;
    float R_minus_x_dot_theta, R_minus_x_dot_theta_inv;
    float x_denom, y_denom;
    float l_phi;
    //float horizontalWeights[2];
    //float v_phi_x_step_A, v_phi_x_step_B, v_phi_x_step_C, v_phi_x_step_D;
    float v_phi_x;
    float z_high_A, z_high_B, z_high_C, z_high_D;
    float row_high_A, row_high_B, row_high_C, row_high_D;
    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;
    float sin_phi, cos_phi;

    //iv* T_g.y + startVals_g.y = T_f.z*k + startVals_f.z = z
    // (z - startVals_g.y) / T_g.y
    const float iv = (z - startVals_g.y) / T_g.y + 0.5f;

    for (int l = 0; l < N_g.x; l++)
    {
        if (l + 3 < N_g.x)
        {
            const float L = (float)l+0.5f;
            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg * cos_phi - sin_phi);
            y_denom = fabs(u_arg * sin_phi + cos_phi);
            l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            //v_phi_x = v_phi_x_start_num * R_minus_x_dot_theta_inv - v0_over_Tv;
            //const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            //row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            //z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float s_oneAndtwo_A = ind_first;
            const float s_three_A = ind_last;

            const float L1 = L + 1.0f;
            sin_phi = sin(phis[l + 1]);
            cos_phi = cos(phis[l + 1]);

            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg * cos_phi - sin_phi);
            y_denom = fabs(u_arg * sin_phi + cos_phi);
            l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_B = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_B = l_phi * (tau_high - tau_low) - horizontalWeights_0_B;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_B;

            //v_phi_x = v_phi_x_start_num * R_minus_x_dot_theta_inv - v0_over_Tv;
            //const float v_phi_x_step_B = Tz_over_Tv * R_minus_x_dot_theta_inv;

            //row_high_B = floor(v_phi_x - 0.5f * v_phi_x_step_B + 0.5f) + 0.5f;
            //z_high_B = v_phi_x + 0.5f * v_phi_x_step_B - row_high_B;

            const float s_oneAndtwo_B = ind_first;
            const float s_three_B = ind_last;

            const float L2 = L + 2.0f;
            sin_phi = sin(phis[l + 2]);
            cos_phi = cos(phis[l + 2]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg * cos_phi - sin_phi);
            y_denom = fabs(u_arg * sin_phi + cos_phi);
            l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_C = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_C = l_phi * (tau_high - tau_low) - horizontalWeights_0_C;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_C;

            //v_phi_x = v_phi_x_start_num * R_minus_x_dot_theta_inv - v0_over_Tv;
            //const float v_phi_x_step_C = Tz_over_Tv * R_minus_x_dot_theta_inv;

            //row_high_C = floor(v_phi_x - 0.5f * v_phi_x_step_C + 0.5f) + 0.5f;
            //z_high_C = v_phi_x + 0.5f * v_phi_x_step_C - row_high_C;

            const float s_oneAndtwo_C = ind_first;
            const float s_three_C = ind_last;

            const float L3 = L + 3.0f;
            sin_phi = sin(phis[l + 3]);
            cos_phi = cos(phis[l + 3]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg * cos_phi - sin_phi);
            y_denom = fabs(u_arg * sin_phi + cos_phi);
            l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_D = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_D = l_phi * (tau_high - tau_low) - horizontalWeights_0_D;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_D;

            //v_phi_x = v_phi_x_start_num * R_minus_x_dot_theta_inv - v0_over_Tv;
            //const float v_phi_x_step_D = Tz_over_Tv * R_minus_x_dot_theta_inv;

            //row_high_D = floor(v_phi_x - 0.5f * v_phi_x_step_D + 0.5f) + 0.5f;
            //z_high_D = v_phi_x + 0.5f * v_phi_x_step_D - row_high_D;

            const float s_oneAndtwo_D = ind_first;
            const float s_three_D = ind_last;

            /*
            const float row_high_plus_one_A = row_high_A + 1.0f;
            const float row_high_plus_two_A = row_high_A + 2.0f;
            const float row_high_plus_one_B = row_high_B + 1.0f;
            const float row_high_plus_two_B = row_high_B + 2.0f;
            const float row_high_plus_one_C = row_high_C + 1.0f;
            const float row_high_plus_two_C = row_high_C + 2.0f;
            const float row_high_plus_one_D = row_high_D + 1.0f;
            const float row_high_plus_two_D = row_high_D + 2.0f;
            //*/

            val += (tex3D<float>(g, s_oneAndtwo_A, iv, L) * horizontalWeights_0_A
                +   tex3D<float>(g, s_three_A, iv, L) * horizontalWeights_1_A)
                +  (tex3D<float>(g, s_oneAndtwo_B, iv, L1) * horizontalWeights_0_B
                  + tex3D<float>(g, s_three_B, iv, L1) * horizontalWeights_1_B)
                +  (tex3D<float>(g, s_oneAndtwo_C, iv, L2) * horizontalWeights_0_C
                  + tex3D<float>(g, s_three_C, iv, L2) * horizontalWeights_1_C)
                +  (tex3D<float>(g, s_oneAndtwo_D, iv, L3) * horizontalWeights_0_D
                  + tex3D<float>(g, s_three_D, iv, L3) * horizontalWeights_1_D);

            /*
            val += (tex3D<float>(g, s_oneAndtwo_A, row_high_A, L) * horizontalWeights_0_A
                + tex3D<float>(g, s_three_A, row_high_A, L) * horizontalWeights_1_A) * min(v_phi_x_step_A, v_phi_x_step_A - z_high_A)
                + (tex3D<float>(g, s_oneAndtwo_A, row_high_plus_one_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, s_three_A, row_high_plus_one_A, L) * horizontalWeights_1_A) * max(0.0f, min(z_high_A, 1.0f))
                + (tex3D<float>(g, s_oneAndtwo_A, row_high_plus_two_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, s_three_A, row_high_plus_two_A, L) * horizontalWeights_1_A) * max(0.0f, z_high_A - 1.0f)

                + (tex3D<float>(g, s_oneAndtwo_B, row_high_B, L1) * horizontalWeights_0_B
                    + tex3D<float>(g, s_three_B, row_high_B, L1) * horizontalWeights_1_B) * min(v_phi_x_step_B, v_phi_x_step_B - z_high_B)
                + (tex3D<float>(g, s_oneAndtwo_B, row_high_plus_one_B, L1) * horizontalWeights_0_B
                    + tex3D<float>(g, s_three_B, row_high_plus_one_B, L1) * horizontalWeights_1_B) * max(0.0f, min(z_high_B, 1.0f))
                + (tex3D<float>(g, s_oneAndtwo_B, row_high_plus_two_B, L1) * horizontalWeights_0_B
                    + tex3D<float>(g, s_three_B, row_high_plus_two_B, L1) * horizontalWeights_1_B) * max(0.0f, z_high_B - 1.0f)
                + (tex3D<float>(g, s_oneAndtwo_C, row_high_C, L2) * horizontalWeights_0_C
                    + tex3D<float>(g, s_three_C, row_high_C, L2) * horizontalWeights_1_C) * min(v_phi_x_step_C, v_phi_x_step_C - z_high_C)
                + (tex3D<float>(g, s_oneAndtwo_C, row_high_plus_one_C, L2) * horizontalWeights_0_C
                    + tex3D<float>(g, s_three_C, row_high_plus_one_C, L2) * horizontalWeights_1_C) * max(0.0f, min(z_high_C, 1.0f))
                + (tex3D<float>(g, s_oneAndtwo_C, row_high_plus_two_C, L2) * horizontalWeights_0_C
                    + tex3D<float>(g, s_three_C, row_high_plus_two_C, L2) * horizontalWeights_1_C) * max(0.0f, z_high_C - 1.0f)
                + (tex3D<float>(g, s_oneAndtwo_D, row_high_D, L3) * horizontalWeights_0_D
                    + tex3D<float>(g, s_three_D, row_high_D, L3) * horizontalWeights_1_D) * min(v_phi_x_step_D, v_phi_x_step_D - z_high_D)
                + (tex3D<float>(g, s_oneAndtwo_D, row_high_plus_one_D, L3) * horizontalWeights_0_D
                    + tex3D<float>(g, s_three_D, row_high_plus_one_D, L3) * horizontalWeights_1_D) * max(0.0f, min(z_high_D, 1.0f))
                + (tex3D<float>(g, s_oneAndtwo_D, row_high_plus_two_D, L3) * horizontalWeights_0_D
                    + tex3D<float>(g, s_three_D, row_high_plus_two_D, L3) * horizontalWeights_1_D) * max(0.0f, z_high_D - 1.0f);
            //*/

            l += 3;
        }
        else
        {
            const float L = (float)l+0.5f;
            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
            R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg * cos_phi - sin_phi);
            y_denom = fabs(u_arg * sin_phi + cos_phi);
            l_phi = T_f.x * sqrt(1.0f + u_arg * u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi) * T_x_over_2;
            else
            {
                A_x = fabs(cos_phi) * T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low + 0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first + 1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi * (tau_high - tau_low) - horizontalWeights_0_A;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first + 0.5f + max(0.0f, min(tau_high - ind_first - 0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            //v_phi_x = v_phi_x_start_num * R_minus_x_dot_theta_inv - v0_over_Tv;
            //const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            //row_high_A = floor(v_phi_x - 0.5f * v_phi_x_step_A + 0.5f) + 0.5f;
            //z_high_A = v_phi_x + 0.5f * v_phi_x_step_A - row_high_A;

            const float s_oneAndtwo_A = ind_first;
            const float s_three_A = ind_last;

            val += (tex3D<float>(g, s_oneAndtwo_A, iv, L) * horizontalWeights_0_A
                + tex3D<float>(g, s_three_A, iv, L) * horizontalWeights_1_A);

            /*
            const float row_high_plus_one_A = row_high_A + 1.0f;
            const float row_high_plus_two_A = row_high_A + 2.0f;
            val += (tex3D<float>(g, s_oneAndtwo_A, row_high_A, L) * horizontalWeights_0_A
                + tex3D<float>(g, s_three_A, row_high_A, L) * horizontalWeights_1_A) * min(v_phi_x_step_A, v_phi_x_step_A - z_high_A)
                + (tex3D<float>(g, s_oneAndtwo_A, row_high_plus_one_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, s_three_A, row_high_plus_one_A, L) * horizontalWeights_1_A) * max(0.0f, min(z_high_A, 1.0f))
                + (tex3D<float>(g, s_oneAndtwo_A, row_high_plus_two_A, L) * horizontalWeights_0_A
                    + tex3D<float>(g, s_three_A, row_high_plus_two_A, L) * horizontalWeights_1_A) * max(0.0f, z_high_A - 1.0f);
            //*/
        }
    }

    f[ind] = val;
}

__global__ void fanBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
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
    const float m_minus_half = (float)m - 0.5f;
    const float m_plus_half = (float)m + 0.5f;

    //const float v0_over_Tv = startVals_g.y / T_g.y;

    const int iz = int(floor(0.5 + (v - startVals_f.z) / T_f.z));

    //const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
    //const float z_ind_offset = -z0_over_Tz_plus_half + 0.0f * ((float)l * T_g.x + startVals_g.x);
    //const float z_ind_slope = (v - 0.5f * T_g.y) / T_f.z;

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
            /*
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
            //*/

            const float num_low = tau - x * sin_phi + y * cos_phi - A_x;
            const float num_high = num_low + 2.0f * A_x;

            const float denom_low = (R_minus_x_dot_theta - B_x) * T_g.z;
            const float denom_high = (R_minus_x_dot_theta + B_x) * T_g.z;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high - Tx_sin) / (denom_high - Tx_cos), n_plus_half) - max((num_low - Tx_sin) / (denom_low - Tx_cos), n_minus_half));
            const float hWeight_2 = 1.0f - hWeight_1 - hWeight_0;

            //const float v_phi_x_step = T_f.z / (T_g.y * R_minus_x_dot_theta);
            //const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            if (volumeDimensionOrder == 0)
            {
                g_output += tex3D<float>(f, iz, j, i) * hWeight_0
                    + tex3D<float>(f, iz, j, i + 1) * hWeight_1
                    + tex3D<float>(f, iz, j, i + 2) * hWeight_2;

                /*
                g_output += (tex3D<float>(f, k, j, i) * hWeight_0
                    + tex3D<float>(f, k, j, i + 1) * hWeight_1
                    + tex3D<float>(f, k, j, i + 2) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, k + 1, j, i) * hWeight_0
                        + tex3D<float>(f, k + 1, j, i + 1) * hWeight_1
                        + tex3D<float>(f, k + 1, j, i + 2) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, k + 2, j, i) * hWeight_0
                        + tex3D<float>(f, k + 2, j, i + 1) * hWeight_1
                        + tex3D<float>(f, k + 2, j, i + 2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
                //*/
            }
            else
            {
                g_output += tex3D<float>(f, i, j, iz) * hWeight_0
                    + tex3D<float>(f, i + 1, j, iz) * hWeight_1
                    + tex3D<float>(f, i + 2, j, iz) * hWeight_2;
                /*
                g_output += (tex3D<float>(f, i, j, k) * hWeight_0
                    + tex3D<float>(f, i + 1, j, k) * hWeight_1
                    + tex3D<float>(f, i + 2, j, k) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, i, j, k + 1) * hWeight_0
                        + tex3D<float>(f, i + 1, j, k + 1) * hWeight_1
                        + tex3D<float>(f, i + 2, j, k + 1) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, i, j, k + 2) * hWeight_0
                        + tex3D<float>(f, i + 1, j, k + 2) * hWeight_1
                        + tex3D<float>(f, i + 2, j, k + 2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
                //*/
            }
        }
        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
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
            /*
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
            //*/

            const float num_low = tau - x * sin_phi + y * cos_phi - A_y;
            const float num_high = num_low + 2.0f * A_y;

            const float denom_low = (R_minus_x_dot_theta - B_y) * T_g.z;
            const float denom_high = (R_minus_x_dot_theta + B_y) * T_g.z;

            const float hWeight_0 = max(0.0f, min(num_high / denom_high, n_plus_half) - max(num_low / denom_low, n_minus_half));
            const float hWeight_1 = max(0.0f, min((num_high + Ty_cos) / (denom_high - Ty_sin), n_plus_half) - max((num_low + Ty_cos) / (denom_low - Ty_sin), n_minus_half));
            const float hWeight_2 = 1.0f - hWeight_1 - hWeight_0;

            //const float v_phi_x_step = T_f.z / (T_g.y * R_minus_x_dot_theta);
            //const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

            if (volumeDimensionOrder == 0)
            {
                g_output += tex3D<float>(f, iz, j, i) * hWeight_0
                    + tex3D<float>(f, iz, j + 1, i) * hWeight_1
                    + tex3D<float>(f, iz, j + 2, i) * hWeight_2;
                /*
                g_output += (tex3D<float>(f, k, j, i) * hWeight_0
                    + tex3D<float>(f, k, j + 1, i) * hWeight_1
                    + tex3D<float>(f, k, j + 2, i) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, k + 1, j, i) * hWeight_0
                        + tex3D<float>(f, k + 1, j + 1, i) * hWeight_1
                        + tex3D<float>(f, k + 1, j + 2, i) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, k + 2, j, i) * hWeight_0
                        + tex3D<float>(f, k + 2, j + 1, i) * hWeight_1
                        + tex3D<float>(f, k + 2, j + 2, i) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
                //*/
            }
            else
            {
                g_output += tex3D<float>(f, i, j, iz) * hWeight_0
                    + tex3D<float>(f, i, j + 1, iz) * hWeight_1
                    + tex3D<float>(f, i, j + 2, iz) * hWeight_2;
                /*
                g_output += (tex3D<float>(f, i, j, k) * hWeight_0
                    + tex3D<float>(f, i, j + 1, k) * hWeight_1
                    + tex3D<float>(f, i, j + 2, k) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, i, j, k + 1) * hWeight_0
                        + tex3D<float>(f, i, j + 1, k + 1) * hWeight_1
                        + tex3D<float>(f, i, j + 2, k + 1) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                    + (tex3D<float>(f, i, j, k + 2) * hWeight_0
                        + tex3D<float>(f, i, j + 1, k + 2) * hWeight_1
                        + tex3D<float>(f, i, j + 2, k + 2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
                //*/
            }
        }
        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
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
    g[i * N_g.z * N_g.y + j * N_g.z + k] *= rsqrt(1.0f + v*v);
}

__global__ void applyInversePolarWeight(float* g, int4 N_g, float4 T_g, float4 startVals_g)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    const float v = j * T_g.y + startVals_g.y;
    g[i * N_g.z * N_g.y + j * N_g.z + k] *= sqrt(1.0f + v * v);
}

__global__ void coneBeamBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k; // XYZ
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i; // ZYX
        
    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x*x + y*y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }
    
    float val = 0.0;
    //float sin_phi, cos_phi;
    const float T_x_over_2 = 0.5f * T_f.x;

    float tau_low, tau_high;
    float ind_first;
    float ind_last;
    float u_arg;
    float A_x;
    float B_x, B_y;
    float x_dot_theta_perp;
    float R_minus_x_dot_theta, R_minus_x_dot_theta_inv;
    float x_denom, y_denom;
    float l_phi;
    //float horizontalWeights[2];
    //float v_phi_x_step_A, v_phi_x_step_B, v_phi_x_step_C, v_phi_x_step_D;
    float v_phi_x;
    float z_high_A, z_high_B, z_high_C, z_high_D;
    float row_high_A, row_high_B, row_high_C, row_high_D;
    const float v0_over_Tv = startVals_g.y / T_g.y;
    const float Tz_over_Tv = T_f.z / T_g.y;
    const float v_phi_x_start_num = z / T_g.y;
    const float Tu_inv = 1.0f/T_g.z;
    float sin_phi, cos_phi;

    for (int l = 0; l < N_g.x; l++)
    {
        if (l+3 < N_g.x)
        {
            const float L = (float)l+0.5f;
            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi*y - sin_phi*x + tau;
            R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg*cos_phi-sin_phi);
            y_denom = fabs(u_arg*sin_phi+cos_phi);
            l_phi = T_f.x * sqrt(1.0f+u_arg*u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi)*T_x_over_2;
            else
            {
                A_x = fabs(cos_phi)*T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;
            //tau_low = ((x_dot_theta_perp - A_x) * R_minus_x_dot_theta_inv*(1.0f+B_x*R_minus_x_dot_theta_inv) - startVals_g.z) * Tu_inv;
            //tau_high = ((x_dot_theta_perp + A_x) * R_minus_x_dot_theta_inv*(1.0f-B_x*R_minus_x_dot_theta_inv) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low+0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first+1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi*(tau_high-tau_low) - horizontalWeights_0_A;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first+0.5f + max(0.0f, min(tau_high-ind_first-0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            v_phi_x = v_phi_x_start_num*R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            row_high_A = floor(v_phi_x - 0.5f*v_phi_x_step_A + 0.5f) + 0.5f;
            z_high_A = v_phi_x + 0.5f*v_phi_x_step_A - row_high_A;

            const float s_oneAndtwo_A = ind_first;
            const float s_three_A = ind_last;

            const float L1 = L+1.0f;
            sin_phi = sin(phis[l+1]);
            cos_phi = cos(phis[l+1]);

            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi*y - sin_phi*x + tau;
            R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg*cos_phi-sin_phi);
            y_denom = fabs(u_arg*sin_phi+cos_phi);
            l_phi = T_f.x * sqrt(1.0f+u_arg*u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi)*T_x_over_2;
            else
            {
                A_x = fabs(cos_phi)*T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low+0.5f); // first detector index

            const float horizontalWeights_0_B = (min(tau_high, ind_first+1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_B = l_phi*(tau_high-tau_low) - horizontalWeights_0_B;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first+0.5f + max(0.0f, min(tau_high-ind_first-0.5f, 1.0f)) * l_phi / horizontalWeights_0_B;

            v_phi_x = v_phi_x_start_num*R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_B = Tz_over_Tv * R_minus_x_dot_theta_inv;

            row_high_B = floor(v_phi_x - 0.5f*v_phi_x_step_B + 0.5f) + 0.5f;
            z_high_B = v_phi_x + 0.5f*v_phi_x_step_B - row_high_B;

            const float s_oneAndtwo_B = ind_first;
            const float s_three_B = ind_last;

            const float L2 = L+2.0f;
            sin_phi = sin(phis[l+2]);
            cos_phi = cos(phis[l+2]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi*y - sin_phi*x + tau;
            R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg*cos_phi-sin_phi);
            y_denom = fabs(u_arg*sin_phi+cos_phi);
            l_phi = T_f.x * sqrt(1.0f+u_arg*u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi)*T_x_over_2;
            else
            {
                A_x = fabs(cos_phi)*T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low+0.5f); // first detector index

            const float horizontalWeights_0_C = (min(tau_high, ind_first+1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_C = l_phi*(tau_high-tau_low) - horizontalWeights_0_C;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first+0.5f + max(0.0f, min(tau_high-ind_first-0.5f, 1.0f)) * l_phi / horizontalWeights_0_C;

            v_phi_x = v_phi_x_start_num*R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_C = Tz_over_Tv * R_minus_x_dot_theta_inv;

            row_high_C = floor(v_phi_x - 0.5f*v_phi_x_step_C + 0.5f) + 0.5f;
            z_high_C = v_phi_x + 0.5f*v_phi_x_step_C - row_high_C;

            const float s_oneAndtwo_C = ind_first;
            const float s_three_C = ind_last;

            const float L3 = L+3.0f;
            sin_phi = sin(phis[l+3]);
            cos_phi = cos(phis[l+3]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi*y - sin_phi*x + tau;
            R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg*cos_phi-sin_phi);
            y_denom = fabs(u_arg*sin_phi+cos_phi);
            l_phi = T_f.x * sqrt(1.0f+u_arg*u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi)*T_x_over_2;
            else
            {
                A_x = fabs(cos_phi)*T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low+0.5f); // first detector index

            const float horizontalWeights_0_D = (min(tau_high, ind_first+1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_D = l_phi*(tau_high-tau_low) - horizontalWeights_0_D;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first+0.5f + max(0.0f, min(tau_high-ind_first-0.5f, 1.0f)) * l_phi / horizontalWeights_0_D;

            v_phi_x = v_phi_x_start_num*R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_D = Tz_over_Tv * R_minus_x_dot_theta_inv;

            row_high_D = floor(v_phi_x - 0.5f*v_phi_x_step_D + 0.5f) + 0.5f;
            z_high_D = v_phi_x + 0.5f*v_phi_x_step_D - row_high_D;

            const float s_oneAndtwo_D = ind_first;
            const float s_three_D = ind_last;

            const float row_high_plus_one_A = row_high_A+1.0f;
            const float row_high_plus_two_A = row_high_A+2.0f;
            const float row_high_plus_one_B = row_high_B+1.0f;
            const float row_high_plus_two_B = row_high_B+2.0f;
            const float row_high_plus_one_C = row_high_C+1.0f;
            const float row_high_plus_two_C = row_high_C+2.0f;
            const float row_high_plus_one_D = row_high_D+1.0f;
            const float row_high_plus_two_D = row_high_D+2.0f;
            val += (tex3D<float>(g,s_oneAndtwo_A, row_high_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_A, L) * horizontalWeights_1_A)*min(v_phi_x_step_A, v_phi_x_step_A-z_high_A)
                +  (tex3D<float>(g,s_oneAndtwo_A, row_high_plus_one_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_plus_one_A, L) * horizontalWeights_1_A)*max(0.0f, min(z_high_A, 1.0f))
                +  (tex3D<float>(g,s_oneAndtwo_A, row_high_plus_two_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_plus_two_A, L) * horizontalWeights_1_A)*max(0.0f, z_high_A-1.0f)
                +  (tex3D<float>(g,s_oneAndtwo_B, row_high_B, L1) * horizontalWeights_0_B
                +   tex3D<float>(g,s_three_B, row_high_B, L1) * horizontalWeights_1_B)*min(v_phi_x_step_B, v_phi_x_step_B-z_high_B)
                +  (tex3D<float>(g,s_oneAndtwo_B, row_high_plus_one_B, L1) * horizontalWeights_0_B
                +   tex3D<float>(g,s_three_B, row_high_plus_one_B, L1) * horizontalWeights_1_B)*max(0.0f, min(z_high_B, 1.0f))
                +  (tex3D<float>(g,s_oneAndtwo_B, row_high_plus_two_B, L1) * horizontalWeights_0_B
                +   tex3D<float>(g,s_three_B, row_high_plus_two_B, L1) * horizontalWeights_1_B)*max(0.0f, z_high_B-1.0f)
                +  (tex3D<float>(g,s_oneAndtwo_C, row_high_C, L2) * horizontalWeights_0_C
                +   tex3D<float>(g,s_three_C, row_high_C, L2) * horizontalWeights_1_C)*min(v_phi_x_step_C, v_phi_x_step_C-z_high_C)
                +  (tex3D<float>(g,s_oneAndtwo_C, row_high_plus_one_C, L2) * horizontalWeights_0_C
                +   tex3D<float>(g,s_three_C, row_high_plus_one_C, L2) * horizontalWeights_1_C)*max(0.0f, min(z_high_C, 1.0f))
                +  (tex3D<float>(g,s_oneAndtwo_C, row_high_plus_two_C, L2) * horizontalWeights_0_C
                +   tex3D<float>(g,s_three_C, row_high_plus_two_C, L2) * horizontalWeights_1_C)*max(0.0f, z_high_C-1.0f)
                +  (tex3D<float>(g,s_oneAndtwo_D, row_high_D, L3) * horizontalWeights_0_D
                +   tex3D<float>(g,s_three_D, row_high_D, L3) * horizontalWeights_1_D)*min(v_phi_x_step_D, v_phi_x_step_D-z_high_D)
                +  (tex3D<float>(g,s_oneAndtwo_D, row_high_plus_one_D, L3) * horizontalWeights_0_D
                +   tex3D<float>(g,s_three_D, row_high_plus_one_D, L3) * horizontalWeights_1_D)*max(0.0f, min(z_high_D, 1.0f))
                +  (tex3D<float>(g,s_oneAndtwo_D, row_high_plus_two_D, L3) * horizontalWeights_0_D
                +   tex3D<float>(g,s_three_D, row_high_plus_two_D, L3) * horizontalWeights_1_D)*max(0.0f, z_high_D-1.0f);

            l+=3;
        }
        else
        {
            const float L = (float)l+0.5f;
            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            if (sin_phi < 0.0f)
                B_x = -cos_phi;
            else
                B_x = cos_phi;
            B_x *= T_x_over_2;
            if (cos_phi < 0.0f)
                B_y = sin_phi;
            else
                B_y = -sin_phi;
            B_y *= T_x_over_2;

            x_dot_theta_perp = cos_phi*y - sin_phi*x + tau;
            R_minus_x_dot_theta = R - x*cos_phi - y*sin_phi;
            R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

            u_arg = x_dot_theta_perp * R_minus_x_dot_theta_inv;
            x_denom = fabs(u_arg*cos_phi-sin_phi);
            y_denom = fabs(u_arg*sin_phi+cos_phi);
            l_phi = T_f.x * sqrt(1.0f+u_arg*u_arg) / max(x_denom, y_denom);
            if (x_denom > y_denom)
                A_x = fabs(sin_phi)*T_x_over_2;
            else
            {
                A_x = fabs(cos_phi)*T_x_over_2;
                B_x = B_y;
            }
            tau_low = ((x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x) - startVals_g.z) * Tu_inv;
            tau_high = ((x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x) - startVals_g.z) * Tu_inv;

            ind_first = floor(tau_low+0.5f); // first detector index

            const float horizontalWeights_0_A = (min(tau_high, ind_first+1.5f) - tau_low) * l_phi;
            const float horizontalWeights_1_A = l_phi*(tau_high-tau_low) - horizontalWeights_0_A;

            ind_last = ind_first + 2.5f;
            ind_first = ind_first+0.5f + max(0.0f, min(tau_high-ind_first-0.5f, 1.0f)) * l_phi / horizontalWeights_0_A;

            v_phi_x = v_phi_x_start_num*R_minus_x_dot_theta_inv - v0_over_Tv;
            const float v_phi_x_step_A = Tz_over_Tv * R_minus_x_dot_theta_inv;

            row_high_A = floor(v_phi_x - 0.5f*v_phi_x_step_A + 0.5f) + 0.5f;
            z_high_A = v_phi_x + 0.5f*v_phi_x_step_A - row_high_A;

            const float s_oneAndtwo_A = ind_first;
            const float s_three_A = ind_last;

            const float row_high_plus_one_A = row_high_A+1.0f;
            const float row_high_plus_two_A = row_high_A+2.0f;
            val += (tex3D<float>(g,s_oneAndtwo_A, row_high_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_A, L) * horizontalWeights_1_A)*min(v_phi_x_step_A, v_phi_x_step_A-z_high_A)
                +  (tex3D<float>(g,s_oneAndtwo_A, row_high_plus_one_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_plus_one_A, L) * horizontalWeights_1_A)*max(0.0f, min(z_high_A, 1.0f))
                +  (tex3D<float>(g,s_oneAndtwo_A, row_high_plus_two_A, L) * horizontalWeights_0_A
                +   tex3D<float>(g,s_three_A, row_high_plus_two_A, L) * horizontalWeights_1_A)*max(0.0f, z_high_A-1.0f);
        }
    }
    
    f[ind] = val;
}

__global__ void coneBeamProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
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
     const float m_minus_half = (float)m - 0.5f;
     const float m_plus_half = (float)m + 0.5f;

     const float v0_over_Tv = startVals_g.y / T_g.y;

     const float z0_over_Tz_plus_half = startVals_f.z / T_f.z + 0.5f;
     const float z_ind_offset = -z0_over_Tz_plus_half + 0.0f*((float)l * T_g.x + startVals_g.x);

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
             const float hWeight_2 = 1.0f - hWeight_1 - hWeight_0;

             const float v_phi_x_step = T_f.z / (T_g.y*R_minus_x_dot_theta);
             const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

             if (volumeDimensionOrder == 0)
             {
                 g_output += (tex3D<float>(f, k, j, i) * hWeight_0
                     + tex3D<float>(f, k, j, i + 1) * hWeight_1
                     + tex3D<float>(f, k, j, i + 2) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, k + 1, j, i) * hWeight_0
                         + tex3D<float>(f, k + 1, j, i + 1) * hWeight_1
                         + tex3D<float>(f, k + 1, j, i + 2) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, k + 2, j, i) * hWeight_0
                         + tex3D<float>(f, k + 2, j, i + 1) * hWeight_1
                         + tex3D<float>(f, k + 2, j, i + 2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             }
             else
             {
                 g_output += (tex3D<float>(f, i, j, k) * hWeight_0
                     + tex3D<float>(f, i+1, j, k) * hWeight_1
                     + tex3D<float>(f, i+2, j, k) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, i, j, k + 1) * hWeight_0
                         + tex3D<float>(f, i + 1, j, k+1) * hWeight_1
                         + tex3D<float>(f, i + 2, j, k+1) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, i, j, k+2) * hWeight_0
                         + tex3D<float>(f, i + 1, j, k + 2) * hWeight_1
                         + tex3D<float>(f, i + 2, j, k + 2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             }
         }
         g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f+u*u) / fabs(u*cos_phi-sin_phi) * g_output;
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
             const float hWeight_2 = 1.0f - hWeight_1 - hWeight_0;

             const float v_phi_x_step = T_f.z / (T_g.y*R_minus_x_dot_theta);
             const float xi_high = ((float)k - z_ind_offset) * v_phi_x_step - v0_over_Tv;

             if (volumeDimensionOrder == 0)
             {
                 g_output += (tex3D<float>(f, k, j, i) * hWeight_0
                     + tex3D<float>(f, k, j + 1, i) * hWeight_1
                     + tex3D<float>(f, k, j + 2, i) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, k + 1, j, i) * hWeight_0
                         + tex3D<float>(f, k + 1, j + 1, i) * hWeight_1
                         + tex3D<float>(f, k + 1, j + 2, i) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, k + 2, j, i) * hWeight_0
                         + tex3D<float>(f, k + 2, j + 1, i) * hWeight_1
                         + tex3D<float>(f, k + 2, j + 2, i) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             }
             else
             {
                 g_output += (tex3D<float>(f, i, j, k) * hWeight_0
                     + tex3D<float>(f, i, j + 1, k) * hWeight_1
                     + tex3D<float>(f, i, j + 2, k) * hWeight_2) * (min(xi_high - m_minus_half, 1.0f)) * ((k >= 0) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, i, j, k+1) * hWeight_0
                         + tex3D<float>(f, i, j + 1, k+1) * hWeight_1
                         + tex3D<float>(f, i, j + 2, k+1) * hWeight_2) * max(0.0f, min(v_phi_x_step, m_plus_half - xi_high)) * ((k >= -1 && k + 1 < N_f.z) ? 1.0f : 0.0f)
                     + (tex3D<float>(f, i, j, k+2) * hWeight_0
                         + tex3D<float>(f, i, j + 1, k+2) * hWeight_1
                         + tex3D<float>(f, i, j + 2, k+2) * hWeight_2) * max(0.0f, min(m_plus_half - xi_high - v_phi_x_step, 1.0f)) * ((k + 2 < N_f.z) ? 1.0f : 0.0f);
             }
         }
         g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f+u*u) / fabs(u*sin_phi+cos_phi) * g_output;
     }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool project_SF_fan(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_SF(g, f, params, cpu_to_gpu);
}

bool backproject_SF_fan(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_SF(g, f, params, cpu_to_gpu);
}

bool project_SF_parallel(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_SF(g, f, params, cpu_to_gpu);
}

bool backproject_SF_parallel(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_SF(g, f, params, cpu_to_gpu);
}

bool project_SF_cone(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_SF(g, f, params, cpu_to_gpu);
}

bool backproject_SF_cone(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_SF(g, f, params, cpu_to_gpu);
}

bool project_SF(float *&g, float *f, parameters* params, bool cpu_to_gpu)
{
    if (params->voxelSizeWorksForFastSF() == false)
    {
        //printf("using extended\n");
        return project_eSF(g, f, params, cpu_to_gpu);
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
    
    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (cpu_to_gpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, false, bool(params->volumeDimensionOrder == 1));

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->geometry == parameters::CONE)
    {
        coneBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
        applyInversePolarWeight <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (cpu_to_gpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_phis);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

bool backproject_SF(float *g, float *&f, parameters* params, bool cpu_to_gpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;
    if (params->voxelSizeWorksForFastSF() == false)
    {
        //printf("using extended\n");
        return backproject_eSF_cone(g, f, params, cpu_to_gpu);
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
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
    
    if (cpu_to_gpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);
    if (params->geometry == parameters::CONE)
    {
        applyInversePolarWeight <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
    }

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, true);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    if (params->geometry == parameters::PARALLEL)
    {
        parallelBeamBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::FAN)
    {
        fanBeamBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else if (params->geometry == parameters::CONE)
    {
        coneBeamBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (cpu_to_gpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_phis);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else if (params->geometry == parameters::CONE)
    {
        applyPolarWeight <<< dimGrid_g, dimBlock_g >>> (dev_g, N_g, T_g, startVal_g);
        cudaStatus = cudaDeviceSynchronize();
    }

    return true;
}
