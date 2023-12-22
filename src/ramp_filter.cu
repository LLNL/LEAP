////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for GPU-based ramp and Hilbert filters
////////////////////////////////////////////////////////////////////////////////
#include "ramp_filter.cuh"
#include "ramp_filter_cpu.h"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.h"
#include "cpu_utils.h"

#define INCLUDE_CUFFT
#ifndef PI
#define PI 3.141592653589793
#endif

#ifdef INCLUDE_CUFFT
#include <cufft.h>
#endif

__global__ void ray_derivative_kernel(float* g, float* Dg, const int4 N, const float4 T, const float4 startVal, const float scalar, const float sampleShift)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    float diff = 0.0f;
    float* aLine = &g[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z)];
    if (sampleShift == 0.0f)
    {
        // central difference
        if (n == 0)
            diff = (aLine[n+1] - aLine[n])*scalar * 0.5f / T.z;
        else if (n == N.z-1)
            diff = (aLine[n] - aLine[n-1]) * scalar * 0.5f / T.z;
        else
            diff = (aLine[n + 1] - aLine[n - 1]) * scalar * 0.5f / T.z;
    }
    else if (sampleShift > 0.0f)
    {
        // forward difference
        if (n == N.z - 1)
            diff = 0.0f;
        else
            diff = (aLine[n + 1] - aLine[n]) * scalar / T.z;
    }
    else
    {
        // backward difference
        if (n == 0)
            diff = 0.0f;
        else
            diff = (aLine[n] - aLine[n-1]) * scalar / T.z;
    }
    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = diff;
}

__global__ void deriv_helical_NHDLH_curved(cudaTextureObject_t g, float* Dg, const int4 N, const float4 T, const float4 startVal, const float R, const float D, const float tau, const float helicalPitch, const float epsilon, const float* phis)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    const float v = m * T.y + startVal.y;
    const float u = n * T.z + startVal.z + 0.5f * T.z;

    const float cos_u = cos(u);
    const float sin_u = sin(u);
    const float lineLength = R * cos_u + tau * sin_u;

    const float one_over_T_v = 1.0f / T.y;
    const float one_over_T_u = 1.0f / T.z;
    const float u_shift = startVal.z * one_over_T_u - 0.5f;
    const float v_shift = startVal.y * one_over_T_v - 0.5f;

    const float l0 = (float)l;
    const int l_prev = max(0, l - 1);
    const int l_next = min(N.x - 1, l + 1);

    const float phi = phis[l];
    float T_phi = T.x;
    if (l == 0)
        T_phi = phis[1] - phis[0];
    else if (l == N.x - 1)
        T_phi = phis[N.x - 1] - phis[N.x - 2];
    else
        T_phi = 0.5f * (phis[l] - phis[l - 1] + phis[l + 1] - phis[l]);

    const float cos_T_phi_epsilon = cos(epsilon * T_phi);
    const float sin_T_phi_epsilon = sin(epsilon * T_phi);
    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);
    const float cos_phi_prev = cos(phis[l_prev]);
    const float sin_phi_prev = sin(phis[l_prev]);
    const float cos_phi_next = cos(phis[l_next]);
    const float sin_phi_next = sin(phis[l_next]);

    int shiftDirection;
    float B0, B1, B2;
    float one_over_neg_B_dot_theta;
    float cos_phi_epsilon, sin_phi_epsilon;
    float cos_phi_shift, sin_phi_shift;
    float u_arg, v_arg;

    shiftDirection = 0;
    cos_phi_epsilon = cos_phi * cos_T_phi_epsilon - sin_phi * sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi * cos_T_phi_epsilon + cos_phi * sin_T_phi_epsilon;
    cos_phi_shift = cos_phi;
    sin_phi_shift = sin_phi;
    
    shiftDirection = 0;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon - sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon + cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi;
    sin_phi_shift = sin_phi;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau*(sin_phi_epsilon-sin_phi_shift) - lineLength*(cos_phi*cos_u+sin_phi*sin_u);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau *(cos_phi_epsilon-cos_phi_shift) - lineLength*(sin_phi*cos_u-cos_phi*sin_u);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)* T_phi + lineLength*v;
    u_arg = one_over_T_u * atan((sin_phi_shift*B0 - cos_phi_shift*B1) / (cos_phi_shift*B0 + sin_phi_shift*B1)) - u_shift;
    v_arg = one_over_T_v * B2*rsqrt(B0*B0 + B1*B1) - v_shift;
    const float term1 = tex3D<float>(g, u_arg, v_arg, l0 + 0.5f);

    shiftDirection = l_next-l;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon - sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon + cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi_next;
    sin_phi_shift = sin_phi_next;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau *(sin_phi_epsilon-sin_phi_shift) - lineLength*(cos_phi*cos_u+sin_phi*sin_u);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau *(cos_phi_epsilon-cos_phi_shift) - lineLength*(sin_phi*cos_u-cos_phi*sin_u);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)* T_phi + lineLength*v;
    u_arg = one_over_T_u * atan((sin_phi_shift*B0 - cos_phi_shift*B1) / (cos_phi_shift*B0 + sin_phi_shift*B1)) - u_shift;
    v_arg = one_over_T_v * B2*rsqrt(B0*B0 + B1*B1) - v_shift;
    const float term2 = tex3D<float>(g, u_arg, v_arg, (float)l_next + 0.5f);

    shiftDirection = 0;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon + sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon - cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi;
    sin_phi_shift = sin_phi;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau *(sin_phi_epsilon-sin_phi_shift) - lineLength*(cos_phi*cos_u+sin_phi*sin_u);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau *(cos_phi_epsilon-cos_phi_shift) - lineLength*(sin_phi*cos_u-cos_phi*sin_u);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)* T_phi + lineLength*v;
    u_arg = one_over_T_u * atan((sin_phi_shift*B0 - cos_phi_shift*B1) / (cos_phi_shift*B0 + sin_phi_shift*B1)) - u_shift;
    v_arg = one_over_T_v * B2*rsqrt(B0*B0 + B1*B1) - v_shift;
    const float term3 = tex3D<float>(g, u_arg, v_arg, l0 + 0.5f);

    shiftDirection = l_prev-l;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon + sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon - cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi_prev;
    sin_phi_shift = sin_phi_prev;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau *(sin_phi_epsilon-sin_phi_shift) - lineLength*(cos_phi*cos_u+sin_phi*sin_u);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau *(cos_phi_epsilon-cos_phi_shift) - lineLength*(sin_phi*cos_u-cos_phi*sin_u);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)* T_phi + lineLength*v;
    u_arg = one_over_T_u * atan((sin_phi_shift*B0 - cos_phi_shift*B1) / (cos_phi_shift*B0 + sin_phi_shift*B1)) - u_shift;
    v_arg = one_over_T_v * B2*rsqrt(B0*B0 + B1*B1) - v_shift;
    const float term4 = tex3D<float>(g, u_arg, v_arg, (float)l_prev + 0.5f);

    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) / (4.0f * epsilon); // ? 1.0f / T_phi
}

__global__ void deriv_helical_NHDLH_flat(cudaTextureObject_t g, float* Dg, const int4 N, const float4 T, const float4 startVal, const float R, const float D, const float tau, const float helicalPitch, const float epsilon, const float* phis)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    const float v = m * T.y + startVal.y;
    const float u = n * T.z + startVal.z + 0.5f*T.z;

    const float lineLength = (R + tau * u) / (1.0f + u * u);

    const float one_over_T_v = 1.0f / T.y;
    const float one_over_T_u = 1.0f / T.z;
    const float u_shift = startVal.z * one_over_T_u - 0.5f;
    const float v_shift = startVal.y * one_over_T_v - 0.5f;

    const float l0 = (float)l;
    const int l_prev = max(0, l - 1);
    const int l_next = min(N.x - 1, l + 1);

    const float phi = phis[l];
    float T_phi = T.x;
    if (l == 0)
        T_phi = phis[1] - phis[0];
    else if (l == N.x - 1)
        T_phi = phis[N.x - 1] - phis[N.x - 2];
    else
        T_phi = 0.5f * (phis[l] - phis[l - 1] + phis[l + 1] - phis[l]);

    const float cos_T_phi_epsilon = cos(epsilon * T_phi);
    const float sin_T_phi_epsilon = sin(epsilon * T_phi);
    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);
    const float cos_phi_prev = cos(phis[l_prev]);
    const float sin_phi_prev = sin(phis[l_prev]);
    const float cos_phi_next = cos(phis[l_next]);
    const float sin_phi_next = sin(phis[l_next]);

    int shiftDirection;
    float B0, B1, B2;
    float one_over_neg_B_dot_theta;
    float cos_phi_epsilon, sin_phi_epsilon;
    float cos_phi_shift, sin_phi_shift;
    float u_arg, v_arg;

    shiftDirection = 0;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon - sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon + cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi;
    sin_phi_shift = sin_phi;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau*(sin_phi_epsilon-sin_phi_shift) + lineLength*(-cos_phi - u*sin_phi);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau*(cos_phi_epsilon-cos_phi_shift) + lineLength*(-sin_phi + u*cos_phi);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)*T_phi + lineLength*v;
    one_over_neg_B_dot_theta = 1.0f/(-B0*cos_phi_shift - B1*sin_phi_shift);
    u_arg = one_over_T_u * (-sin_phi_shift*B0 + cos_phi_shift*B1) * one_over_neg_B_dot_theta - u_shift;
    v_arg = one_over_T_v * B2 * one_over_neg_B_dot_theta - v_shift;
    const float term1 = tex3D<float>(g, u_arg, v_arg, l0+0.5f);

    shiftDirection = l_next-l;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon - sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon + cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi_next;
    sin_phi_shift = sin_phi_next;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau*(sin_phi_epsilon-sin_phi_shift) + lineLength*(-cos_phi - u*sin_phi);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau*(cos_phi_epsilon-cos_phi_shift) + lineLength*(-sin_phi + u*cos_phi);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)*T_phi + lineLength*v;
    one_over_neg_B_dot_theta = 1.0f/(-B0*cos_phi_shift - B1*sin_phi_shift);
    u_arg = one_over_T_u * (-sin_phi_shift*B0 + cos_phi_shift*B1) * one_over_neg_B_dot_theta - u_shift;
    v_arg = one_over_T_v * B2 * one_over_neg_B_dot_theta - v_shift;
    const float term2 = tex3D<float>(g, u_arg, v_arg, (float)l_next + 0.5f);

    shiftDirection = 0;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon + sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon - cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi;
    sin_phi_shift = sin_phi;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau*(sin_phi_epsilon-sin_phi_shift) + lineLength*(-cos_phi - u*sin_phi);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau*(cos_phi_epsilon-cos_phi_shift) + lineLength*(-sin_phi + u*cos_phi);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)*T_phi + lineLength*v;
    one_over_neg_B_dot_theta = 1.0f/(-B0*cos_phi_shift - B1*sin_phi_shift);
    u_arg = one_over_T_u * (-sin_phi_shift*B0 + cos_phi_shift*B1) * one_over_neg_B_dot_theta - u_shift;
    v_arg = one_over_T_v * B2 * one_over_neg_B_dot_theta - v_shift;
    const float term3 = tex3D<float>(g, u_arg, v_arg, l0 + 0.5f);

    shiftDirection = l_prev-l;
    cos_phi_epsilon = cos_phi*cos_T_phi_epsilon + sin_phi*sin_T_phi_epsilon;
    sin_phi_epsilon = sin_phi*cos_T_phi_epsilon - cos_phi*sin_T_phi_epsilon;
    cos_phi_shift = cos_phi_prev;
    sin_phi_shift = sin_phi_prev;
    B0 = R*(cos_phi_epsilon-cos_phi_shift) + tau *(sin_phi_epsilon-sin_phi_shift) + lineLength*(-cos_phi - u*sin_phi);
    B1 = R*(sin_phi_epsilon-sin_phi_shift) - tau *(cos_phi_epsilon-cos_phi_shift) + lineLength*(-sin_phi + u*cos_phi);
    B2 = helicalPitch *(epsilon-(float)shiftDirection)*T_phi + lineLength*v;
    one_over_neg_B_dot_theta = 1.0f/(-B0*cos_phi_shift - B1*sin_phi_shift);
    u_arg = one_over_T_u * (-sin_phi_shift*B0 + cos_phi_shift*B1) * one_over_neg_B_dot_theta - u_shift;
    v_arg = one_over_T_v * B2 * one_over_neg_B_dot_theta - v_shift;
    const float term4 = tex3D<float>(g, u_arg, v_arg, (float)l_prev + 0.5f);

    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) / (4.0f * epsilon); // ? 1.0f / T_phi
}

#ifdef INCLUDE_CUFFT
__global__ void splitLeftAndRight(const float* g, float* g_left, float* g_right, int4 N, float4 T, float4 startVal)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    int ind = j * N.z + k;

    float val = g[ind];

    //float s_ind_center = -startVal.z / T.z;

    float s = k * T.z + startVal.z;
    float s_conj = -s;
    float s_conj_ind = (s_conj - startVal.z) / T.z;
    float val_conj = 0.0f;
    if (0.0f <= s_conj_ind && s_conj_ind <= float(N.z - 1))
    {
        int s_lo = int(s_conj_ind);
        int s_hi = min(s_lo + 1, N.z - 1);
        float ds = s_conj_ind - float(s_lo);
        /*
        if (s_conj > 0.0f)
        {
            s_lo = max(s_lo, int(ceil(s_ind_center)));
            s_hi = max(s_hi, int(ceil(s_ind_center)));
        }
        else if (s_conj < 0.0f)
        {
            s_lo = min(s_lo, int(floor(s_ind_center)));
            s_hi = min(s_hi, int(floor(s_ind_center)));
        }
        //*/
        val_conj = (1.0f - ds) * g[j * N.z+s_lo] + ds * g[j * N.z+s_hi];
    }

    if (s > 0.0f)
    {
        g_right[ind] = val;
        g_left[ind] = val_conj;
    }
    else if (s < 0.0f)
    {
        g_right[ind] = val_conj;
        g_left[ind] = val;
    }
    else
    {
        g_left[ind] = val;
        g_right[ind] = val;
    }
}

__global__ void mergeLeftAndRight(float* g, const float* g_left, const float* g_right, int4 N, float4 T, float4 startVal)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    int ind = j * N.z + k;

    float s = k * T.z + startVal.z;
    if (s >= 0.0f)
        g[ind] = g_right[ind];
    else
        g[ind] = g_left[ind];
}

__global__ void multiply2DRampFilterKernel(cufftComplex* F, const float* H, int3 N)
{
    // int k = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    //if (k > 0) return;
    if (k > N.z - 1)
    	return;
    cufftComplex* F_slice = &F[uint64(k) * uint64(N.x * N.y)];
    for (int j = 0; j < N.y; j++)
    {
        for (int i = 0; i < N.x; i++)
        {
            F_slice[j * N.x + i].x *= H[j * N.x + i];
            F_slice[j * N.x + i].y *= H[j * N.x + i];
        }
    }
}

__global__ void setPaddedDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, int numExtrapolate, const float4 T, const float4 startVals, const float R, const float helicalPitch)
{
    int j = threadIdx.x;
    int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[uint64(i - startView) * uint64(N_pad * N.y) + uint64(j * N_pad)];
    float* data_block = &data[uint64(i) * uint64(N.z * N.y) + uint64(j * N.z)];

    if (helicalPitch == 0.0f)
    {
        for (int k = 0; k < N.z; k++)
            data_padded_block[k] = data_block[k];
    }
    else
    {
        // Do helical forward height rebinning while copying data to padded array
        const float last_v_sample = float(N.y - 1);
        float* aProj = &data[uint64(i) * uint64(N.z * N.y)];
        for (int k = 0; k < N.z; k++)
        {
            const float theShift = helicalPitch / R * (k * T.z + startVals.z);
            const float v_new = j*T.y + startVals.y;
            const float v_old = v_new - theShift;

            const float v_ind = (v_old - startVals.y)/T.y;
            const int v_closest = int(0.5f + v_ind);
            if (v_ind <= 0.0f)
            {
                data_padded_block[k] = aProj[0 * N.z + k];
            }
            else if (v_ind >= last_v_sample)
            {
                data_padded_block[k] = aProj[(N.y - 1)*N.z + k];
            }
            else
            {
                const int v_low = int(v_ind);
                const int v_high = v_low + 1;
                const float dv = v_ind - float(v_low);
                data_padded_block[k] = (1.0f - dv) * aProj[v_low * N.z + k] + dv * aProj[v_high * N.z + k];
            }
        }
    }

    for (int k = N.z; k < N_pad; k++)
        data_padded_block[k] = 0.0;

    if (numExtrapolate > 0)
    {
        const float leftVal = data_block[0];
        const float rightVal = data_block[N.z - 1];
        for (int k = N.z; k < N.z + numExtrapolate; k++)
            data_padded_block[k] = rightVal;
        for (int k = N_pad - numExtrapolate; k < N_pad; k++)
            data_padded_block[k] = leftVal;
    }
}

__global__ void multiplyRampFilterKernel(cufftComplex* G, const float* H, int3 N)
{
    int j = threadIdx.x;
    int i = blockIdx.x;
    if (i > N.x - 1 || j > N.y - 1)
        return;
    cufftComplex* aProj = &G[uint64(i) * uint64(N.y * N.z)];
    for (int k = 0; k < N.z; k++)
    {
        aProj[j * N.z + k].x *= H[k];
        aProj[j * N.z + k].y *= H[k];
    }
}

__global__ void multiplyComplexFilterKernel(cufftComplex* G, const cufftComplex* H, int3 N)
{
    int j = threadIdx.x;
    int i = blockIdx.x;
    if (i > N.x - 1 || j > N.y - 1)
        return;
    cufftComplex* G_row = &G[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z)];
    for (int k = 0; k < N.z; k++)
    {
        const float realPart = G_row[k].x * H[k].x - G_row[k].y * H[k].y;
        const float imagPart = G_row[k].x * H[k].y + G_row[k].y * H[k].x;
        G_row[k].x = realPart;
        G_row[k].y = imagPart;
    }
}

__global__ void setFilteredDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, const float4 T, const float4 startVals, const float R, const float helicalPitch)
{
    int j = threadIdx.x;
    int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[uint64(i - startView) * uint64(N_pad * N.y) + uint64(j * N_pad)];
    float* data_block = &data[uint64(i) * uint64(N.z * N.y) + uint64(j * N.z)];

    if (helicalPitch == 0.0f)
    {
        for (int k = 0; k < N.z; k++)
            data_block[k] = data_padded_block[k];
    }
    else
    {
        // Do helical backward height rebinning while copying data to padded array
        const float last_v_sample = float(N.y - 1);
        float* aProj = &data_padded[uint64(i - startView) * uint64(N_pad * N.y)];
        for (int k = 0; k < N.z; k++)
        {
            const float theShift = helicalPitch / R * (k * T.z + startVals.z);
            const float v_new = j * T.y + startVals.y;
            const float v_old = v_new + theShift;

            const float v_ind = (v_old - startVals.y) / T.y;
            const int v_closest = int(0.5f + v_ind);
            if (v_ind <= 0.0f)
            {
                data_block[k] = aProj[0 * N_pad + k];
            }
            else if (v_ind >= last_v_sample)
            {
                data_block[k] = aProj[(N.y - 1) * N_pad + k];
            }
            else
            {
                const int v_low = int(v_ind);
                const int v_high = v_low + 1;
                const float dv = v_ind - float(v_low);
                data_block[k] = (1.0f - dv) * aProj[v_low * N_pad + k] + dv * aProj[v_high * N_pad + k];
            }
        }
    }
}

cufftComplex* HilbertTransformFrequencyResponse(int N, parameters* params, float scalar, float sampleShift)
{
    cudaError_t cudaStatus;
    double* h_d = NULL;
    if (sampleShift == 0.0)
        h_d = HilbertTransformImpulseResponse(N, 0);
    else if (sampleShift > 0.0)
    {
        h_d = HilbertTransformImpulseResponse(N, 1);
        params->colShiftFromFilter -= 0.5;
    }
    else
    {
        h_d = HilbertTransformImpulseResponse(N, -1);
        params->colShiftFromFilter += 0.5;
    }
    float* h = new float[N];
    for (int i = 0; i < N; i++)
        h[i] = h_d[i] * scalar / float(N);
    delete[] h_d;

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N, CUFFT_R2C, 1))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft");
        return NULL;
    }

    float* dev_h = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        return NULL;
    }
    cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of ramp filter) failed!\n");
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    cudaDeviceSynchronize();

    // get result
    cufftComplex* H_Hilb = new cufftComplex[N_over2];
    cudaStatus = cudaMemcpy(H_Hilb, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;

    return H_Hilb;
}

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params)
{
    float T = params->pixelWidth;
    bool isCurved = false;
    if (params->geometry == parameters::FAN || params->geometry == parameters::CONE)
    {
        T *= params->sod / params->sdd;
        if (params->detectorType == parameters::CURVED)
            isCurved = true;
    }

    cudaError_t cudaStatus;
    double* h_d = rampImpulseResponse(N, T, params);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = h_d[i];

        if (i != 0 && isCurved == true)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= temp * temp;
        }
    }
    delete[] h_d;

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N, CUFFT_R2C, 1))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft");
        return NULL;
    }

    float* dev_h = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        return NULL;
    }
    cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of ramp filter) failed!\n");
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    cudaDeviceSynchronize();

    // get result
    cufftComplex* H_ramp = new cufftComplex[N_over2];
    cudaStatus = cudaMemcpy(H_ramp, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    float* H_real = new float[N_over2];
    for (int i = 0; i < N_over2; i++)
    {
        H_real[i] = H_ramp[i].x / float(N);
    }

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;
    delete[] H_ramp;

    return H_real;
}

bool rampFilter1D_symmetric(float*& g, parameters* params, float scalar)
{
    //printf("rampFilter1D_symmetric...\n");
    bool data_on_cpu = false;
    bool retVal = true;
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

    float* dev_g = g;

    float* dev_g_left = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g_left, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(projections) failed!\n");
        return false;
    }

    float* dev_g_right = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g_right, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(projections) failed!\n");
        return false;
    }

    // Make thread block structure
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);

    // Make copies to dev_g_lef and dev_g_right
    splitLeftAndRight <<< dimGrid, dimBlock >>> (dev_g, dev_g_left, dev_g_right, N_g, T_g, startVal_g);
    cudaStatus = cudaDeviceSynchronize();

    // Do ramp filter
    rampFilter1D(dev_g_left, params, data_on_cpu, scalar);
    rampFilter1D(dev_g_right, params, data_on_cpu, scalar);

    // Merge back to g
    mergeLeftAndRight <<< dimGrid, dimBlock >>> (dev_g, dev_g_left, dev_g_right, N_g, T_g, startVal_g);
    cudaStatus = cudaDeviceSynchronize();

    // Clean up
    cudaFree(dev_g_left);
    cudaFree(dev_g_right);

    return retVal;
}

bool rampFilter1D(float*& g, parameters* params, bool data_on_cpu, float scalar)
{
    return conv1D(g, params, data_on_cpu, scalar, 0);
}

bool Hilbert1D(float*& g, parameters* params, bool data_on_cpu, float scalar, float sampleShift)
{
    return conv1D(g, params, data_on_cpu, scalar, 1, sampleShift);
}

bool conv1D(float*& g, parameters* params, bool data_on_cpu, float scalar, int which, float sampleShift)
{
    bool retVal = true;
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    if (data_on_cpu)
    {
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    }
    else
    {
        dev_g = g;
    }

    // PUT CODE HERE
    //int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    int N_H = optimalFFTsize(2 * params->numCols);
    //printf("FFT size = %d\n", N_H);
    int N_H_over2 = N_H / 2 + 1;
    float* H_real = NULL;
    cufftComplex* H_comp = NULL;
    if (which == 0)
        H_real = rampFilterFrequencyResponseMagnitude(N_H, params);
    else
        H_comp = HilbertTransformFrequencyResponse(N_H, params, scalar, sampleShift);
    if (scalar != 1.0)
    {
        for (int i = 0; i < N_H_over2; i++)
        {
            if (H_real != NULL)
                H_real[i] *= scalar;
        }
    }

    //if (H_comp != NULL)
    //    printf("doing Hilbert filter\n");

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);
    float helicalPitch = params->helicalPitch;
    if (H_comp == NULL || params->numRows == 1)
        helicalPitch = 0.0;
    helicalPitch = 0.0;
    //printf("max shift = %f\n", params->helicalPitch/params->sod*(startVal_g.z));
    //printf("pitch/R = %f\n", params->helicalPitch / params->sod);
    //printf("T_v = %f\n", T_g.y);

    int numRows = params->numRows;
    int numAngles = params->numAngles;
    if (numAngles == 1)
    {
        numRows = 1;
        numAngles = params->numRows;
    }

    //int N_viewChunk = params->numAngles;
    int N_viewChunk = max(1, numAngles / 40); // number of views in a chunk (needs to be optimized)
    int numChunks = int(ceil(double(numAngles) / double(N_viewChunk)));

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N_H, CUFFT_R2C, uint64(N_viewChunk) * uint64(numRows)))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft (size %d)\n", N_H);
        return false;
    }
    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&backward_plan, N_H, CUFFT_C2R, uint64(N_viewChunk) * uint64(numRows))) // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 1d c2r ifft\n");
        return false;
    }
    //return true;

    float* dev_g_pad = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_g_pad, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H) * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_G = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_G, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H_over2) * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded projection data) failed!\n");
        retVal = false;
    }

    // Copy filter to device
    float* dev_H = 0;
    cufftComplex* dev_cH = 0;
    if (H_real != NULL)
    {
        if (cudaSuccess != cudaMalloc((void**)&dev_H, N_H_over2 * sizeof(float)))
            fprintf(stderr, "cudaMalloc failed!\n");
        cudaStatus = cudaMemcpy(dev_H, H_real, N_H_over2 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaSuccess != cudaStatus)
        {
            fprintf(stderr, "cudaMemcpy(H) failed!\n");
            fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
            fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            retVal = false;
        }
    }
    else if (H_comp != NULL)
    {
        if (cudaSuccess != cudaMalloc((void**)&dev_cH, N_H_over2 * sizeof(cufftComplex)))
            fprintf(stderr, "cudaMalloc failed!\n");
        cudaStatus = cudaMemcpy(dev_cH, H_comp, N_H_over2 * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        if (cudaSuccess != cudaStatus)
        {
            fprintf(stderr, "cudaMemcpy(H) failed!\n");
            fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
            fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            retVal = false;
        }
    }
    int3 dataSize; dataSize.x = N_viewChunk; dataSize.y = numRows; dataSize.z = N_H_over2;
    int3 origSize; origSize.x = numAngles; origSize.y = numRows; origSize.z = params->numCols;

    int numExtrapolate = 0;
    if (params->truncatedScan)
        numExtrapolate = min(N_H - params->numCols - 1, 100);

    if (retVal == true)
    {
        for (int iChunk = 0; iChunk < numChunks; iChunk++)
        {
            int startView = iChunk * N_viewChunk;
            int endView = min(numAngles - 1, startView + N_viewChunk - 1);
            //printf("filtering %d to %d\n", startView, endView);

            setPaddedDataKernel <<< endView - startView + 1, numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate, T_g, startVal_g, params->sod, helicalPitch);
            //cudaDeviceSynchronize();

            // FFT
            result = cufftExecR2C(forward_plan, (cufftReal*)dev_g_pad, dev_G);

            // Multiply Filter
            if (dev_H != 0)
                multiplyRampFilterKernel <<< N_viewChunk, numRows >>> (dev_G, dev_H, dataSize);
            else if (dev_cH != 0)
                multiplyComplexFilterKernel <<< N_viewChunk, numRows >>> (dev_G, dev_cH, dataSize);
            //cudaDeviceSynchronize();

            // IFFT
            result = cufftExecC2R(backward_plan, (cufftComplex*)dev_G, (cufftReal*)dev_g_pad);

            setFilteredDataKernel <<< endView - startView + 1, numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, T_g, startVal_g, params->sod, helicalPitch);
            //cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();

        if (data_on_cpu)
        {
            // Copy result back to host
            cudaStatus = cudaMemcpy(g, dev_g, uint64(numAngles) * uint64(numRows) * uint64(params->numCols) * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            }
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_g_pad);
    if (data_on_cpu)
        cudaFree(dev_g);
    if (dev_H != 0)
        cudaFree(dev_H);
    if (dev_cH != 0)
        cudaFree(dev_cH);
    cudaFree(dev_G);
    if (H_real != NULL)
        delete[] H_real;
    if (H_comp != NULL)
        delete[] H_comp;

    return retVal;
}

bool rampFilter2D(float*& f, parameters* params, bool data_on_cpu)
{
    if (data_on_cpu == false)
    {
        printf("Error: current implementation of rampFilter2D requires that data reside on the CPU\n");
        return false;
    }

    int N_x = params->numX;
    int N_y = params->numY;
    int N_z = params->numZ;

    // Pad and then find next power of 2
    //int N_H1 = int(pow(2.0, ceil(log2(2 * max(N_y, N_x)))));
    int N_H1 = optimalFFTsize(2 * max(N_y, N_x));
    int N_H2 = N_H1;
    int N_H2_over2 = N_H2 / 2 + 1;

    cudaSetDevice(params->whichGPU);
    bool retVal = true;

    int smoothingLevel = 0;

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan2d(&forward_plan, N_H1, N_H2, CUFFT_R2C))
    {
        fprintf(stderr, "Failed to plan 2d r2c fft");
        return false;
    }
    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan2d(&backward_plan, N_H1, N_H2, CUFFT_C2R))  // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 2d c2r ifft");
        return false;
    }

    float* paddedSlice = (float*)malloc(sizeof(float) * N_H1 * N_H2);
    // Make zero-padded array, copy data to 1st half of array and set remaining slots to zero
    cudaError_t cudaStatus;
    float* dev_f_pad = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_f_pad, N_H1 * N_H2 * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded volume data) failed!\n");
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_F = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_F, N_H1 * N_H2_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded volume data) failed!\n");
        retVal = false;
    }

    // Copy filter to device
    float* H = rampFrequencyResponse2D(N_H1, 1.0, 1.0, smoothingLevel);  // FIXME?
    float* dev_H = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_H, N_H1 * N_H2_over2 * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    cudaStatus = cudaMemcpy(dev_H, H, N_H1 * N_H2_over2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "cudaMemcpy(H) failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    for (int k = 0; k < N_z; k++)
    {
        if (params->volumeDimensionOrder == parameters::XYZ)
        {
            for (int j = 0; j < N_H1; j++)
            {
                int j_source = j;
                if (j >= N_y)
                {
                    if (j - N_y < N_H1 - j)
                        j_source = N_y - 1;
                    else
                        j_source = 0;
                }
                for (int i = 0; i < N_H2; i++)
                {
                    int i_source = i;
                    if (i >= N_x)
                    {
                        if (i - N_x < N_H2 - i)
                            i_source = N_x - 1;
                        else
                            i_source = 0;
                    }
                    paddedSlice[j * N_H2 + i] = f[uint64(i_source) * uint64(N_y * N_z) + uint64(j_source*N_z + k)];
                }
            }
        }
        else //if (params->volumeDimensionOrder == parameters::ZYX)
        {
            float* f_slice = &f[uint64(k) * uint64(N_x * N_y)];
            for (int j = 0; j < N_H1; j++)
            {
                int j_source = j;
                if (j >= N_y)
                {
                    if (j - N_y < N_H1 - j)
                        j_source = N_y - 1;
                    else
                        j_source = 0;
                }
                for (int i = 0; i < N_H2; i++)
                {
                    int i_source = i;
                    if (i >= N_x)
                    {
                        if (i - N_x < N_H2 - i)
                            i_source = N_x - 1;
                        else
                            i_source = 0;
                    }
                    paddedSlice[j * N_H2 + i] = f_slice[j_source * N_x + i_source];
                }
            }
        }
        if (cudaMemcpy(dev_f_pad, paddedSlice, N_H1 * N_H2 * sizeof(float), cudaMemcpyHostToDevice))
        {
            fprintf(stderr, "cudaMemcpy(padded volume data) failed!\n");
            retVal = false;
        }

        // FFT
        result = cufftExecR2C(forward_plan, (cufftReal*)dev_f_pad, dev_F);

        // Multiply Filter
        int3 dataSize;
        dataSize.z = N_z;
        dataSize.y = N_H1;
        dataSize.x = N_H2_over2;
        multiply2DRampFilterKernel<<<1, 1>>>(dev_F, dev_H, dataSize);

        // IFFT
        result = cufftExecC2R(backward_plan, (cufftComplex*)dev_F, (cufftReal*)dev_f_pad);

        // Copy result back to host
        if (retVal)
        {
            cudaStatus = cudaMemcpy(paddedSlice, dev_f_pad, N_H1 * N_H2 * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
                retVal = false;
            }
            if (params->volumeDimensionOrder == parameters::XYZ)
            {
                for (int j = 0; j < N_y; j++)
                {
                    for (int i = 0; i < N_x; i++)
                    {
                        f[uint64(i) * uint64(N_y * N_z) + uint64(j * N_z + k)] = paddedSlice[j * N_H2 + i] / float(N_H1 * N_H2);
                    }
                }
            }
            else
            {
                float* f_slice = &f[uint64(k) * uint64(N_x * N_y)];
                for (int j = 0; j < N_y; j++)
                {
                    for (int i = 0; i < N_x; i++)
                    {
                        f_slice[j * N_x + i] = paddedSlice[j * N_H2 + i] / float(N_H1 * N_H2);
                    }
                }
            }
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_f_pad);
    cudaFree(dev_H);
    cudaFree(dev_F);
    free(H);
    free(paddedSlice);

    return retVal;
}
#else
bool rampFilter1D(float*& g, parameters* params, bool data_on_cpu)
{
    //printf("CUFFT libraries not available!\n");
    //return false;
    return rampFilter1D_cpu(g, params);
}

bool Hilbert1D(float*& g, parameters* params, bool data_on_cpu)
{
    //printf("CUFFT libraries not available!\n");
    //return false;
    return Hilbert1D_cpu(g, params);
}

bool rampFilter2D(float*& f, parameters* params, bool data_on_cpu)
{
    printf("CUFFT libraries not available!\n");
    return false;
}
#endif

bool ray_derivative(float*& g, parameters* params, bool data_on_cpu, float scalar, float sampleShift)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float* dev_g = 0;
    float* dev_Dg = 0;
    if (data_on_cpu)
    {
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
        if (cudaSuccess != cudaMalloc((void**)&dev_Dg, params->projectionData_numberOfElements() * sizeof(float)))
            fprintf(stderr, "cudaMalloc failed!\n");
    }
    else
    {
        dev_Dg = g;
        if (cudaSuccess != cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float)))
            fprintf(stderr, "cudaMalloc failed!\n");
        equal(dev_g, dev_Dg, make_int3(N_g.x, N_g.y, N_g.z), params->whichGPU);
    }

    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    ray_derivative_kernel <<< dimGrid, dimBlock >>> (dev_g, dev_Dg, N_g, T_g, startVal_g, scalar, sampleShift);
    params->colShiftFromFilter += -0.5*sampleShift; // opposite sign as LTT

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_Dg, params->whichGPU);

    if (data_on_cpu == true && dev_Dg != 0)
        cudaFree(dev_Dg);
    if (dev_g != 0)
        cudaFree(dev_g);

    return true;
}

bool parallelRay_derivative(float*& g, parameters* params, bool data_on_cpu)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float epsilon = std::min(0.01, T_g.z / (4.0 * fabs(params->T_phi())));

    float* dev_g = 0;
    float* dev_Dg = 0;
    if (data_on_cpu)
    {
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
        if (cudaSuccess != cudaMalloc((void**)&dev_Dg, params->projectionData_numberOfElements() * sizeof(float)))
            fprintf(stderr, "cudaMalloc failed!\n");
    }
    else
    {
        dev_Dg = g;
        if (cudaSuccess != cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float)))
            fprintf(stderr, "cudaMalloc failed!\n");
        equal(dev_g, dev_Dg, make_int3(N_g.x, N_g.y, N_g.z), params->whichGPU);
    }

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, true, true);

    float* dev_phis = copyAngleArrayToGPU(params);

    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->detectorType == parameters::FLAT)
        deriv_helical_NHDLH_flat <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis);
    else
        deriv_helical_NHDLH_curved <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis);
    params->colShiftFromFilter += -0.5; // opposite sign as LTT
    //params->rowShiftFromFilter += -0.5;


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_Dg, params->whichGPU);

    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);

    if (data_on_cpu == true && dev_Dg != 0)
        cudaFree(dev_Dg);
    if (dev_phis != 0)
        cudaFree(dev_phis);
    if (dev_g != 0)
        cudaFree(dev_g);

    return true;
}
