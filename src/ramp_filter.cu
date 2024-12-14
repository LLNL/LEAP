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
#include "ray_weighting_cpu.h"
#include "log.h"

#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.h"
#include "cpu_utils.h"

#ifndef PI
#define PI 3.141592653589793
#endif

#define NUM_RAYS_PER_THREAD 8

#ifdef __INCLUDE_CUFFT
#include <cufft.h>

__global__ void zeroPadForOffsetScanKernel(float* g, float* g_pad, const int3 N, const int N_add, const bool padOnLeft, const float* offsetScanWeights)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    const int numCols = N.z - N_add;
    const uint64 ind = uint64(i) * uint64(numCols * N.y) + uint64(j * numCols);
    const uint64 ind_pad = uint64(i) * uint64(N.z * N.y) + uint64(j * N.z);

    if (padOnLeft)
    {
        if (k < N_add)
            g_pad[ind_pad + k + N_add] = 0.0f;
        else
            g_pad[ind_pad + uint64(k)] = g[ind + uint64(k - N_add)] * 2.0f * offsetScanWeights[j * numCols + k-N_add];
    }
    else
    {
        if (k < numCols)
            g_pad[ind_pad + uint64(k)] = g[ind + uint64(k)] * 2.0f * offsetScanWeights[j * numCols + k];
        else
            g_pad[ind_pad + uint64(k)] = 0.0f;
    }
}

__global__ void multiplyRampFilterKernel(cufftComplex* G, const float* H, int3 N)
{
    /*
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
    //*/

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;
    uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    G[ind].x *= H[k];
    G[ind].y *= H[k];
}

__global__ void multiplyComplexFilterKernel(cufftComplex* G, const cufftComplex* H, int3 N)
{
    /*
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
    //*/

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;
    uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);

    const float realPart = G[ind].x * H[k].x - G[ind].y * H[k].y;
    const float imagPart = G[ind].x * H[k].y + G[ind].y * H[k].x;

    G[ind].x = realPart;
    G[ind].y = imagPart;
}

__global__ void multiply2DRampFilterKernel(cufftComplex* F, const float* H, int3 N)
{
    /*
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
    //*/

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    const uint64 offset = uint64(j * N.x + i);
    const uint64 ind = uint64(k) * uint64(N.x * N.y) + uint64(j * N.x + i);
    F[ind].x *= H[offset];
    F[ind].y *= H[offset];
}
#endif

__global__ void explicit_convolution(cudaTextureObject_t g, float* filtered_data, cudaTextureObject_t h, int3 N, int N_filter)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = (threadIdx.z + blockIdx.z * blockDim.z) * NUM_RAYS_PER_THREAD;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    const uint64 ind = uint64(l) * uint64(N.z * N.y) + uint64(m * N.z);

    // y[n] \sum_j h[j] x[n-j]
    // 0 <= n-j <= N.z-1
    // 1-N.z <= j-n <= 0
    // 1-N.z+n <= j <= n
    float ys[NUM_RAYS_PER_THREAD];
    for (int j = 0; j < NUM_RAYS_PER_THREAD; j++)
        ys[j] = 0.0f;

    for (int j = 0; j < N.z; j+=8)
    {
        const float x0 = tex3D<float>(g, j + 0, m, l);
        const float x1 = tex3D<float>(g, j + 1, m, l);
        const float x2 = tex3D<float>(g, j + 2, m, l);
        const float x3 = tex3D<float>(g, j + 3, m, l);
        const float x4 = tex3D<float>(g, j + 4, m, l);
        const float x5 = tex3D<float>(g, j + 5, m, l);
        const float x6 = tex3D<float>(g, j + 6, m, l);
        const float x7 = tex3D<float>(g, j + 7, m, l);

        const int n_minus_j_plus_N = n - j + N.z;
        for (int s = 0; s < NUM_RAYS_PER_THREAD; s++)
        {
            ys[s] += tex1D<float>(h, n_minus_j_plus_N + s - 0) * x0 + tex1D<float>(h, n_minus_j_plus_N + s - 1) * x1
                  +  tex1D<float>(h, n_minus_j_plus_N + s - 2) * x2 + tex1D<float>(h, n_minus_j_plus_N + s - 3) * x3
                  +  tex1D<float>(h, n_minus_j_plus_N + s - 4) * x4 + tex1D<float>(h, n_minus_j_plus_N + s - 5) * x5
                  +  tex1D<float>(h, n_minus_j_plus_N + s - 6) * x6 + tex1D<float>(h, n_minus_j_plus_N + s - 7) * x7;
        }
    }
    for (int s = 0; s < min(NUM_RAYS_PER_THREAD, N.z-n); s++)
        filtered_data[ind + n + s] = ys[s];
}

__global__ void setPaddedDataFor2DFilter_reverse(float* g, float* g_pad, const int numRows, const int numCols, const int N_H1, const int N_H2, const float minValue, const bool isAttenuationData)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x; // rows
    const int i = threadIdx.y + blockIdx.y * blockDim.y; // columns

    if (j >= numRows || i >= numCols)
        return;

    if (isAttenuationData)
        g[j * numCols + i] = -log(max(minValue, g_pad[j * N_H2 + i] / float(N_H1 * N_H2)));
    else
        g[j * numCols + i] = max(minValue, g_pad[j * N_H2 + i] / float(N_H1 * N_H2));
}

__global__ void setPaddedDataFor2DFilter(float* g, float* g_pad, const int numRows, const int numCols, const int N_H1, const int N_H2, const bool isAttenuationData)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x; // rows
    const int i = threadIdx.y + blockIdx.y * blockDim.y; // columns

    if (j >= N_H1 || i >= N_H2)
        return;

    int j_source = j;
    if (j >= numRows)
    {
        if (j - numRows < N_H1 - j)
            j_source = numRows - 1;
        else
            j_source = 0;
    }

    int i_source = i;
    if (i >= numCols)
    {
        if (i - numCols < N_H2 - i)
            i_source = numCols - 1;
        else
            i_source = 0;
    }

    if (isAttenuationData)
        g_pad[j * N_H2 + i] = expf(-g[j_source * numCols + i_source]);
    else
        g_pad[j * N_H2 + i] = g[j_source * numCols + i_source];

    //int N_x = params->numCols;
    //int N_y = params->numRows;
    //int N_z = params->numAngles;

    /*
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
            if (isAttenuationData)
                paddedProj[j * N_H2 + i] = exp(-aProj[j_source * N_x + i_source]);
            else
                paddedProj[j * N_H2 + i] = aProj[j_source * N_x + i_source];
        }
    }
    //*/
}

__global__ void Laplacian_kernel(float* g, float* Dg, const int4 N, const float4 T, const float4 startVal, const int numDims, const bool smooth, const float scalar)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    float diff = 0.0f;
    float* proj = &g[uint64(l) * uint64(N.z * N.y)];
    if (smooth)
    {
        if (N.z >= 3)
        {
            const int n_plus_two = min(n + 2, N.z - 1);
            const int n_minus_two = max(n - 2, 0);
            diff = 0.25f * (proj[m * N.z + n_plus_two] + proj[m * N.z + n_minus_two]) - 0.5f * proj[m * N.z + n];
        }
        if (N.y >= 3 && numDims >= 2)
        {
            const int m_plus_two = min(m + 2, N.y - 1);
            const int m_minus_two = max(m - 2, 0);
            diff += 0.25f * (proj[m_plus_two * N.z + n] + proj[m_minus_two * N.z + n]) - 0.5f * proj[m * N.z + n];
        }
    }
    else
    {
        if (N.z >= 3)
        {
            const int n_plus_one = min(n + 1, N.z - 1);
            const int n_minus_one = max(n - 1, 0);
            diff = proj[m * N.z + n_plus_one] + proj[m * N.z + n_minus_one] - 2.0f * proj[m * N.z + n];
        }
        if (N.y >= 3 && numDims >= 2)
        {
            const int m_plus_one = min(m + 1, N.y - 1);
            const int m_minus_one = max(m - 1, 0);
            diff += proj[m_plus_one * N.z + n] + proj[m_minus_one * N.z + n] - 2.0f * proj[m * N.z + n];
        }
    }
    
    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = diff * scalar;
}

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

__global__ void deriv_helical_NHDLH_curved(cudaTextureObject_t g, float* Dg, const int4 N, const float4 T, const float4 startVal, const float R, const float D, const float tau, const float helicalPitch, const float epsilon, const float* phis, const int iphi_offset)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x + iphi_offset;
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
    //float one_over_neg_B_dot_theta;
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

    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) / (2.0f * epsilon * R * T_phi); // ? 1.0f / T_phi
    //Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) * 2.0f * PI / (R * T.z);
}

__global__ void deriv_helical_NHDLH_flat(cudaTextureObject_t g, float* Dg, const int4 N, const float4 T, const float4 startVal, const float R, const float D, const float tau, const float helicalPitch, const float epsilon, const float* phis, const int iphi_offset)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x + iphi_offset;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N.x || m >= N.y || n >= N.z)
        return;

    const float v = m * T.y + startVal.y;
    const float u = n * T.z + startVal.z + 0.5f*T.z;

    //const float lineLength = (R + tau * u) / (1.0f + u * u);
    //const float lineLength = (R + tau * u) * rsqrtf(1.0f + u * u);
    //const float lineLength = R + tau * u;
    const float lineLength = R + tau * u / (1.0f + u * u);

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

    Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) / (2.0f * epsilon * R * T_phi); // ? 1.0f / T_phi
    //Dg[uint64(l) * uint64(N.z * N.y) + uint64(m * N.z + n)] = ((1.0f - epsilon) * (term1 - term3) + epsilon * (term2 - term4)) * 2.0f * PI / (R * T.z);
}

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

__global__ void setPaddedDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, int numExtrapolate, const float4 T, const float4 startVals, const float R, const float helicalPitch)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x + startView;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //int j = threadIdx.x;
    //int i = blockIdx.x + startView;
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
            //const int v_closest = int(0.5f + v_ind);
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
        data_padded_block[k] = 0.0f;

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

__global__ void setFilteredDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, const float4 T, const float4 startVals, const float R, const float helicalPitch)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x + startView;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //int j = threadIdx.x;
    //int i = blockIdx.x + startView;
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
            //const int v_closest = int(0.5f + v_ind);
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LAUNCHING FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __INCLUDE_CUFFT
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
    float T = params->pixelWidth * params->sod / params->sdd;
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = float(h_d[i] * scalar / float(N));
        if (i != 0 && params->geometry == parameters::CONE && params->detectorType == parameters::CURVED)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= temp;// *temp;
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
    if ((cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Hilbert filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(Hilbert filter) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of Hilbert filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!\n");
        printf("cudaDeviceSynchronize Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    // get result
    cufftComplex* H_Hilb = new cufftComplex[N_over2];
    if ((cudaStatus = cudaMemcpy(H_Hilb, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    float theExponent = 1.0;
    if (params->FBPlowpass >= 2.0)
    {
        theExponent = 1.0 / (1.0 - log2(1.0 + cos(PI / params->FBPlowpass)));
        //printf("theExponent = %f\n", theExponent);
        for (int i = 0; i < N_over2; i++)
        {
            //float omega = float(i)*PI / N_over2;
            float omega = float(i) * PI / N;
            float theWeight = pow(max(0.0, cos(omega)), 2.0 * theExponent);

            H_Hilb[i].x *= theWeight;
            H_Hilb[i].y *= theWeight;
            //printf("H(%f) = %f (%d)\n", omega, theWeight, i);
        }
    }
    

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;

    return H_Hilb;
}

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params)
{
    cudaError_t cudaStatus;
    float* h = rampImpulseResponse_modified(N, params);

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N, CUFFT_R2C, 1))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft");
        return NULL;
    }

    float* dev_h = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(ramp filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(ramp filter) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of ramp filter) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_h);
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!\n");
        printf("cudaDeviceSynchronize Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    // get result
    cufftComplex* H_ramp = new cufftComplex[N_over2];
    if ((cudaStatus = cudaMemcpy(H_ramp, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }

    float theExponent = 1.0;
    if (params->FBPlowpass >= 2.0)
    {
        theExponent = 1.0 / (1.0 - log2(1.0 + cos(PI / params->FBPlowpass)));
        //printf("theExponent = %f\n", theExponent);
    }

    float* H_real = new float[N_over2];
    for (int i = 0; i < N_over2; i++)
    {
        H_real[i] = H_ramp[i].x / float(N);
        if (params->FBPlowpass >= 2.0)
        {
            //float omega = float(i)*PI / N_over2;
            float omega = float(i) * PI / N;
            float theWeight = pow(max(0.0, cos(omega)), 2.0 * theExponent);

            H_real[i] *= theWeight;
            //printf("H(%f) = %f (%d)\n", omega, theWeight, i);
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;
    delete[] H_ramp;

    return H_real;
}

bool conv1D(float*& g, parameters* params, bool data_on_cpu, float scalar, int which, float sampleShift)
{
    LOG(logDEBUG, "ramp_filter", "conv1D") << "GPU " << params->whichGPU << ": start" << std::endl;
    //printf("size = %d x %d x %d\n", params->numAngles, params->numRows, params->numCols);

    //cudaStream_t stream;
    //if (CUFFT_SUCCESS != cudaStreamCreate(&stream))
    //    return false;

    //return true;
    bool retVal = true;
    cudaError_t cudaStatus;
    cudaSetDevice(params->whichGPU);

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
    if ((cudaStatus = cudaMalloc((void**)&dev_g_pad, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H) * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_G = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_G, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H_over2) * sizeof(cufftComplex))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded projection data) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
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
        dim3 dimBlock_viewChunk = setBlockSize(dataSize);
        dim3 dimGrid_viewChunk = setGridSize(dataSize, dimBlock_viewChunk);

        for (int iChunk = 0; iChunk < numChunks; iChunk++)
        {
            int startView = iChunk * N_viewChunk;
            int endView = min(numAngles - 1, startView + N_viewChunk - 1);
            //printf("filtering %d to %d\n", startView, endView);

            dim3 dimBlock_setting(min(8, endView - startView + 1), min(8, numRows));
            dim3 dimGrid_setting(int(ceil(double(endView - startView + 1) / double(dimBlock_setting.x))), int(ceil(double(numRows) / double(dimBlock_setting.y))));

            //setPaddedDataKernel <<< endView - startView + 1, numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate, T_g, startVal_g, params->sod, helicalPitch);
            setPaddedDataKernel <<< dimGrid_setting, dimBlock_setting >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate, T_g, startVal_g, params->sod, helicalPitch);
            //cudaDeviceSynchronize();

            //cudaMemset(dev_g_pad, 0, uint64(N_viewChunk) * uint64(numRows) * uint64(N_H) * sizeof(float));
            //cudaMemcpy2D(dev_g_pad, N_H*sizeof(float), &dev_g[uint64(startView)*uint64(params->numCols*numRows)], params->numCols * sizeof(float), params->numCols * sizeof(float), numRows * (endView - startView + 1), cudaMemcpyDeviceToDevice);

            // FFT
            result = cufftExecR2C(forward_plan, (cufftReal*)dev_g_pad, dev_G);
            if (result != CUFFT_SUCCESS)
            {
                printf("cufftExecR2C failed!\n");
            }

            /*
            // Multiply Filter
            if (dev_H != 0)
                multiplyRampFilterKernel <<< N_viewChunk, numRows >>> (dev_G, dev_H, dataSize);
            else if (dev_cH != 0)
                multiplyComplexFilterKernel <<< N_viewChunk, numRows >>> (dev_G, dev_cH, dataSize);
            //*/
            //*
            // Multiply Filter
            if (dev_H != 0)
                multiplyRampFilterKernel <<< dimGrid_viewChunk, dimBlock_viewChunk >>> (dev_G, dev_H, dataSize);
            else if (dev_cH != 0)
                multiplyComplexFilterKernel <<< dimGrid_viewChunk, dimBlock_viewChunk >>> (dev_G, dev_cH, dataSize);
            //*/
            //cudaDeviceSynchronize();

            // IFFT
            result = cufftExecC2R(backward_plan, (cufftComplex*)dev_G, (cufftReal*)dev_g_pad);
            if (result != CUFFT_SUCCESS)
            {
                printf("cufftExecC2R failed!\n");
            }

            //setFilteredDataKernel <<< endView - startView + 1, numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, T_g, startVal_g, params->sod, helicalPitch);
            setFilteredDataKernel <<< dimGrid_setting, dimBlock_setting >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, T_g, startVal_g, params->sod, helicalPitch);
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

    if (retVal)
    {
        LOG(logDEBUG, "ramp_filter", "conv1D") << "GPU " << params->whichGPU << ": completed successfully" << std::endl;
    }
    else
    {
        LOG(logDEBUG, "ramp_filter", "conv1D") << "GPU " << params->whichGPU << ": completed with errors" << std::endl;
    }

    return retVal;
}

bool transmissionFilter_gpu(float*& g, parameters* params, bool data_on_cpu, float* H_full, int N_H1, int N_H2, bool isAttenuationData)
{
    float minValue = float(pow(2.0, -24.0));

    int N_x = params->numCols;
    int N_y = params->numRows;
    int N_z = params->numAngles;

    //if (N_H < max(N_x, N_y))
    if (N_H1 < N_y || N_H2 < N_x)
    {
        printf("Error: invalid filter size\n");
        return false;
    }

    // Pad and then find next power of 2
    //int N_H1 = int(pow(2.0, ceil(log2(2 * max(N_y, N_x)))));
    //int N_H1 = optimalFFTsize(2 * max(N_y, N_x));
    //int N_H1 = N_H;
    //int N_H2 = N_H1;
    int N_H2_over2 = N_H2 / 2 + 1;

    float* H = new float[N_H1 * N_H2_over2];
    for (int j = 0; j < N_H1; j++)
    {
        for (int i = 0; i < N_H2_over2; i++)
        {
            H[j * N_H2_over2 + i] = H_full[j * N_H2 + i];
        }
    }

    cudaSetDevice(params->whichGPU);
    bool retVal = true;

    float* dev_g = 0;
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

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

    //float* paddedProj = (float*)malloc(sizeof(float) * N_H1 * N_H2);
    // Make zero-padded array, copy data to 1st half of array and set remaining slots to zero
    cudaError_t cudaStatus;
    float* dev_g_pad = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_g_pad, N_H1 * N_H2 * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded volume data) failed!\n");
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_G = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_G, N_H1 * N_H2_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded volume data) failed!\n");
        retVal = false;
    }

    // Copy filter to device
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
        /*
        float* aProj = &g[uint64(k) * uint64(N_x * N_y)];
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
                if (isAttenuationData)
                    paddedProj[j * N_H2 + i] = exp(-aProj[j_source * N_x + i_source]);
                else
                    paddedProj[j * N_H2 + i] = aProj[j_source * N_x + i_source];
            }
        }
        if (cudaMemcpy(dev_g_pad, paddedProj, N_H1 * N_H2 * sizeof(float), cudaMemcpyHostToDevice))
        {
            fprintf(stderr, "cudaMemcpy(padded volume data) failed!\n");
            retVal = false;
        }
        //*/

        float* dev_proj = &dev_g[uint64(k) * uint64(params->numRows * params->numCols)];

        dim3 dimBlock_padding(8, 8);
        dim3 dimGrid_padding(int(ceil(double(N_H1) / double(dimBlock_padding.x))), int(ceil(double(N_H2) / double(dimBlock_padding.y))));
        setPaddedDataFor2DFilter <<< dimGrid_padding, dimBlock_padding >>> (dev_proj, dev_g_pad, params->numRows, params->numCols, N_H1, N_H2, isAttenuationData);

        // FFT
        result = cufftExecR2C(forward_plan, (cufftReal*)dev_g_pad, dev_G);

        // Multiply Filter
        int3 dataSize;
        dataSize.z = N_z;
        dataSize.z = 1;
        dataSize.y = N_H1;
        dataSize.x = N_H2_over2;

        dim3 dimBlock_mult = setBlockSize(dataSize);
        dim3 dimGrid_mult = setGridSize(dataSize, dimBlock_mult);

        multiply2DRampFilterKernel <<< dimGrid_mult, dimBlock_mult >>> (dev_G, dev_H, dataSize);

        // IFFT
        result = cufftExecC2R(backward_plan, (cufftComplex*)dev_G, (cufftReal*)dev_g_pad);

        // Copy result back to host
        if (retVal)
        {
            dimGrid_padding.x = int(ceil(double(params->numRows) / double(dimBlock_padding.x)));
            dimGrid_padding.y = int(ceil(double(params->numCols) / double(dimBlock_padding.x)));
            setPaddedDataFor2DFilter_reverse <<< dimGrid_padding, dimBlock_padding >>> (dev_proj, dev_g_pad, params->numRows, params->numCols, N_H1, N_H2, minValue, isAttenuationData);

            /*
            cudaStatus = cudaMemcpy(paddedProj, dev_g_pad, N_H1 * N_H2 * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
                retVal = false;
            }
            
            float* aProj = &g[uint64(k) * uint64(N_x * N_y)];
            for (int j = 0; j < N_y; j++)
            {
                for (int i = 0; i < N_x; i++)
                {
                    if (isAttenuationData)
                        aProj[j * N_x + i] = -log(max(minValue, paddedProj[j * N_H2 + i] / float(N_H1 * N_H2)));
                    else
                        aProj[j * N_x + i] = max(minValue, paddedProj[j * N_H2 + i] / float(N_H1 * N_H2));
                }
            }
            //*/
        }
    }
    cudaStatus = cudaDeviceSynchronize();

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_g_pad);
    cudaFree(dev_H);
    cudaFree(dev_G);
    //free(paddedProj);
    delete[] H;

    if (data_on_cpu)
    {
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
        if (dev_g != 0)
            cudaFree(dev_g);
    }

    return retVal;
}

bool rampFilter2D(float*& f, parameters* params, bool data_on_cpu)
{
    int N_x = params->numX;
    int N_y = params->numY;
    int N_z = params->numZ;

    float minValue = float(-1.0e30);

    // Pad and then find next power of 2
    //int N_H1 = int(pow(2.0, ceil(log2(2 * max(N_y, N_x)))));
    int N_H1 = optimalFFTsize(2 * max(N_y, N_x));
    int N_H2 = N_H1;
    int N_H2_over2 = N_H2 / 2 + 1;

    cudaSetDevice(params->whichGPU);
    bool retVal = true;

    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;

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

    //float* paddedProj = (float*)malloc(sizeof(float) * N_H1 * N_H2);
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
    int smoothingLevel = 0;
    float* H = rampFrequencyResponse2D(N_H1, params->voxelWidth, 1.0, smoothingLevel);  // FIXME?
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
        float* dev_slice = &dev_f[uint64(k) * uint64(params->numY * params->numX)];

        dim3 dimBlock_padding(8, 8);
        dim3 dimGrid_padding(int(ceil(double(N_H1) / double(dimBlock_padding.x))), int(ceil(double(N_H2) / double(dimBlock_padding.y))));
        setPaddedDataFor2DFilter <<< dimGrid_padding, dimBlock_padding >>> (dev_slice, dev_f_pad, params->numY, params->numX, N_H1, N_H2, false);

        // FFT
        result = cufftExecR2C(forward_plan, (cufftReal*)dev_f_pad, dev_F);

        // Multiply Filter
        int3 dataSize;
        dataSize.z = N_z;
        dataSize.z = 1;
        dataSize.y = N_H1;
        dataSize.x = N_H2_over2;

        dim3 dimBlock_mult = setBlockSize(dataSize);
        dim3 dimGrid_mult = setGridSize(dataSize, dimBlock_mult);

        multiply2DRampFilterKernel <<< dimGrid_mult, dimBlock_mult >>> (dev_F, dev_H, dataSize);

        // IFFT
        result = cufftExecC2R(backward_plan, (cufftComplex*)dev_F, (cufftReal*)dev_f_pad);

        // Copy result back to host
        if (retVal)
        {
            dimGrid_padding.x = int(ceil(double(params->numY) / double(dimBlock_padding.x)));
            dimGrid_padding.y = int(ceil(double(params->numX) / double(dimBlock_padding.x)));
            setPaddedDataFor2DFilter_reverse <<< dimGrid_padding, dimBlock_padding >>> (dev_slice, dev_f_pad, params->numY, params->numX, N_H1, N_H2, minValue, false);
        }
    }
    cudaStatus = cudaDeviceSynchronize();

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_f_pad);
    cudaFree(dev_H);
    cudaFree(dev_F);
    free(H);

    if (data_on_cpu)
    {
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return retVal;
}

bool rampFilter2D_XYZ(float*& f, parameters* params, bool data_on_cpu)
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
    float* H = rampFrequencyResponse2D(N_H1, params->voxelWidth, 1.0, smoothingLevel);  // FIXME?
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
        //multiply2DRampFilterKernel<<<1, 1>>>(dev_F, dev_H, dataSize);

        dataSize.z = 1;
        dim3 dimBlock_mult = setBlockSize(dataSize);
        dim3 dimGrid_mult = setGridSize(dataSize, dimBlock_mult);
        multiply2DRampFilterKernel <<< dimGrid_mult, dimBlock_mult >>> (dev_F, dev_H, dataSize);

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
#endif

bool Laplacian_gpu(float*& g, int numDims, bool smooth, parameters* params, bool data_on_cpu, float scalar)
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
    Laplacian_kernel <<< dimGrid, dimBlock >>> (dev_g, dev_Dg, N_g, T_g, startVal_g, numDims, smooth, scalar);

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
    params->colShiftFromFilter += float(-0.5*sampleShift); // opposite sign as LTT

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
    return parallelRay_derivative_chunk(g, params, data_on_cpu);

    //printf("parallelRay_derivative\n");
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    //float epsilon = float(std::min(0.01, T_g.z / (4.0 * fabs(params->T_phi()))));
    float epsilon = float(std::min(0.5, T_g.z / (4.0 * fabs(params->T_phi()))));

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
        deriv_helical_NHDLH_flat <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis, 0);
    else
        deriv_helical_NHDLH_curved <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis, 0);
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

    //printf("parallelRay_derivative done\n");

    return true;
}

bool parallelRay_derivative_chunk(float*& g, parameters* params, bool data_on_cpu)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    //float epsilon = float(std::min(0.01, T_g.z / (4.0 * fabs(params->T_phi()))));
    float epsilon = float(std::min(0.5, T_g.z / (4.0 * fabs(params->T_phi()))));
    //float epsilon = T_g.z / (4.0 * fabs(params->T_phi()));

    int maxChunkSize = 100;
    int numChunks = int(ceil(double(params->numAngles) / double(maxChunkSize)));

    float* dev_phis = copyAngleArrayToGPU(params);

    uint64 N_g_chunk_prod = uint64(maxChunkSize+2) * uint64(params->numRows) * uint64(params->numCols);
    float* dev_Dg = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_Dg, N_g_chunk_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    float* dev_interface = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_interface, uint64(params->numRows) * uint64(params->numCols) * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    float* dev_g = 0;
    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    for (int ichunk = 0; ichunk < numChunks; ichunk++)
    {
        int iphi_lo = ichunk * maxChunkSize;
        int iphi_hi = min(params->numAngles-1, iphi_lo + maxChunkSize - 1);
        //printf("iphi_lo, iphi_hi = %d, %d\n", iphi_lo, iphi_hi);

        int iphi_pad_lo = max(0, iphi_lo - 1);
        int iphi_pad_hi = min(params->numAngles - 1, iphi_hi + 1);

        int4 N_g_chunk; N_g_chunk.x = iphi_pad_hi - iphi_pad_lo + 1; N_g_chunk.y = params->numRows; N_g_chunk.z = params->numCols;

        float* dev_g_pad_chunk = &dev_g[uint64(iphi_pad_lo) * uint64(N_g_chunk.y * N_g_chunk.z)];
        float* dev_phis_chunk = &dev_phis[iphi_pad_lo];

        cudaTextureObject_t d_data_txt = NULL;
        cudaArray* d_data_array = loadTexture(d_data_txt, dev_g_pad_chunk, N_g_chunk, true, true);

        dim3 dimBlock = setBlockSize(N_g_chunk);
        dim3 dimGrid = setGridSize(N_g_chunk, dimBlock);
        if (params->detectorType == parameters::FLAT)
            deriv_helical_NHDLH_flat <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g_chunk, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis_chunk, 0);
        else
            deriv_helical_NHDLH_curved <<< dimGrid, dimBlock >>> (d_data_txt, dev_Dg, N_g_chunk, T_g, startVal_g, params->sod, params->sdd, params->tau, params->helicalPitch, epsilon, dev_phis_chunk, 0);

        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(d_data_txt);

        // Copy over what I can
        if (iphi_lo - 1 >= 0)
        {
            // should be here every time except first chunk
            // this copies over the last projection from the previous chunk
            float* dev_g_last = &dev_g[uint64(iphi_lo - 1) * uint64(N_g_chunk.y * N_g_chunk.z)];
            equal(dev_g_last, dev_interface, make_int3(1, N_g.y, N_g.z), params->whichGPU);
        }

        float* dev_g_chunk = &dev_g[uint64(iphi_lo) * uint64(N_g_chunk.y * N_g_chunk.z)];
        float* dev_Dg_shift = &dev_Dg[uint64(iphi_lo - iphi_pad_lo) * uint64(N_g_chunk.y * N_g_chunk.z)]; // first chunk shift by zero, otherwise shift by 1
        if (iphi_hi == params->numAngles - 1)
        {
            //printf("setting projections: %d to %d\n", iphi_lo, iphi_lo + iphi_hi - iphi_lo + 1-1);
            equal(dev_g_chunk, dev_Dg_shift, make_int3(iphi_hi - iphi_lo + 1, N_g.y, N_g.z), params->whichGPU);
        }
        else
        {
            //printf("setting projections: %d to %d\n", iphi_lo, iphi_lo + iphi_hi - iphi_lo - 1);
            equal(dev_g_chunk, dev_Dg_shift, make_int3(iphi_hi - iphi_lo, N_g.y, N_g.z), params->whichGPU); // does not copy last value

            if (iphi_lo == 0)
            {
                float* dev_Dg_last = &dev_Dg[uint64(iphi_hi - iphi_lo) * uint64(N_g_chunk.y * N_g_chunk.z)];
                equal(dev_interface, dev_Dg_last, make_int3(1, N_g.y, N_g.z), params->whichGPU); // save the last value
            }
            else
            {
                float* dev_Dg_last = &dev_Dg[uint64(iphi_hi - iphi_lo + 1) * uint64(N_g_chunk.y * N_g_chunk.z)];
                equal(dev_interface, dev_Dg_last, make_int3(1, N_g.y, N_g.z), params->whichGPU); // save the last value
            }
        }
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

    if (data_on_cpu == true && dev_g != 0)
        cudaFree(dev_g);

    if (dev_Dg != 0)
        cudaFree(dev_Dg);
    if (dev_interface != 0)
        cudaFree(dev_interface);
    if (dev_phis != 0)
        cudaFree(dev_phis);

    params->colShiftFromFilter += -0.5; // opposite sign as LTT
    //params->rowShiftFromFilter += -0.5;

    return true;
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

float* rampImpulseResponse_modified(int N, parameters* params)
{
    float T = params->pixelWidth;
    bool isCurved = false;
    if (params->geometry == parameters::FAN || params->geometry == parameters::CONE || params->geometry == parameters::MODULAR)
    {
        T *= params->sod / params->sdd;
        if (params->detectorType == parameters::CURVED)
            isCurved = true;
    }

    //cudaError_t cudaStatus;
    double* h_d = rampImpulseResponse(N, T, params);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = float(h_d[i]);

        if (i != 0 && isCurved == true)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= float(temp * temp);
        }
    }
    delete[] h_d;
    return h;
}

float* zeroPadForOffsetScan_GPU(float* g, parameters* params, float* g_out, bool data_on_cpu)
{
    // it is assumed that either g_out is NULL or g_out is data on the GPU
    if (g == NULL || params == NULL)
        return NULL;
    else if (params->helicalPitch != 0.0 || params->offsetScan == false)
        return NULL;
    if (params->geometry == parameters::MODULAR && params->modularbeamIsAxiallyAligned() == false)
        return NULL;

    bool padOnLeft;
    int N_add = zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);

    float* offsetScanWeights = setOffsetScanWeights(params);
    if (N_add > 0 && offsetScanWeights != NULL)
    {
        cudaSetDevice(params->whichGPU);
        cudaError_t cudaStatus;

        float* dev_g = 0;
        if (data_on_cpu)
        {
            dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
        }
        else
            dev_g = g;

        float* g_pad = g_out;
        if (g_out == NULL)
        {
            if ((cudaStatus = cudaMalloc((void**)&g_pad, params->numAngles * params->numRows * (params->numCols + N_add) * sizeof(float))) != cudaSuccess)
            {
                fprintf(stderr, "cudaMalloc(projections) failed!\n");
                return NULL;
            }
        }
        float* dev_offsetScanWeights = copy1DdataToGPU(offsetScanWeights, params->numRows * params->numCols, params->whichGPU);
        free(offsetScanWeights);

        int3 N_g = make_int3(params->numAngles, params->numRows, params->numCols + N_add);
        dim3 dimBlock = setBlockSize(N_g);
        dim3 dimGrid = setGridSize(N_g, dimBlock);

        zeroPadForOffsetScanKernel <<< dimGrid, dimBlock >>> (dev_g, g_pad, N_g, N_add, padOnLeft, dev_offsetScanWeights);
        cudaStatus = cudaDeviceSynchronize();

        if (padOnLeft)
            params->centerCol += N_add;
        params->numCols += N_add;

        if (dev_offsetScanWeights != 0)
            cudaFree(dev_offsetScanWeights);
        if (data_on_cpu == true && dev_g != 0)
            cudaFree(dev_g);
        return g_pad;
    }
    else
    {
        if (offsetScanWeights != NULL)
            free(offsetScanWeights);
        return NULL;
    }
}

#ifndef __INCLUDE_CUFFT
bool transmissionFilter_gpu(float*& g, parameters* params, bool data_on_cpu, float* H_full, int N_H1, int N_H2, bool isAttenuationData)
{
    printf("Error: 2D transmission filter cannot be run without CUFFT libraries!\n");
    return false;
}

bool rampFilter2D(float*& f, parameters* params, bool data_on_cpu)
{
    printf("Error: 2D ramp filter cannot be run without CUFFT libraries!\n");
    return false;
}

bool rampFilter2D_XYZ(float*& f, parameters* params, bool data_on_cpu)
{
    printf("Error: 2D ramp filter cannot be run without CUFFT libraries!\n");
    return false;
}

bool conv1D(float*& g, parameters* params, bool data_on_cpu, float scalar, int which, float sampleShift)
{
    //printf("This is the explicit convolution version!\n");
    // This is the explicit convolution version
    //return true;
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
    //int N_H = optimalFFTsize(2 * params->numCols);
    int N_H = 2 * params->numCols;
    int N_H_over2 = N_H / 2 + 1;

    float* h = NULL;
    if (which == 0)
    {
        h = rampImpulseResponse_modified(N_H, params);
        for (int i = 0; i < N_H; i++)
            h[i] *= scalar;
    }
    else
    {
        double* h_d = NULL;
        if (sampleShift == 0.0)
            h_d = HilbertTransformImpulseResponse(N_H, 0);
        else if (sampleShift > 0.0)
        {
            h_d = HilbertTransformImpulseResponse(N_H, 1);
            params->colShiftFromFilter -= 0.5;
        }
        else
        {
            h_d = HilbertTransformImpulseResponse(N_H, -1);
            params->colShiftFromFilter += 0.5;
        }
        h = new float[N_H];
        for (int i = 0; i < N_H; i++)
            h[i] = h_d[i] * scalar;// / float(N_H);
        delete[] h_d;
    }
    fftshift(h, N_H);

    float* dev_h = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_h, N_H * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_h, h, N_H * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(filter) failed!\n");

    cudaTextureObject_t d_h_txt = NULL;
    cudaArray* d_h_array = loadTexture1D(d_h_txt, h, N_H, false, false);

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);
    float helicalPitch = params->helicalPitch;
    if (which == 0 || params->numRows == 1)
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

    //printf("numAngles = %d\n", numAngles);
    //printf("numRows = %d\n", numRows);

    //int N_viewChunk = params->numAngles;
    int N_viewChunk = max(1, numAngles / 40); // number of views in a chunk (needs to be optimized)
    int numChunks = int(ceil(double(numAngles) / double(N_viewChunk)));

    //int3 dataSize; dataSize.x = N_viewChunk; dataSize.y = numRows; dataSize.z = N_H_over2;
    //int3 origSize; origSize.x = numAngles; origSize.y = numRows; origSize.z = params->numCols;

    int numExtrapolate = 0;
    if (params->truncatedScan)
        numExtrapolate = min(N_H - params->numCols - 1, 100);

    float* dev_g_chunk = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_g_chunk, uint64(N_viewChunk) * uint64(numRows) * uint64(params->numCols) * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        retVal = false;
    }

    if (retVal == true)
    {
        //dim3 dimBlock_viewChunk = setBlockSize(dataSize);
        //dim3 dimGrid_viewChunk = setGridSize(dataSize, dimBlock_viewChunk);

        for (int iChunk = 0; iChunk < numChunks; iChunk++)
        {
            int startView = iChunk * N_viewChunk;
            int endView = min(numAngles - 1, startView + N_viewChunk - 1);
            //printf("filtering %d to %d\n", startView, endView);

            int3 chunkSize = make_int3(endView - startView + 1, numRows, params->numCols);
            //printf("chunkSize = %d, %d, %d\n", chunkSize.x, chunkSize.y, chunkSize.z);

            equal(dev_g_chunk, &dev_g[uint64(startView)*uint64(numRows*params->numCols)], chunkSize, params->whichGPU);
            //setToConstant(&dev_g[uint64(startView) * uint64(numRows * params->numCols)], 0.0, chunkSize, params->whichGPU);
            //setPaddedDataKernel <<< endView - startView + 1, numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate, T_g, startVal_g, params->sod, helicalPitch);

            cudaTextureObject_t d_data_txt = NULL;
            cudaArray* d_data_array = loadTexture(d_data_txt, dev_g_chunk, chunkSize, false, false);

            // Perform convolution
            //dim3 dimBlock = setBlockSize(chunkSize);
            //dim3 dimGrid = setGridSize(chunkSize, dimBlock);

            int3 N_g_mod = make_int3(chunkSize.x, chunkSize.y, int(ceil(float(chunkSize.z) / float(NUM_RAYS_PER_THREAD))));
            dim3 dimBlock = setBlockSize(N_g_mod);
            dim3 dimGrid = setGridSize(N_g_mod, dimBlock);
            //dim3 dimBlock(8, 8, 8);
            //dim3 dimGrid(int(ceil(double(chunkSize.x) / double(dimBlock.x))), int(ceil(double(chunkSize.y) / double(dimBlock.y))), int(ceil(double(chunkSize.z) / double(NUM_RAYS_PER_THREAD*dimBlock.z))));

            //explicit_convolution <<< dimGrid, dimBlock >>> (dev_g_chunk , &dev_g[uint64(startView) * uint64(numRows * params->numCols)], dev_h, chunkSize, N_H);
            explicit_convolution <<< dimGrid, dimBlock >>> (d_data_txt, &dev_g[uint64(startView) * uint64(numRows * params->numCols)], d_h_txt, chunkSize, N_H);

            cudaFreeArray(d_data_array);
            cudaDestroyTextureObject(d_data_txt);
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
    cudaFreeArray(d_h_array);
    cudaDestroyTextureObject(d_h_txt);
    //cudaFree(dev_g_pad);
    if (data_on_cpu)
        cudaFree(dev_g);
    cudaFree(dev_h);
    cudaFree(dev_g_chunk);
    delete[] h;

    return retVal;
}
#endif
