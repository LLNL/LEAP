////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for anisotropic Total Variation (TV)
////////////////////////////////////////////////////////////////////////////////
#include "total_variation.cuh"

#include <math.h>

#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "cpu_utils.h"
//#include "device_launch_parameters.h"

// Commenting out this define will cause LEAP to use a Huber loss function
// of power p=1 for TV denoising
// Leaving this define will use a Huber-like loss function of power 1.2
#define USE_P_HUBER

__constant__ float d_HUBER_P;
__constant__ float d_HUBER_DELTA;
__constant__ float d_HUBER_SLOPE;
__constant__ float d_HUBER_SHIFT;

__forceinline__ __device__ float square(const float x)
{
    return x * x;
}

__forceinline__ __device__ float Huber(const float x)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= d_HUBER_DELTA)
        return 0.5f * x * x;
    else
        return d_HUBER_SLOPE * pow(fabs(x), d_HUBER_P) + d_HUBER_SHIFT;
        //return delta * delta * (5.0f * pow(fabs(x / delta), 1.2f) - 2.0f) / 6.0f;
#else
    if (fabs(x) <= d_HUBER_DELTA)
        return 0.5f * x * x;
    else
        return d_HUBER_DELTA * (fabs(x) - 0.5f * d_HUBER_DELTA);
#endif
}

__forceinline__ __device__ float DHuber(const float x)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= d_HUBER_DELTA)
        return x;
    else
        return d_HUBER_P * d_HUBER_SLOPE * pow(fabs(x), d_HUBER_P) / x;
        //return x * pow(fabs(x / delta), -0.8f); // p=1.2
        //return delta * pow(fabs(x/delta), 0.2f);
#else
    if (fabs(x) <= d_HUBER_DELTA)
        return x;
    else
        return (x > 0.0f) ? d_HUBER_DELTA : -d_HUBER_DELTA;
#endif
}

__forceinline__ __device__ float DDHuber(const float x)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= d_HUBER_DELTA)
        return  1.0;
    else
        return d_HUBER_P * d_HUBER_SLOPE * pow(fabs(x), d_HUBER_P - 2.0f);
        //return pow(fabs(x / delta), d_HUBER_P - 2.0f); // require a divide
        //return pow(fabs(x / delta), -0.8f); // p=1.2
#else
    return d_HUBER_DELTA / max(d_HUBER_DELTA, fabs(x));
#endif
}

__device__ float aTV_Huber_costTerm(float* f, const int i, const int j, const int k, int3 N, float delta, float beta, int numNeighbors)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    float* f_i = &f[uint64(i) * uint64(N.y * N.z)];
    

    //*
    const float curVal = f_i[j * N.z + k];

    if (N.x == 1)
    {
        if (numNeighbors == 6)
        {
            return (Huber(curVal - f_i[j_plus * N.z + k]) +
                Huber(curVal - f_i[j_minus * N.z + k]) +
                Huber(curVal - f_i[j * N.z + k_plus]) +
                Huber(curVal - f_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            return (Huber(curVal - f_i[j_plus * N.z + k]) +
                Huber(curVal - f_i[j_minus * N.z + k]) +
                Huber(curVal - f_i[j * N.z + k_plus]) +
                Huber(curVal - f_i[j * N.z + k_minus])) * dist_1 +
                (Huber(curVal - f_i[j_plus * N.z + k_plus]) +
                    Huber(curVal - f_i[j_plus * N.z + k_minus]) +
                    Huber(curVal - f_i[j_minus * N.z + k_plus]) +
                    Huber(curVal - f_i[j_minus * N.z + k_minus])) * dist_2;
        }
    }
    else
    {
        float* f_i_minus = &f[uint64(i_minus) * uint64(N.y * N.z)];
        float* f_i_plus = &f[uint64(i_plus) * uint64(N.y * N.z)];
        if (numNeighbors == 6)
        {
            return (Huber(curVal - f_i_plus[j * N.z + k]) +
                Huber(curVal - f_i_minus[j * N.z + k]) +
                Huber(curVal - f_i[j_plus * N.z + k]) +
                Huber(curVal - f_i[j_minus * N.z + k]) +
                Huber(curVal - f_i[j * N.z + k_plus]) +
                Huber(curVal - f_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            return (Huber(curVal - f_i_plus[j * N.z + k]) +
                Huber(curVal - f_i_minus[j * N.z + k]) +
                Huber(curVal - f_i[j_plus * N.z + k]) +
                Huber(curVal - f_i[j_minus * N.z + k]) +
                Huber(curVal - f_i[j * N.z + k_plus]) +
                Huber(curVal - f_i[j * N.z + k_minus])) *
                dist_1 +
                (Huber(curVal - f_i_plus[j_plus * N.z + k]) +
                    Huber(curVal - f_i_plus[j_minus * N.z + k]) +
                    Huber(curVal - f_i_plus[j * N.z + k_plus]) +
                    Huber(curVal - f_i_plus[j * N.z + k_minus]) +
                    Huber(curVal - f_i_minus[j_plus * N.z + k]) +
                    Huber(curVal - f_i_minus[j_minus * N.z + k]) +
                    Huber(curVal - f_i_minus[j * N.z + k_plus]) +
                    Huber(curVal - f_i_minus[j * N.z + k_minus]) +
                    Huber(curVal - f_i[j_plus * N.z + k_plus]) +
                    Huber(curVal - f_i[j_plus * N.z + k_minus]) +
                    Huber(curVal - f_i[j_minus * N.z + k_plus]) +
                    Huber(curVal - f_i[j_minus * N.z + k_minus])) *
                dist_2 +
                (Huber(curVal - f_i_plus[j_plus * N.z + k_plus]) +
                    Huber(curVal - f_i_plus[j_plus * N.z + k_minus]) +
                    Huber(curVal - f_i_plus[j_minus * N.z + k_plus]) +
                    Huber(curVal - f_i_plus[j_minus * N.z + k_minus]) +
                    Huber(curVal - f_i_minus[j_plus * N.z + k_plus]) +
                    Huber(curVal - f_i_minus[j_plus * N.z + k_minus]) +
                    Huber(curVal - f_i_minus[j_minus * N.z + k_plus]) +
                    Huber(curVal - f_i_minus[j_minus * N.z + k_minus])) *
                dist_3;
        }
    }
    //*/

    /*
    const float curVal = f[i * N.y * N.z + j * N.z + k];

    return (Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) +
        Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) +
        Huber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta)) *
        dist_1 +
        (Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.z + j_plus * N.z + k], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_2 +
        (Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_3;
    //*/
}

__global__ void aTV_Huber_cost(float* f, float* d, int3 N, float delta, float beta, int sliceStart, int sliceEnd, int numNeighbors)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        d[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    d[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = aTV_Huber_costTerm(f, i, j, k, N, delta, beta, numNeighbors);
}

__device__ float aTV_Huber_quadFormTerm(const float* f, const float* d, const int i, const int j, const int k, int3 N, float delta, float beta, int numNeighbors)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float* f_i = &f[uint64(i) * uint64(N.y * N.z)];
    const float* d_i = &d[uint64(i) * uint64(N.y * N.z)];

    //*
    const float curVal = f_i[j * N.z + k];
    const float curVal_d = d_i[j * N.z + k];

    if (N.x == 1)
    {
        if (numNeighbors == 6)
        {
            return (DDHuber(curVal - f_i[j_plus * N.z + k]) *
                square(curVal_d - d_i[j_plus * N.z + k]) +
                DDHuber(curVal - f_i[j_minus * N.z + k]) *
                square(curVal_d - d_i[j_minus * N.z + k]) +
                DDHuber(curVal - f_i[j * N.z + k_plus]) *
                square(curVal_d - d_i[j * N.z + k_plus]) +
                DDHuber(curVal - f_i[j * N.z + k_minus]) *
                square(curVal_d - d_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            return (DDHuber(curVal - f_i[j_plus * N.z + k]) *
                square(curVal_d - d_i[j_plus * N.z + k]) +
                DDHuber(curVal - f_i[j_minus * N.z + k]) *
                square(curVal_d - d_i[j_minus * N.z + k]) +
                DDHuber(curVal - f_i[j * N.z + k_plus]) *
                square(curVal_d - d_i[j * N.z + k_plus]) +
                DDHuber(curVal - f_i[j * N.z + k_minus]) *
                square(curVal_d - d_i[j * N.z + k_minus])) *
                dist_1 +
                (DDHuber(curVal - f_i[j_plus * N.z + k_plus]) *
                    square(curVal_d - d_i[j_plus * N.z + k_plus]) +
                    DDHuber(curVal - f_i[j_plus * N.z + k_minus]) *
                    square(curVal_d - d_i[j_plus * N.z + k_minus]) +
                    DDHuber(curVal - f_i[j_minus * N.z + k_plus]) *
                    square(curVal_d - d_i[j_minus * N.z + k_plus]) +
                    DDHuber(curVal - f_i[j_minus * N.z + k_minus]) *
                    square(curVal_d - d_i[j_minus * N.z + k_minus])) *
                dist_2;
        }
    }
    else
    {
        const float* f_i_minus = &f[uint64(i_minus) * uint64(N.y * N.z)];
        const float* f_i_plus = &f[uint64(i_plus) * uint64(N.y * N.z)];

        const float* d_i_minus = &d[uint64(i_minus) * uint64(N.y * N.z)];
        const float* d_i_plus = &d[uint64(i_plus) * uint64(N.y * N.z)];
        if (numNeighbors == 6)
        {
            return (DDHuber(curVal - f_i_plus[j * N.z + k]) *
                square(curVal_d - d_i_plus[j * N.z + k]) +
                DDHuber(curVal - f_i_minus[j * N.z + k]) *
                square(curVal_d - d_i_minus[j * N.z + k]) +
                DDHuber(curVal - f_i[j_plus * N.z + k]) *
                square(curVal_d - d_i[j_plus * N.z + k]) +
                DDHuber(curVal - f_i[j_minus * N.z + k]) *
                square(curVal_d - d_i[j_minus * N.z + k]) +
                DDHuber(curVal - f_i[j * N.z + k_plus]) *
                square(curVal_d - d_i[j * N.z + k_plus]) +
                DDHuber(curVal - f_i[j * N.z + k_minus]) *
                square(curVal_d - d_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            return (DDHuber(curVal - f_i_plus[j * N.z + k]) *
                square(curVal_d - d_i_plus[j * N.z + k]) +
                DDHuber(curVal - f_i_minus[j * N.z + k]) *
                square(curVal_d - d_i_minus[j * N.z + k]) +
                DDHuber(curVal - f_i[j_plus * N.z + k]) *
                square(curVal_d - d_i[j_plus * N.z + k]) +
                DDHuber(curVal - f_i[j_minus * N.z + k]) *
                square(curVal_d - d_i[j_minus * N.z + k]) +
                DDHuber(curVal - f_i[j * N.z + k_plus]) *
                square(curVal_d - d_i[j * N.z + k_plus]) +
                DDHuber(curVal - f_i[j * N.z + k_minus]) *
                square(curVal_d - d_i[j * N.z + k_minus])) *
                dist_1 +
                (DDHuber(curVal - f_i_plus[j_plus * N.z + k]) *
                    square(curVal_d - d_i_plus[j_plus * N.z + k]) +
                    DDHuber(curVal - f_i_plus[j_minus * N.z + k]) *
                    square(curVal_d - d_i_plus[j_minus * N.z + k]) +
                    DDHuber(curVal - f_i_plus[j * N.z + k_plus]) *
                    square(curVal_d - d_i_plus[j * N.z + k_plus]) +
                    DDHuber(curVal - f_i_plus[j * N.z + k_minus]) *
                    square(curVal_d - d_i_plus[j * N.z + k_minus]) +
                    DDHuber(curVal - f_i_minus[j_plus * N.z + k]) *
                    square(curVal_d - d_i_minus[j_plus * N.z + k]) +
                    DDHuber(curVal - f_i_minus[j_minus * N.z + k]) *
                    square(curVal_d - d_i_minus[j_minus * N.z + k]) +
                    DDHuber(curVal - f_i_minus[j * N.z + k_plus]) *
                    square(curVal_d - d_i_minus[j * N.z + k_plus]) +
                    DDHuber(curVal - f_i_minus[j * N.z + k_minus]) *
                    square(curVal_d - d_i_minus[j * N.z + k_minus]) +
                    DDHuber(curVal - f_i[j_plus * N.z + k_plus]) *
                    square(curVal_d - d_i[j_plus * N.z + k_plus]) +
                    DDHuber(curVal - f_i[j_plus * N.z + k_minus]) *
                    square(curVal_d - d_i[j_plus * N.z + k_minus]) +
                    DDHuber(curVal - f_i[j_minus * N.z + k_plus]) *
                    square(curVal_d - d_i[j_minus * N.z + k_plus]) +
                    DDHuber(curVal - f_i[j_minus * N.z + k_minus]) *
                    square(curVal_d - d_i[j_minus * N.z + k_minus])) *
                dist_2 +
                (DDHuber(curVal - f_i_plus[j_plus * N.z + k_plus]) *
                    square(curVal_d - d_i_plus[j_plus * N.z + k_plus]) +
                    DDHuber(curVal - f_i_plus[j_plus * N.z + k_minus]) *
                    square(curVal_d - d_i_plus[j_plus * N.z + k_minus]) +
                    DDHuber(curVal - f_i_plus[j_minus * N.z + k_plus]) *
                    square(curVal_d - d_i_plus[j_minus * N.z + k_plus]) +
                    DDHuber(curVal - f_i_plus[j_minus * N.z + k_minus]) *
                    square(curVal_d - d_i_plus[j_minus * N.z + k_minus]) +
                    DDHuber(curVal - f_i_minus[j_plus * N.z + k_plus]) *
                    square(curVal_d - d_i_minus[j_plus * N.z + k_plus]) +
                    DDHuber(curVal - f_i_minus[j_plus * N.z + k_minus]) *
                    square(curVal_d - d_i_minus[j_plus * N.z + k_minus]) +
                    DDHuber(curVal - f_i_minus[j_minus * N.z + k_plus]) *
                    square(curVal_d - d_i_minus[j_minus * N.z + k_plus]) +
                    DDHuber(curVal - f_i_minus[j_minus * N.z + k_minus]) *
                    square(curVal_d - d_i_minus[j_minus * N.z + k_minus])) *
                dist_3;
        }
    }
    //*/

    /*
    const float curVal = f[i * N.y * N.z + j * N.z + k];
    const float curVal_d = d[i * N.y * N.z + j * N.z + k];

    return (DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) *
        square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k]) +
        DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) *
        square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) *
        square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) *
        square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) *
        square(curVal_d - d[i * N.y * N.z + j * N.z + k_plus]) +
        DDHuber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta) *
        square(curVal_d - d[i * N.y * N.z + j * N.z + k_minus])) *
        dist_1 +
        (DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k_minus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k_minus])) *
        dist_2 +
        (DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k_minus])) *
        dist_3;
    //*/
}

__global__ void aTV_Huber_quadForm_reduce(const float* __restrict__ f,
    const float* __restrict__ d,
    float* result,
    int3 N,
    float delta,
    float beta,
    int sliceStart,
    int sliceEnd,
    int numNeighbors)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    extern __shared__ float sdata[];

    float val = 0.0f;

    if (i < N.x && j < N.y && k < N.z && i >= sliceStart && i <= sliceEnd)
        val = aTV_Huber_quadFormTerm((float*)f, (float*)d, i, j, k, N, delta, beta, numNeighbors);

    sdata[tid] = val;
    __syncthreads();

    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    for (int stride = blockSize / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

__global__ void aTV_Huber_quadForm(float* f, float* d, float* quad, int3 N, float delta, float beta, int sliceStart, int sliceEnd, int numNeighbors)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        quad[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    quad[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = aTV_Huber_quadFormTerm(f, d, i, j, k, N, delta, beta, numNeighbors);
}

__global__ void aTV_Huber_gradient(float* f, float* Df, int3 N, float delta, float beta, int sliceStart, int sliceEnd, int numNeighbors)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        Df[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    float* f_i = &f[uint64(i) * uint64(N.y * N.z)];

    //*
    const float curVal = f_i[j * N.z + k];

    // dist 1: 6
    // dist 2: 12
    // dist 3: 8
    if (N.x == 1)
    {
        if (numNeighbors == 6)
        {
            Df[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = (DHuber(curVal - f_i[j_plus * N.z + k]) +
                DHuber(curVal - f_i[j_minus * N.z + k]) +
                DHuber(curVal - f_i[j * N.z + k_plus]) +
                DHuber(curVal - f_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            Df[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = (DHuber(curVal - f_i[j_plus * N.z + k]) +
                DHuber(curVal - f_i[j_minus * N.z + k]) +
                DHuber(curVal - f_i[j * N.z + k_plus]) +
                DHuber(curVal - f_i[j * N.z + k_minus])) *
                dist_1 +
                (DHuber(curVal - f_i[j_plus * N.z + k_plus]) +
                    DHuber(curVal - f_i[j_plus * N.z + k_minus]) +
                    DHuber(curVal - f_i[j_minus * N.z + k_plus]) +
                    DHuber(curVal - f_i[j_minus * N.z + k_minus])) *
                dist_2;
        }
    }
    else
    {
        float* f_i_minus = &f[uint64(i_minus) * uint64(N.y * N.z)];
        float* f_i_plus = &f[uint64(i_plus) * uint64(N.y * N.z)];
        if (numNeighbors == 6)
        {
            Df[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = (DHuber(curVal - f_i_plus[j * N.z + k]) +
                DHuber(curVal - f_i_minus[j * N.z + k]) +
                DHuber(curVal - f_i[j_plus * N.z + k]) +
                DHuber(curVal - f_i[j_minus * N.z + k]) +
                DHuber(curVal - f_i[j * N.z + k_plus]) +
                DHuber(curVal - f_i[j * N.z + k_minus])) * dist_1;
        }
        else
        {
            Df[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = (DHuber(curVal - f_i_plus[j * N.z + k]) +
                DHuber(curVal - f_i_minus[j * N.z + k]) +
                DHuber(curVal - f_i[j_plus * N.z + k]) +
                DHuber(curVal - f_i[j_minus * N.z + k]) +
                DHuber(curVal - f_i[j * N.z + k_plus]) +
                DHuber(curVal - f_i[j * N.z + k_minus])) *
                dist_1 +
                (DHuber(curVal - f_i_plus[j_plus * N.z + k]) +
                    DHuber(curVal - f_i_plus[j_minus * N.z + k]) +
                    DHuber(curVal - f_i_plus[j * N.z + k_plus]) +
                    DHuber(curVal - f_i_plus[j * N.z + k_minus]) +
                    DHuber(curVal - f_i_minus[j_plus * N.z + k]) +
                    DHuber(curVal - f_i_minus[j_minus * N.z + k]) +
                    DHuber(curVal - f_i_minus[j * N.z + k_plus]) +
                    DHuber(curVal - f_i_minus[j * N.z + k_minus]) +
                    DHuber(curVal - f_i[j_plus * N.z + k_plus]) +
                    DHuber(curVal - f_i[j_plus * N.z + k_minus]) +
                    DHuber(curVal - f_i[j_minus * N.z + k_plus]) +
                    DHuber(curVal - f_i[j_minus * N.z + k_minus])) *
                dist_2 +
                (DHuber(curVal - f_i_plus[j_plus * N.z + k_plus]) +
                    DHuber(curVal - f_i_plus[j_plus * N.z + k_minus]) +
                    DHuber(curVal - f_i_plus[j_minus * N.z + k_plus]) +
                    DHuber(curVal - f_i_plus[j_minus * N.z + k_minus]) +
                    DHuber(curVal - f_i_minus[j_plus * N.z + k_plus]) +
                    DHuber(curVal - f_i_minus[j_plus * N.z + k_minus]) +
                    DHuber(curVal - f_i_minus[j_minus * N.z + k_plus]) +
                    DHuber(curVal - f_i_minus[j_minus * N.z + k_minus])) *
                dist_3;
        }
    }
    //*/

    /*
    const float curVal = f[i * N.y * N.z + j * N.z + k];

    // dist 1: 6
    // dist 2: 12
    // dist 3: 8
    Df[i * N.y * N.z + j * N.z + k] = (DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) +
        DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) +
        DHuber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta)) *
        dist_1 +
        (DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_2 +
        (DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_3;
    //*/
}

void setConstantMemoryParameters(const float delta, const float p)
{
    const float HuberSlope = float(pow(delta, 2.0 - p) / p);
    //const float HuberShift = float((0.5 * p - 1.0) * pow(delta, p));
    const float HuberShift = float(delta*delta*(0.5 - 1.0/p));

    //d_HUBER_SLOPE * pow(fabs(x), d_HUBER_P) + d_HUBER_SHIFT

    cudaMemcpyToSymbol(d_HUBER_P, &p, sizeof(float));
    cudaMemcpyToSymbol(d_HUBER_DELTA, &delta, sizeof(float));
    cudaMemcpyToSymbol(d_HUBER_SLOPE, &HuberSlope, sizeof(float));
    cudaMemcpyToSymbol(d_HUBER_SHIFT, &HuberShift, sizeof(float));
}

bool anisotropicTotalVariation_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, int numNeighbors, bool doMean)
{
    if (f == NULL) return false;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    setConstantMemoryParameters(delta, p);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    float* dev_Df = 0;
    if (data_on_cpu)
    {
        dev_f = copy3DdataToGPU(f, N, whichGPU);
        
        if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
            return false;
        }
    }
    else
    {
        dev_f = f;
        dev_Df = Df;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_gradient <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, delta, beta, sliceStart, sliceEnd, numNeighbors);
    if (doMean)
    {
        //printf("doing mean\n");
        mean_over_slices(dev_Df, N, whichGPU);
    }
    cudaDeviceSynchronize();

    // pull result off GPU
    if (data_on_cpu)
    {
        float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
        int3 N_crop = make_int3(sliceEnd-sliceStart+1, N_2, N_3);
        pull3DdataFromGPU(Df, N_crop, dev_Df_shift, whichGPU);
    }

    // Clean up
    if (data_on_cpu && dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (data_on_cpu && dev_Df != 0)
    {
        cudaFree(dev_Df);
    }
    return true;
}

float anisotropicTotalVariation_quadraticForm(float* f, float* d, int N_1, int N_2, int N_3,
    float delta, float beta, float p, bool data_on_cpu,
    int whichGPU, int sliceStart, int sliceEnd,
    int numNeighbors)
{
    if (f == NULL || d == NULL) return -1.0;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return -1.0;

    cudaSetDevice(whichGPU);

    setConstantMemoryParameters(delta, p);

    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    float* dev_d = 0;
    if (data_on_cpu)
    {
        dev_f = copy3DdataToGPU(f, N, whichGPU);
        dev_d = copy3DdataToGPU(d, N, whichGPU);
    }
    else
    {
        dev_f = f;
        dev_d = d;
    }

    float* dev_result = 0;
    if (cudaMalloc((void**)&dev_result, sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(dev_result) failed!\n");
        if (data_on_cpu && dev_f != 0) cudaFree(dev_f);
        if (data_on_cpu && dev_d != 0) cudaFree(dev_d);
        return -1.0;
    }
    cudaMemset(dev_result, 0, sizeof(float));

    dim3 dimBlock = setBlockSize(N);
    if (dimBlock.x == 1)
        dimBlock.x = 2;
    if (dimBlock.y == 1)
        dimBlock.y = 2;
    if (dimBlock.z == 1)
        dimBlock.z = 2;
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))), int(ceil(double(N.z) / double(dimBlock.z))));

    const int threadsPerBlock = dimBlock.x * dimBlock.y * dimBlock.z;
    const size_t sharedMemBytes = threadsPerBlock * sizeof(float);

    aTV_Huber_quadForm_reduce<<<dimGrid, dimBlock, sharedMemBytes>>>(dev_f, dev_d, dev_result, N, delta, beta, sliceStart, sliceEnd, numNeighbors);
    cudaDeviceSynchronize();

    float retVal = 0.0f;
    cudaMemcpy(&retVal, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    if (data_on_cpu && dev_f != 0)
        cudaFree(dev_f);
    if (data_on_cpu && dev_d != 0)
        cudaFree(dev_d);
    if (dev_result != 0)
        cudaFree(dev_result);

    return retVal;
}

/*
float anisotropicTotalVariation_quadraticForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, int numNeighbors)
{
    if (f == NULL || d == NULL) return -1.0;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    setConstantMemoryParameters(delta, p);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    float* dev_d = 0;
    if (data_on_cpu)
    {
        dev_f = copy3DdataToGPU(f, N, whichGPU);
        dev_d = copy3DdataToGPU(d, N, whichGPU);
    }
    else
    {
        dev_f = f;
        dev_d = d;
    }

    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_quad = 0;
    if (cudaMalloc((void**)&dev_quad, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_quadForm <<< dimGrid, dimBlock >>> (dev_f, dev_d, dev_quad, N, delta, beta, sliceStart, sliceEnd, numNeighbors);
    cudaDeviceSynchronize();

    float retVal = sum(dev_quad, N, whichGPU);

    // Clean up
    if (data_on_cpu && dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (data_on_cpu && dev_d != 0)
    {
        cudaFree(dev_d);
    }
    if (dev_quad != 0)
    {
        cudaFree(dev_quad);
    }

    return retVal;
}
//*/

float anisotropicTotalVariation_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, int numNeighbors)
{
    if (f == NULL) return -1.0;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    setConstantMemoryParameters(delta, p);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_d = 0;
    if (cudaMalloc((void**)&dev_d, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_cost <<< dimGrid, dimBlock >>> (dev_f, dev_d, N, delta, beta, sliceStart, sliceEnd, numNeighbors);
    cudaDeviceSynchronize();

    /*
    float* temp = (float*) malloc(size_t(uint64(N.x) * uint64(N.y) * uint64(N.z)) * sizeof(float));
    pull3DdataFromGPU(temp, N, dev_d, whichGPU);
    float retVal = sum_cpu(temp, N_1, N_2, N_3);
    free(temp);
    //*/

    float retVal = sum(dev_d, N, whichGPU);

    // Clean up
    if (data_on_cpu && dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (dev_d != 0)
    {
        cudaFree(dev_d);
    }

    return retVal;
}

bool diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu, int whichGPU, int numNeighbors)
{
    if (f == NULL) return false;
    float beta = 1.0;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    //setConstantMemoryParameters(delta, p);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    float* dev_d = 0;
    if (cudaMalloc((void**)&dev_d, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return false;
    }

    for (int n = 0; n < numIter; n++)
    {
        anisotropicTotalVariation_gradient(dev_f, dev_d, N_1, N_2, N_3, delta, beta, p, false, whichGPU, -1, -1, numNeighbors);
        float num = innerProduct(dev_d, dev_d, N, whichGPU);
        float denom = anisotropicTotalVariation_quadraticForm(dev_f, dev_d, N_1, N_2, N_3, delta, beta, p, false, whichGPU, -1, -1, numNeighbors);
        if (denom <= 1.0e-16)
            break;
        float stepSize = num / denom;
        scalarAdd(dev_f, -stepSize, dev_d, N, whichGPU);

        //printf("cost = %f\n", anisotropicTotalVariation_cost(dev_f, N_1, N_2, N_3, delta, beta, false, whichGPU, numNeighbors));
    }

    // pull result off GPU
    if (data_on_cpu)
        pull3DdataFromGPU(f, N, dev_f, whichGPU);

    // Clean up
    if (data_on_cpu && dev_f != 0)
    {
        cudaFree(dev_f);
    }

    if (dev_d != 0)
    {
        cudaFree(dev_d);
    }
    return true;
}


__global__ void copy_linear_slice_to_buffer(const float* __restrict__ f,
                                            float* __restrict__ slice,
                                            const int nx,
                                            const int ny,
                                            const int nz,
                                            const int k)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny)
        return;

    // Slices are stacked along the slowest-varying axis (nz).
    // Within a slice the fast (contiguous) axis is nx and the slow axis is ny.
    slice[j * nx + i] = f[uint64(k) * uint64(nx * ny) + uint64(j * nx + i)];
}

__global__ void regularize_slice_ringbuffer_linear(float* __restrict__ f,
                                                   const float* __restrict__ slice_prev,
                                                   const float* __restrict__ slice_cur,
                                                   const float* __restrict__ slice_next,
                                                   const int nx,
                                                   const int ny,
                                                   const int nz,
                                                   const int k,
                                                   const float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny)
        return;

    const int i_minus = max(0, i - 1);
    const int i_plus  = min(nx - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus  = min(ny - 1, j + 1);
    const bool has_prev = (k > 0);
    const bool has_next = (k + 1 < nz);

    const float dist_2 = 0.7071067811865475f;
    const float dist_3 = 0.5773502691896258f;

    const float curVal = slice_cur[j * nx + i];
    float smoothing_term = 0.0f;

    smoothing_term += DHuber(curVal - slice_cur[j * nx + i_plus]);
    smoothing_term += DHuber(curVal - slice_cur[j * nx + i_minus]);
    smoothing_term += DHuber(curVal - slice_cur[j_plus * nx + i]);
    smoothing_term += DHuber(curVal - slice_cur[j_minus * nx + i]);
    if (has_next) smoothing_term += DHuber(curVal - slice_next[j * nx + i]);
    if (has_prev) smoothing_term += DHuber(curVal - slice_prev[j * nx + i]);

    smoothing_term += dist_2 * DHuber(curVal - slice_cur[j_plus * nx + i_plus]);
    smoothing_term += dist_2 * DHuber(curVal - slice_cur[j_minus * nx + i_plus]);
    smoothing_term += dist_2 * DHuber(curVal - slice_cur[j_plus * nx + i_minus]);
    smoothing_term += dist_2 * DHuber(curVal - slice_cur[j_minus * nx + i_minus]);

    if (has_next) {
        smoothing_term += dist_2 * DHuber(curVal - slice_next[j * nx + i_plus]);
        smoothing_term += dist_2 * DHuber(curVal - slice_next[j * nx + i_minus]);
        smoothing_term += dist_2 * DHuber(curVal - slice_next[j_plus * nx + i]);
        smoothing_term += dist_2 * DHuber(curVal - slice_next[j_minus * nx + i]);
    }
    if (has_prev) {
        smoothing_term += dist_2 * DHuber(curVal - slice_prev[j * nx + i_plus]);
        smoothing_term += dist_2 * DHuber(curVal - slice_prev[j * nx + i_minus]);
        smoothing_term += dist_2 * DHuber(curVal - slice_prev[j_plus * nx + i]);
        smoothing_term += dist_2 * DHuber(curVal - slice_prev[j_minus * nx + i]);
    }

    if (has_next) {
        smoothing_term += dist_3 * DHuber(curVal - slice_next[j_plus * nx + i_plus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_next[j_minus * nx + i_plus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_next[j_plus * nx + i_minus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_next[j_minus * nx + i_minus]);
    }
    if (has_prev) {
        smoothing_term += dist_3 * DHuber(curVal - slice_prev[j_plus * nx + i_plus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_prev[j_minus * nx + i_plus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_prev[j_plus * nx + i_minus]);
        smoothing_term += dist_3 * DHuber(curVal - slice_prev[j_minus * nx + i_minus]);
    }

    //const float updated = fmaxf(0.0f, curVal - beta * smoothing_term);
    const float updated = curVal - beta * smoothing_term;
    f[uint64(k) * uint64(nx * ny) + uint64(j * nx + i)] = updated;
}

bool diffuse_sqs(float* f, int N_1, int N_2, int N_3, float beta, float delta, float p, int numIter, bool data_on_cpu, int whichGPU, int numNeighbors)
{
    if (f == NULL) return false;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);
    float total_weight = 2.0*(6.0 + 8.0/sqrt(3.0) + 12.0/sqrt(2.0));
    beta = min(2.0, beta) / total_weight;

    cudaSetDevice(whichGPU);
    setConstantMemoryParameters(delta, p);

    // Copy volume to GPU.  Linear layout is the same as the rest of this file:
    // f[i * N.y * N.z + j * N.z + k].
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // N_1, N_2, N_3 are generic names for the volume dimensions numZ, numY, numX.
    // Slices are taken along the slowest-varying axis (numZ = N_1) and each slice
    // is a numX-by-numY plane with numX (= N_3) as the fast, contiguous in-plane axis.
    const int nx = N_3;  // numX: fastest (contiguous) in-plane axis
    const int ny = N_2;  // numY: slow in-plane axis
    const int nz = N_1;  // numZ: slice (stacking) axis

    const size_t slice_bytes = size_t(nx) * size_t(ny) * sizeof(float);
    float* d_slice_ring[3] = { nullptr, nullptr, nullptr };
    if (cudaMalloc(&d_slice_ring[0], slice_bytes) != cudaSuccess ||
        cudaMalloc(&d_slice_ring[1], slice_bytes) != cudaSuccess ||
        cudaMalloc(&d_slice_ring[2], slice_bytes) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(slice ring) failed!\n");
        if (d_slice_ring[0] != nullptr) cudaFree(d_slice_ring[0]);
        if (d_slice_ring[1] != nullptr) cudaFree(d_slice_ring[1]);
        if (d_slice_ring[2] != nullptr) cudaFree(d_slice_ring[2]);
        if (data_on_cpu && dev_f != 0) cudaFree(dev_f);
        return false;
    }

    dim3 dimBlock_reg(16, 16, 1);
    dim3 dimGrid_reg((nx + dimBlock_reg.x - 1) / dimBlock_reg.x,
                     (ny + dimBlock_reg.y - 1) / dimBlock_reg.y,
                     1);

    for (int n = 0; n < numIter; n++)
    {
        copy_linear_slice_to_buffer<<<dimGrid_reg, dimBlock_reg>>>(dev_f, d_slice_ring[0], nx, ny, nz, 0);
        if (nz > 1)
            copy_linear_slice_to_buffer<<<dimGrid_reg, dimBlock_reg>>>(dev_f, d_slice_ring[1], nx, ny, nz, 1);

        for (int k = 0; k < nz; k++)
        {
            if (k + 1 < nz && k > 0)
                copy_linear_slice_to_buffer<<<dimGrid_reg, dimBlock_reg>>>(dev_f, d_slice_ring[(k + 1) % 3], nx, ny, nz, k + 1);

            const float* slice_prev = (k > 0) ? d_slice_ring[(k + 2) % 3] : nullptr;
            const float* slice_cur  = d_slice_ring[k % 3];
            const float* slice_next = (k + 1 < nz) ? d_slice_ring[(k + 1) % 3] : nullptr;

            regularize_slice_ringbuffer_linear<<<dimGrid_reg, dimBlock_reg>>>(dev_f,
                                                                              slice_prev,
                                                                              slice_cur,
                                                                              slice_next,
                                                                              nx,
                                                                              ny,
                                                                              nz,
                                                                              k,
                                                                              beta);
        }
    }
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "diffuse_sqs failed: %s\n", cudaGetErrorString(cudaStatus));
        if (d_slice_ring[0] != nullptr) cudaFree(d_slice_ring[0]);
        if (d_slice_ring[1] != nullptr) cudaFree(d_slice_ring[1]);
        if (d_slice_ring[2] != nullptr) cudaFree(d_slice_ring[2]);
        if (data_on_cpu && dev_f != 0) cudaFree(dev_f);
        return false;
    }

    // pull result off GPU
    if (data_on_cpu)
        pull3DdataFromGPU(f, N, dev_f, whichGPU);

    // Clean up
    if (d_slice_ring[0] != nullptr) cudaFree(d_slice_ring[0]);
    if (d_slice_ring[1] != nullptr) cudaFree(d_slice_ring[1]);
    if (d_slice_ring[2] != nullptr) cudaFree(d_slice_ring[2]);
    if (data_on_cpu && dev_f != 0) cudaFree(dev_f);

    return true;
}

bool TVdenoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool data_on_cpu, int whichGPU, int numNeighbors, bool doMean)
{
    if (f == NULL) return false;
    if (delta < 1.0e-8)
        delta = float(1.0e-8);

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    //setConstantMemoryParameters(delta, p);

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    float* dev_f_0 = 0;
    if (cudaMalloc((void**)&dev_f_0, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return false;
    }
    equal(dev_f_0, dev_f, N, whichGPU);

    float* dev_d = 0;
    if (cudaMalloc((void**)&dev_d, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return false;
    }

    for (int n = 0; n < numIter; n++)
    {
        anisotropicTotalVariation_gradient(dev_f, dev_d, N_1, N_2, N_3, delta, beta, p, false, whichGPU, -1, -1, numNeighbors, doMean);
        float num = innerProduct(dev_d, dev_d, N, whichGPU);
        float denom = anisotropicTotalVariation_quadraticForm(dev_f, dev_d, N_1, N_2, N_3, delta, beta, p, false, whichGPU, -1, -1, numNeighbors);
        if (denom <= 1.0e-16)
            break;
        float stepSize = num / denom;
        scale(dev_f, 1.0 - stepSize, N, whichGPU);
        sub(dev_d, dev_f_0, N, whichGPU);
        scalarAdd(dev_f, -stepSize, dev_d, N, whichGPU);
        //scalarAdd(dev_f, -stepSize, dev_d, N, whichGPU);

        //printf("cost = %f\n", anisotropicTotalVariation_cost(dev_f, N_1, N_2, N_3, delta, beta, false, whichGPU, numNeighbors));
    }

    // pull result off GPU
    if (data_on_cpu)
        pull3DdataFromGPU(f, N, dev_f, whichGPU);

    // Clean up
    if (data_on_cpu && dev_f != 0)
    {
        cudaFree(dev_f);
    }

    if (dev_d != 0)
        cudaFree(dev_d);
    if (dev_f_0 != 0)
        cudaFree(dev_f_0);
    return true;
}
