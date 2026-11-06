////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// basic CUDA operations
////////////////////////////////////////////////////////////////////////////////
#include "parameters.h"
#include <string.h>
#include <algorithm>

#ifndef __USE_CPU
#include "cuda_utils.h"
#include "cpu_utils.h"
#include "cuda_runtime.h"
#include "fbp/ramp_filter.cuh"

float max_gpu_memory = 4096.0; // 4 TB limit

__global__ void DualTransferFunctionKernel(float* x, float* y, const float* LUT, const int3 N, const float firstSample, const float sampleRate, const int numSamples)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    const float lastSample = float(numSamples - 1) * sampleRate + firstSample;

    const uint64 sample_ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    
    const float curVal_1 = x[sample_ind];
    const float curVal_2 = y[sample_ind];

    const float* LUT_1 = &LUT[0];
    const float* LUT_2 = &LUT[numSamples * numSamples];

    int ind_lo_1, ind_hi_1;
    float d_1;
    if (curVal_1 >= lastSample)
    {
        ind_lo_1 = numSamples - 1;
        ind_hi_1 = numSamples - 1;
        d_1 = 0.0;
    }
    else if (curVal_1 <= firstSample)
    {
        ind_lo_1 = 0;
        ind_hi_1 = 0;
        d_1 = 0.0;
    }
    else
    {
        float ind = (curVal_1 - firstSample) / sampleRate;
        ind_lo_1 = int(ind);
        ind_hi_1 = ind_lo_1 + 1;
        d_1 = ind - float(ind_lo_1);
    }

    int ind_lo_2, ind_hi_2;
    float d_2;
    if (curVal_2 >= lastSample)
    {
        ind_lo_2 = numSamples - 1;
        ind_hi_2 = numSamples - 1;
        d_2 = 0.0;
    }
    else if (curVal_2 <= firstSample)
    {
        ind_lo_2 = 0;
        ind_hi_2 = 0;
        d_2 = 0.0;
    }
    else
    {
        float ind = (curVal_2 - firstSample) / sampleRate;
        ind_lo_2 = int(ind);
        ind_hi_2 = ind_lo_2 + 1;
        d_2 = ind - float(ind_lo_2);
    }

    const float partA_1 = (1.0 - d_2) * LUT_1[ind_lo_1 * numSamples + ind_lo_2] + d_2 * LUT_1[ind_lo_1 * numSamples + ind_hi_2];
    const float partB_1 = (1.0 - d_2) * LUT_1[ind_hi_1 * numSamples + ind_lo_2] + d_2 * LUT_1[ind_hi_1 * numSamples + ind_hi_2];

    const float partA_2 = (1.0 - d_2) * LUT_2[ind_lo_1 * numSamples + ind_lo_2] + d_2 * LUT_2[ind_lo_1 * numSamples + ind_hi_2];
    const float partB_2 = (1.0 - d_2) * LUT_2[ind_hi_1 * numSamples + ind_lo_2] + d_2 * LUT_2[ind_hi_1 * numSamples + ind_hi_2];

    x[sample_ind] = (1.0 - d_1) * partA_1 + d_1 * partB_1;
    y[sample_ind] = (1.0 - d_1) * partA_2 + d_1 * partB_2;
}

__global__ void TransferFunctionKernel(float* f, const float* LUT, const int3 N, const float firstSample, const float sampleRate, const int numSamples)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= N.x || j >= N.y || k >= N.z)
        return;

    const float lastSample = float(numSamples - 1) * sampleRate + firstSample;

    uint64 ind = uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k);
    const float curVal = f[ind];

    if (curVal >= lastSample)
    {
        const float slope = (LUT[numSamples - 1] - LUT[numSamples - 2]) / sampleRate;
        f[ind] = LUT[numSamples - 1] + slope * (curVal - lastSample);
    }
    else if (curVal <= firstSample)
    {
        //f[ind] = firstSample;
        const float slope = (LUT[1] - LUT[0]) / sampleRate;
        f[ind] = LUT[0] + slope * (curVal - firstSample);
    }
    else
    {
        const float arg = (curVal - firstSample) / sampleRate;
        const int arg_low = int(arg);
        const float d = arg - float(arg_low);
        f[ind] = (1.0 - d) * LUT[arg_low] + d * LUT[arg_low + 1];
    }
}

__global__ void copyVolumeDataToMaskKernel(float* f, float* mask, const int4 N, const bool do_forward)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= N.x || iy >= N.y || iz >= N.z)
        return;

    const uint64 ind = uint64(iz) * uint64(N.y * N.x) + uint64(iy * N.x + ix);
    if (do_forward)
    {
        if (mask[ind] == 0.0f)
        {
            if (f[ind] == 0.0f)
                mask[ind] = NAN;
            else
            {
                mask[ind] = f[ind];
                f[ind] = 0.0f;
            }
        }
        else
        {
            if (f[ind] == 0.0f)
                mask[ind] = -1.0f;
            //else
            //	mask[ind] = 1.0;
        }
    }
    else
    {
        if (f[ind] == 0.0f)
        {
            if (isnan(mask[ind]))
                mask[ind] = 0.0f;
            else if (mask[ind] == -1.0f)
                mask[ind] = 1.0f;
            else
            {
                f[ind] = mask[ind];
                mask[ind] = 0.0f;
            }
        }
        else
            mask[ind] = 1.0f;
    }
}

__global__ void windowFOVKernel(float* __restrict__ f, const int4 N, const float4 T, const float4 startVal, const float rFOVsq, const int volumeDimensionOrder)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= N.x || iy >= N.y || iz >= N.z)
        return;

    const float x = ix * T.x + startVal.x;
    const float y = iy * T.y + startVal.y;

    if (x * x + y * y > rFOVsq)
    {
        uint64 ind;
        if (volumeDimensionOrder == 0)
            ind = uint64(ix) * uint64(N.y * N.z) + uint64(iy * N.z + iz);
        else
            ind = uint64(iz) * uint64(N.y * N.x) + uint64(iy * N.x + ix);

        f[ind] = 0.0f;
    }
}

__global__ void reciprocalKernel(float* __restrict__ lhs, const int3 dim, const float divide_by_zero_value)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (lhs[ind] == 0.0f)
        lhs[ind] = divide_by_zero_value;
    else
        lhs[ind] = 1.0f / lhs[ind];
}

__global__ void replaceZerosKernel(float* __restrict__ lhs, const int3 dim, const float newVal)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (lhs[ind] == 0.0f)
        lhs[ind] = newVal;
}

__global__ void replaceNANKernel(float* __restrict__ lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (isnan(lhs[ind]))
    {
        float* aSlice = &lhs[uint64(iz) * uint64(dim.x * dim.y)];
        int count = 0;
        float accum = 0.0f;
        for (int j = -1; j <= 1; j++)
        {
            const int dj = max(0, min(dim.y-1, iy + j));
            for (int k = -1; k <= 1; k++)
            {
                const int dk = max(0, min(dim.x-1, ix + k));
                const float val = aSlice[dj * dim.x + dk];
                if (!isnan(val))
                {
                    accum += val;
                    count += 1;
                }
            }
        }

        if (count > 0)
            lhs[ind] = accum / float(count);
        else
            lhs[ind] = 0.0f;
    }
}

__global__ void clipKernel(float* __restrict__ lhs, const int3 dim, const float clipVal)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (lhs[ind] < clipVal)
        lhs[ind] = clipVal;
}

__global__ void cosKernel(float* __restrict__ lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = cos(lhs[ind]);
}

__global__ void sinKernel(float* __restrict__ lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = sin(lhs[ind]);
}

__global__ void expKernel(float* __restrict__ lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = expf(lhs[ind]);
}

__global__ void negExpKernel(float* __restrict__ lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = expf(-lhs[ind]);
}

__global__ void setToConstantKernel(float* __restrict__ lhs, const float c, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = c;
}

__global__ void equalKernel(float* __restrict__ lhs, const float* __restrict__ rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = rhs[ind];
}

__global__ void multiplyKernel(float* __restrict__ lhs, const float* __restrict__ rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] *= rhs[ind];
}

__global__ void divideKernel(float* __restrict__ lhs, const float* __restrict__ rhs, const int3 dim, const bool skip_zero_denominator)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;
    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    const float rhs_val = rhs[ind];

    if (rhs_val == 0.0f)
    {
        if (!skip_zero_denominator)
            lhs[ind] = 1.0f;
    }
    else
        lhs[ind] /= rhs_val;
}

__global__ void rdivideKernel(float* __restrict__ lhs, const float* __restrict__ rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;
    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    const float rhs_val = rhs[ind];
    const float lhs_val = lhs[ind];

    if (lhs_val == 0.0f)
        lhs[ind] = 1.0f;
    else
        lhs[ind] = rhs_val / lhs_val;
}

__global__ void addKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += rhs[ind];
}

__global__ void addKernel(float* __restrict__ lhs, const float rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += rhs;
}

__global__ void subKernel(float* __restrict__ lhs, const float* __restrict__ rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] -= rhs[ind];
}

__global__ void scaleKernel(float* __restrict__ lhs, const float c, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] *= c;
}

__global__ void scalarAddKernel(float* __restrict__ lhs, const float c, const float* __restrict__ rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += c*rhs[ind];
}

__global__ void meanOverSlicesKernel(float* x, const int3 dim)
{
    const int iy = threadIdx.x + blockIdx.x * blockDim.x;
    const int iz = threadIdx.y + blockIdx.y * blockDim.y;

    if (iy >= dim.y || iz >= dim.z)
        return;

    float accum = 0.0f;
    for (int ix = 0; ix < dim.x; ix++)
    {
        const uint64 ind = uint64(ix) * uint64(dim.y * dim.z) + uint64(iy * dim.z + iz);
        accum += x[ind];
    }
    accum = accum / float(dim.z);
    for (int ix = 0; ix < dim.x; ix++)
    {
        const uint64 ind = uint64(ix) * uint64(dim.y * dim.z) + uint64(iy * dim.z + iz);
        x[ind] = accum;
    }
}

__global__ void sumKernel(const float* x, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[uint64(j * N.z + k)];
        }
    }
}

__global__ void innerProductKernel(const float* x, const float* y, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[uint64(j * N.z + k)] * y_slice[uint64(j * N.z + k)];
        }
    }
}

__global__ void sum_2D(const float* x, float* sum_x, int3 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N.x)
        return;

    const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
    float accum = 0.0f;
    for (int j = 0; j < N.y; j++)
    {
        for (int k = 0; k < N.z; k++)
            accum += x_slice[j * N.z + k];
    }
    sum_x[i] = accum;
}

__global__ void innerProductKernel_2D(const float* x, const float* y, float* sum_x, const int3 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N.x)
        return;

    const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
    const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
    float accum = 0.0f;
    for (int j = 0; j < N.y; j++)
    {
        for (int k = 0; k < N.z; k++)
            accum += x_slice[j * N.z + k] * y_slice[j * N.z + k];
    }
    sum_x[i] = accum;
}

__global__ void innerProductKernel_partial(const float* x, const float* y, float* partial_sum, const uint64 N, const int numberOfChunks, const uint64 maxNumItemsPerCore)
{
    const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;
    if (iprocess >= numberOfChunks)
        return;
    uint64 ind_offset = uint64(iprocess) * uint64(maxNumItemsPerCore);
    const uint64 i_max = min(maxNumItemsPerCore, N - ind_offset);
    float accum = 0.0f;
    for (int i = 0; i < i_max; i++)
        accum += x[ind_offset + i] * y[ind_offset + i];
    partial_sum[iprocess] = accum;
}

__global__ void sum_partial(const float* x, float* partial_sum, const uint64 N, const int numberOfChunks, const uint64 maxNumItemsPerCore)
{
    const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;
    if (iprocess >= numberOfChunks)
        return;
    uint64 ind_offset = uint64(iprocess) * uint64(maxNumItemsPerCore);
    const uint64 i_max = min(maxNumItemsPerCore, N - ind_offset);
    float accum = 0.0f;
    for (int i = 0; i < i_max; i++)
        accum += x[ind_offset + i];
    partial_sum[iprocess] = accum;
}

__global__ void sum_1D(const float* x, float* sum_x, int N)
{
    if (threadIdx.x > 0)
        return;

    /*
    *sum_x = 0.0f;
    for (int i = 0; i < N; i++)
        *sum_x += x[i];
    //*/
    //*
    float accum = 0.0f;
    for (int i = 0; i < N; i++)
        accum += x[i];
    *sum_x = accum;
    //*/
    /*
    double accum = 0.0;
    for (int i = 0; i < N; i++)
        accum += double(x[i]);
    *sum_x = float(accum);
    //*/
}

__global__ void weightedInnerProductKernel(const float* x, const float* w, const float* y, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
        const float* w_slice = &w[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[j * N.z + k] * y_slice[j * N.z + k] * w_slice[j * N.z + k];
        }
    }
}

void printGPUproperties(int whichGPU)
{
    if (whichGPU >= 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, whichGPU);
        printf("maxThreadsPerMultiProcessor = %d\n", prop.maxThreadsPerMultiProcessor);
        printf("multiProcessorCount = %d\n", prop.multiProcessorCount);
        printf("maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
        /*
        const int num_gpus = numberOfGPUs();
        std::vector<cudaDeviceProp> res(num_gpus);
        for (int g_idx = 0; g_idx < num_gpus; g_idx++)
        {
            CUDA_CHECK(cudaGetDeviceProperties(&(res[g_idx]), g_idx));
        }
        return res;
        //*/
    }
}

int getSPcores(int whichGPU)
{
    if (whichGPU >= 0)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, whichGPU);

        int cores = 0;
        int mp = devProp.multiProcessorCount;
        switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            cores = mp * 128;
            //printf("Unknown device type\n");
            break;
        }
        return cores;
    }
    else
        return 0;
}

int numberOfGPUs()
{
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err == cudaSuccess)
        return num_gpus;
    else
        return 0;
}

float getAvailableGPUmemory(std::vector<int> whichGPUs)
{
    if (whichGPUs.size() == 0)
        return 0.0;
    else if (whichGPUs.size() == 1)
        return getAvailableGPUmemory(whichGPUs[0]);
    else
    {
        float retVal = getAvailableGPUmemory(whichGPUs[0]);
        for (int i = 1; i < int(whichGPUs.size()); i++)
            retVal = std::min(retVal, getAvailableGPUmemory(whichGPUs[i]));
        return retVal;
    }
}

float getAvailableGPUmemory(int whichGPU)
{
    if (whichGPU >= 0)
    {
        cudaSetDevice(whichGPU);
        float freeGB;
        if (physically_shared_memory(whichGPU))
        {
            freeGB = getAvailableSystemMemory();
        }
        else
        {
            std::size_t free_byte = 0;
            std::size_t total_byte = 0;
            cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "cudaMemGetInfo failed!\n");
                fprintf(stderr, "error message: %s\n", cudaGetErrorString(err));
                return 0.0;
            }
            freeGB = float(double(free_byte) / pow(2.0, 30.0));
        }
        freeGB = min(max_gpu_memory, freeGB);
        freeGB = max(freeGB * GPU_MEMORY_SAFETY_MULTIPLIER, freeGB - 0.5);
        return freeGB;
    }
    else
        return 0.0;
}

bool physically_shared_memory(int whichGPU)
{
    if (whichGPU >= 0)
    {
        //int device = whichGPU;
        cudaSetDevice(whichGPU);
        //cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, whichGPU);
        //printf("prop.managedMemory = %d, prop.unifiedAddressing = %d\n", prop.managedMemory, prop.unifiedAddressing);
        bool has_physically_shared_memory = prop.managedMemory && prop.unifiedAddressing && prop.integrated;
        return has_physically_shared_memory;
    }
    else
        return false;
}

bool sort_gpus_by_memory(std::vector<int>& whichGPUs)
{
    if (whichGPUs.size() == 0)
        return false;
    if (whichGPUs.size() == 1)
        return true;

    std::vector<float> mem;
    for (int i = 0; i < int(whichGPUs.size()); i++)
        mem.push_back(getAvailableGPUmemory(whichGPUs[i]));

    std::vector<size_t> idx(whichGPUs.size());
    for (size_t i = 0; i < idx.size(); i++)
        idx[i] = i;

    // Sort indices based on a (descending)
    std::sort(idx.begin(), idx.end(),
              [&](size_t i, size_t j) {
                  return mem[i] > mem[j];
              });

    // Reorder both vectors
    std::vector<float> mem_sorted;
    std::vector<int> whichGPUs_sorted;

    mem_sorted.reserve(mem.size());
    whichGPUs_sorted.reserve(whichGPUs.size());

    for (size_t i : idx)
    {
        mem_sorted.push_back(mem[i]);
        whichGPUs_sorted.push_back(whichGPUs[i]);
    }

    //a = std::move(a_sorted);
    whichGPUs = std::move(whichGPUs_sorted);

    return true;
}

cudaError_t setToConstant(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    setToConstantKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t equal(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    if (N.x <= 0 || N.y <= 0 || N.z <= 0)
        return cudaSuccess; // empty copy; a zero-dim kernel launch is a CUDA error
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    cudaGetLastError(); // clear stale error state so only this launch is reported
    equalKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaGetLastError();
}

cudaError_t multiply(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    multiplyKernel<<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t divide(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU, bool skip_zero_denominator)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    divideKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N, skip_zero_denominator);
    return cudaPeekAtLastError();
}

cudaError_t rdivide(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    rdivideKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t add(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    addKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t add(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    addKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t sub(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    subKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t scale(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    scaleKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t scalarAdd(float* dev_lhs, const float c, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    scalarAddKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t mean_over_slices(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock_temp = setBlockSize(N);
    dim3 dimBlock(dimBlock_temp.y, dimBlock_temp.z);
    dim3 dimGrid(int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    meanOverSlicesKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t reciprocal(float* dev_lhs, const int3 N, float divide_by_zero_value, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    reciprocalKernel <<< dimGrid, dimBlock >>> (dev_lhs, N, divide_by_zero_value);
    return cudaPeekAtLastError();
}

extern cudaError_t replaceZeros(float* dev_lhs, const int3 N, int whichGPU, float newVal)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    replaceZerosKernel <<< dimGrid, dimBlock >>> (dev_lhs, N, newVal);
    return cudaPeekAtLastError();
}

extern cudaError_t replaceNAN(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    replaceNANKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

extern cudaError_t clip(float* dev_lhs, const int3 N, int whichGPU, float clipVal)
{
    if (N.x <= 0 || N.y <= 0 || N.z <= 0)
        return cudaSuccess; // nothing to clip; a zero-dim kernel launch is a CUDA error
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    cudaGetLastError(); // clear stale error state so only this launch is reported
    clipKernel <<< dimGrid, dimBlock >>> (dev_lhs, N, clipVal);
    return cudaGetLastError();
}

cudaError_t cosFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    cosKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t sinFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    sinKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t expFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    expKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t negExpFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    negExpKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

float sum(const float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "sum: cudaMalloc failed!\n");
        fprintf(stderr, "error message: %s\n", cudaGetErrorString(cudaStatus));
        return 0.0;
    }
    //sumKernel<<<1,1>>>(dev_lhs, dev_sum, N);

    uint64 numberOfElements = uint64(N.x) * uint64(N.y) * uint64(N.z);
    //* Newest method which makes sure all cores are busy
    int num_gpu_cores = max(1024, getSPcores(whichGPU));
    if (uint64(sqrt(double(numberOfElements))) < uint64(num_gpu_cores))
        num_gpu_cores = int(sqrt(double(numberOfElements)));
    //if (numberOfElements < uint64(num_gpu_cores))
    //    num_gpu_cores = int(numberOfElements);
    uint64 maxNumItemsPerCore = uint64(ceil(double(numberOfElements) / double(num_gpu_cores)));
    //printf("number of cores = %d, number of chunks = %d\n", num_gpu_cores, int(numChunks));

    int blockSize = 8;
    int numBlocks = int(ceil(double(num_gpu_cores) / double(blockSize)));
    int numDataCopies = numBlocks * blockSize; // about equal to num_gpu_cores (rounded up)
    float* dev_partial_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_partial_sum, numDataCopies * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "sum: cudaMalloc failed!\n");
        fprintf(stderr, "error message: %s\n", cudaGetErrorString(cudaStatus));
        return 0.0;
    }
    sum_partial <<< numBlocks, blockSize >>> (dev_lhs, dev_partial_sum, numberOfElements, num_gpu_cores, maxNumItemsPerCore);
    sum_1D <<< 1, 1 >>> (dev_partial_sum, dev_sum, num_gpu_cores);
    cudaFree(dev_partial_sum);
    //*/

    /* Old method
    float* dev_sum_1D = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum_1D, N.x * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "sum: cudaMalloc failed!\n");
        return 0.0;
    }
    int blockSize = 8;
    int gridSize = int(ceil(double(N.x) / double(blockSize)));
    sum_2D <<< gridSize, blockSize >>> (dev_lhs, dev_sum_1D, N);
    sum_1D <<< 1, 1 >>> (dev_sum_1D, dev_sum, N.x);
    cudaFree(dev_sum_1D);
    //*/

    // Slowest method
    //sum_1D <<< 1, 1 >>> (dev_lhs, dev_sum, int(numberOfElements));

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

float innerProduct(const float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "innerProduct: cudaMalloc failed!\n");
        fprintf(stderr, "error message: %s\n", cudaGetErrorString(cudaStatus));
        return 0.0;
    }
    //innerProductKernel <<<1, 1 >>> (dev_lhs, dev_rhs, dev_sum, N);

    uint64 numberOfElements = uint64(N.x) * uint64(N.y) * uint64(N.z);
    //* Newest method which makes sure all cores are busy
    int num_gpu_cores = max(1024, getSPcores(whichGPU));
    if (uint64(sqrt(double(numberOfElements))) < uint64(num_gpu_cores))
        num_gpu_cores = int(sqrt(double(numberOfElements)));
    //if (numberOfElements < uint64(num_gpu_cores))
    //    num_gpu_cores = int(numberOfElements);
    uint64 maxNumItemsPerCore = uint64(ceil(double(numberOfElements) / double(num_gpu_cores)));
    //printf("number of cores = %d, number of chunks = %d\n", num_gpu_cores, int(numChunks));

    int blockSize = 8;
    int numBlocks = int(ceil(double(num_gpu_cores) / double(blockSize)));
    int numDataCopies = numBlocks * blockSize; // about equal to num_gpu_cores (rounded up)
    float* dev_partial_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_partial_sum, numDataCopies * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "innerProduct: cudaMalloc failed!\n");
        fprintf(stderr, "error message: %s\n", cudaGetErrorString(cudaStatus));
        return 0.0;
    }
    innerProductKernel_partial <<< numBlocks, blockSize >>> (dev_lhs, dev_rhs, dev_partial_sum, numberOfElements, num_gpu_cores, maxNumItemsPerCore);
    sum_1D <<< 1, 1 >>> (dev_partial_sum, dev_sum, num_gpu_cores);
    cudaFree(dev_partial_sum);
    //*/

    /*
    float* dev_sum_1D = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum_1D, N.x * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "innerProduct: cudaMalloc failed!\n");
        return 0.0;
    }
    int blockSize = 8;
    int gridSize = int(ceil(double(N.x) / double(blockSize)));
    innerProductKernel_2D <<< gridSize, blockSize >>> (dev_lhs, dev_rhs, dev_sum_1D, N);
    sum_1D <<< 1, 1 >>> (dev_sum_1D, dev_sum, N.x);
    cudaFree(dev_sum_1D);
    //*/

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

float weightedInnerProduct(const float* dev_lhs, const float* dev_w, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus = cudaSuccess;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "weightedInnerProduct: cudaMalloc failed!\n");
        fprintf(stderr, "error message: %s\n", cudaGetErrorString(cudaStatus));
        return 0.0;
    }
    weightedInnerProductKernel <<<1, 1 >>> (dev_lhs, dev_w, dev_rhs, dev_sum, N);

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

dim3 setBlockSize(int3 N)
{
	dim3 dimBlock(8, 8, 8);  // needs to be optimized
    //dim3 dimBlock(32, 4, 2);  // needs to be optimized
	if (N.z < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 16;
		dimBlock.z = 1;
	}
	else if (N.y < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 1;
		dimBlock.z = 16;
	}
	else if (N.x < 8)
	{
		dimBlock.x = 1;
		dimBlock.y = 16;
		dimBlock.z = 16;
	}
	return dimBlock;
}

dim3 setGridSize(int3 N, dim3 dimBlock)
{
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))), int(ceil(double(N.z) / double(dimBlock.z))));
    return dimGrid;
}

dim3 setBlockSize(int4 N)
{
    return setBlockSize(make_int3(N.x, N.y, N.z));
}

dim3 setGridSize(int4 N, dim3 dimBlock)
{
    return setGridSize(make_int3(N.x, N.y, N.z), dimBlock);
}

extern TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    return loadTexture_from_cpu(tex_object, data, N_txt, make_int3(0, 0, 0), make_int3(N_txt.x, N_txt.y, N_txt.z), useExtrapolation, useLinearInterpolation, swapFirstAndLastDimensions);
}

TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    return loadTexture_from_cpu(tex_object, data, N_txt, make_int3(0, 0, 0), make_int3(N_txt.x, N_txt.y, N_txt.z), useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    return loadTexture_from_cpu(tex_object, data, N_txt, make_int3(0, 0, 0), make_int3(N_txt.x, N_txt.y, N_txt.z), useExtrapolation, useLinearInterpolation);
}

// int4 router: unpacks to int3 (optionally swapping x/z), then delegates to the int3 canonical.
TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int4 N_txt, int3 srcPos, int3 srcShape, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N = make_int3(N_txt.z, N_txt.y, N_txt.x);
        srcPos = make_int3(srcPos.z, srcPos.y, srcPos.x);
        srcShape = make_int3(srcShape.z, srcShape.y, srcShape.x);
    }
    return loadTexture_from_cpu(tex_object, data, N, srcPos, srcShape, useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, parameters* params, bool useExtrapolation, bool useLinearInterpolation)
{
    int stride   = params->projectionDataStride > 0 ? params->projectionDataStride : params->numRows;
    int3 srcShape = make_int3(params->numAngles, stride, params->numCols);
    int3 srcPos   = make_int3(0, params->projectionDataFirstRow, 0);
    int3 N_txt    = make_int3(params->numAngles, params->numRows, params->numCols);
    return loadTexture_from_cpu(tex_object, data, N_txt, srcPos, srcShape, useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture_from_cpu(TEX_DATA& tex_object, float* data, const int3 N_txt, int3 srcPos, int3 srcShape, bool useExtrapolation, bool useLinearInterpolation)
{
    // The copy region [srcPos, srcPos + N_txt) must lie inside srcShape on every axis.
    // Out-of-bounds here means cudaMemcpy3D would silently read past one slab into the
    // next (e.g. firstRow + numRows > stride spills into the next angle's data).
    if (srcPos.x < 0 || srcPos.y < 0 || srcPos.z < 0 ||
        srcPos.x + N_txt.z > srcShape.z ||   // x of cudaPos addresses the "cols" axis
        srcPos.y + N_txt.y > srcShape.y ||
        srcPos.z + N_txt.x > srcShape.x)
    {
        printf("loadTexture_from_cpu: srcPos/N_txt out of bounds for srcShape "
               "(srcPos=(%d,%d,%d) N_txt=(%d,%d,%d) srcShape=(%d,%d,%d))\n",
               srcPos.x, srcPos.y, srcPos.z,
               N_txt.x, N_txt.y, N_txt.z,
               srcShape.x, srcShape.y, srcShape.z);
        return nullptr;
    }

    if (data == nullptr)
        return nullptr;

#ifdef __USE_NOTEX
    // Software path: repack the requested sub-region into a dense, extent-order
    // device buffer (axis 0 = cols, fastest-varying) that the software sampler
    // indexes as data[k*n1*n0 + j*n0 + i], matching hardware tex3D(t, i, j, k).
    const long long nElem = (long long)N_txt.x * N_txt.y * N_txt.z;
    float* dev_buf = nullptr;
    if (cudaMalloc((void**)&dev_buf, (size_t)nElem * sizeof(float)) != cudaSuccess)
    {
        printf("loadTexture_from_cpu (NOTEX): cudaMalloc failed\n");
        return nullptr;
    }
    {
        float* host_dense = (float*)malloc((size_t)nElem * sizeof(float));
        for (int d = 0; d < N_txt.x; d++)      // slowest axis (depth)
            for (int h = 0; h < N_txt.y; h++)  // middle axis (rows)
            {
                const float* srow = data + ((long long)(srcPos.z + d) * srcShape.y + (srcPos.y + h)) * srcShape.z + srcPos.x;
                float* drow = host_dense + ((long long)d * N_txt.y + h) * N_txt.z;
                memcpy(drow, srow, (size_t)N_txt.z * sizeof(float));
            }
        cudaMemcpy(dev_buf, host_dense, (size_t)nElem * sizeof(float), cudaMemcpyHostToDevice);
        free(host_dense);
    }
    tex_object.data = dev_buf;
    tex_object.n0 = N_txt.z; tex_object.n1 = N_txt.y; tex_object.n2 = N_txt.x;
    tex_object.address = useExtrapolation ? TEX_CLAMP : TEX_BORDER;
    tex_object.filter  = useLinearInterpolation ? TEX_LINEAR : TEX_NEAREST;
    tex_object.owns = 1;
    return dev_buf;
#else
    cudaArray* d_data_array = nullptr;
    cudaError_t cudaStatus;

    // Allocate CUDA memory array
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    if ((cudaStatus = cudaMalloc3DArray(&d_data_array, &channel_desc, make_cudaExtent(N_txt.z, N_txt.y, N_txt.x), 0)) != cudaSuccess)
    {
        printf("cudaMalloc3DArray Error: %s\n", cudaGetErrorString(cudaStatus));
        printf("attempted to allocate 3D array of size: %d x %d x %d\n", N_txt.z, N_txt.y, N_txt.x);
        int which;
        cudaGetDevice(&which);
        printf("memory available: %f GB\n", getAvailableGPUmemory(which));
        return nullptr;
    }

    // Bind array memory to texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)d_data_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;  // Texture coordinates normalization

    if (useExtrapolation)
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeClamp;
        texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeClamp;
    }
    else
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeBorder;
        texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeBorder;
    }

    if (useLinearInterpolation)
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
    }
    else
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
    }
    if ((cudaStatus = cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr)) != cudaSuccess)
    {
        printf("cudaCreateTextureObject Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        return nullptr;
    }

    //###################################################################
    //void setData(T* src_data, cudaMemcpyKind memcpy_kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0)
    cudaMemcpy3DParms copy_params_;
    memset(&copy_params_, 0, sizeof(cudaMemcpy3DParms));

    // srcPtr describes the full source layout; srcPos offsets into it to skip the first firstRow rows.
    copy_params_.dstArray = (cudaArray_t)d_data_array;
    copy_params_.dstPos = make_cudaPos(0, 0, 0);
    copy_params_.extent = make_cudaExtent(N_txt.z, N_txt.y, N_txt.x);
    copy_params_.srcPos = make_cudaPos(srcPos.x, srcPos.y, srcPos.z);
    copy_params_.kind = cudaMemcpyHostToDevice;
    copy_params_.srcPtr = make_cudaPitchedPtr((void*)data, srcShape.z * sizeof(float), N_txt.z, srcShape.y);

    /*
    cudaStream_t stream;
    if ((cudaStatus = cudaMemcpy3DAsync(&copy_params_, stream)) != cudaSuccess)
    {
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(tex_object);
        return nullptr;
    }
    //*/
    if ((cudaStatus = cudaMemcpy3D(&copy_params_)) != cudaSuccess)
    {
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(tex_object);
        return nullptr;
    }
    //###################################################################

    return d_data_array;
#endif
}

TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    int3 N3 = make_int3(N_txt.x, N_txt.y, N_txt.z);
    return loadTexture(tex_object, dev_data, N3, useExtrapolation, useLinearInterpolation);
}

TEX_ARRAY loadTexture(TEX_DATA& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    if (dev_data == nullptr)
        return nullptr;

#ifdef __USE_NOTEX
    // Software path: alias the caller's device buffer (already in extent order:
    // axis 0 = N.z, fastest). No copy, so freeTexture must not release it.
    tex_object.data = dev_data;
    tex_object.n0 = N_txt.z; tex_object.n1 = N_txt.y; tex_object.n2 = N_txt.x;
    tex_object.address = useExtrapolation ? TEX_CLAMP : TEX_BORDER;
    tex_object.filter  = useLinearInterpolation ? TEX_LINEAR : TEX_NEAREST;
    tex_object.owns = 0;
    return dev_data; // non-NULL so existing "== nullptr" error checks still pass
#else
    cudaArray* d_data_array = nullptr;
    cudaError_t cudaStatus;

    // Allocate 3D array memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    if ((cudaStatus = cudaMalloc3DArray(&d_data_array, &channelDesc, make_cudaExtent(N_txt.z, N_txt.y, N_txt.x), 0)) != cudaSuccess)
    {
        printf("cudaMalloc3DArray Error: %s\n", cudaGetErrorString(cudaStatus));
        printf("attempted to allocate 3D array of size: %d x %d x %d\n", N_txt.z, N_txt.y, N_txt.x);
        int which;
        cudaGetDevice(&which);
        printf("memory available: %f GB\n", getAvailableGPUmemory(which));
        return nullptr;
    }

    // Bind 3D array to texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)d_data_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;  // Texture coordinates normalization

    if (useExtrapolation)
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeClamp;
        texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeClamp;
    }
    else
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeBorder;
        texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeBorder;
    }

    if (useLinearInterpolation)
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
    }
    else
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
    }
    if ((cudaStatus = cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr)) != cudaSuccess)
    {
        printf("cudaCreateTextureObject Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        return nullptr;
    }

    // Update the texture memory
    cudaMemcpy3DParms cudaparams = { 0 };
    cudaparams.extent = make_cudaExtent(N_txt.z, N_txt.y, N_txt.x);
    cudaparams.kind = cudaMemcpyDeviceToDevice;
    cudaparams.srcPos = make_cudaPos(0, 0, 0);
    cudaparams.srcPtr = make_cudaPitchedPtr(dev_data, N_txt.z * sizeof(float), N_txt.z, N_txt.y);
    cudaparams.dstPos = make_cudaPos(0, 0, 0);
    cudaparams.dstArray = (cudaArray_t)d_data_array;
    if ((cudaStatus = cudaMemcpy3D(&cudaparams)) != cudaSuccess)
    {
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(tex_object);
        return nullptr;
    }
    return d_data_array;
#endif
}

TEX_ARRAY loadTexture1D(TEX_DATA& tex_object, float* data, const int N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
#ifdef __USE_NOTEX
    // Software path: copy the 1D samples into an owned device buffer.
    // cudaMemcpyDefault relies on UVA to accept either a host or device source.
    if (data == nullptr)
        return nullptr;
    float* dev_buf = nullptr;
    if (cudaMalloc((void**)&dev_buf, (size_t)N_txt * sizeof(float)) != cudaSuccess)
    {
        printf("loadTexture1D (NOTEX): cudaMalloc failed\n");
        return nullptr;
    }
    cudaMemcpy(dev_buf, data, (size_t)N_txt * sizeof(float), cudaMemcpyDefault);
    tex_object.data = dev_buf;
    tex_object.n0 = N_txt; tex_object.n1 = 1; tex_object.n2 = 1;
    tex_object.address = useExtrapolation ? TEX_CLAMP : TEX_BORDER;
    tex_object.filter  = useLinearInterpolation ? TEX_LINEAR : TEX_NEAREST;
    tex_object.owns = 1;
    return dev_buf;
#else
    /*
    //int2 N_2d = make_int2(1, N_txt);
    int2 N_2d = make_int2(N_txt, 1);
    return loadTexture2D(tex_object, data, N_2d, useExtrapolation, useLinearInterpolation);
    //*/
    //*
    if (data == nullptr)
        return nullptr;
    cudaArray* d_data_array = nullptr;
    cudaError_t cudaStatus;

    // Allocate 3D array memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    if ((cudaStatus = cudaMallocArray(&d_data_array, &channelDesc, N_txt, 1)) != cudaSuccess)
    {
        printf("cudaMallocArray Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    // Bind 1D array to texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)d_data_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;  // Texture coordinates normalization

    if (useExtrapolation)
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
    }
    else
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
    }

    if (useLinearInterpolation)
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
    }
    else
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
    }
    if ((cudaStatus = cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr)) != cudaSuccess)
    {
        printf("cudaCreateTextureObject Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        return nullptr;
    }

    if ((cudaStatus = cudaMemcpy2DToArray(d_data_array, 0, 0, data, sizeof(float) * N_txt, sizeof(float) * N_txt, 1, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(tex_object);
        return nullptr;
    }

    return d_data_array;
    //*/
#endif
}

extern TEX_ARRAY loadTexture2D(TEX_DATA& tex_object, float* data, const int2 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    if (data == nullptr)
        return nullptr;

#ifdef __USE_NOTEX
    // Software path: dense owned buffer, axis 0 = N_txt.y (fastest), axis 1 = N_txt.x.
    const long long nElem = (long long)N_txt.x * N_txt.y;
    float* dev_buf = nullptr;
    if (cudaMalloc((void**)&dev_buf, (size_t)nElem * sizeof(float)) != cudaSuccess)
    {
        printf("loadTexture2D (NOTEX): cudaMalloc failed\n");
        return nullptr;
    }
    cudaMemcpy(dev_buf, data, (size_t)nElem * sizeof(float), cudaMemcpyDefault);
    tex_object.data = dev_buf;
    tex_object.n0 = N_txt.y; tex_object.n1 = N_txt.x; tex_object.n2 = 1;
    tex_object.address = useExtrapolation ? TEX_CLAMP : TEX_BORDER;
    tex_object.filter  = useLinearInterpolation ? TEX_LINEAR : TEX_NEAREST;
    tex_object.owns = 1;
    return dev_buf;
#else
    cudaArray* d_data_array = nullptr;
    cudaError_t cudaStatus;

    // Allocate 2D array memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    if ((cudaStatus = cudaMallocArray(&d_data_array, &channelDesc, N_txt.y, N_txt.x)) != cudaSuccess)
    {
        printf("cudaMallocArray Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    // Bind 1D array to texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)d_data_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;  // Texture coordinates normalization

    if (useExtrapolation)
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeClamp;
    }
    else
    {
        texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
        texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeBorder;
    }

    if (useLinearInterpolation)
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
    }
    else
    {
        texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
    }
    if ((cudaStatus = cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr)) != cudaSuccess)
    {
        printf("cudaCreateTextureObject Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        return nullptr;
    }

    if ((cudaStatus = cudaMemcpy2DToArray(d_data_array, 0, 0, data, sizeof(float) * N_txt.y, sizeof(float) * N_txt.y, N_txt.x, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeArray(d_data_array);
        cudaDestroyTextureObject(tex_object);
        return nullptr;
    }

    return d_data_array;
#endif
}

void freeTexture(TEX_ARRAY& tex_array, TEX_DATA& tex_object)
{
#ifdef __USE_NOTEX
    // Only release buffers that loadTexture* allocated; aliased caller memory
    // (owns == 0, e.g. the device loadTexture path) is left for the caller.
    if (tex_object.owns && tex_array != nullptr)
        cudaFree((void*)tex_array);
    tex_array = nullptr;
    tex_object.data = nullptr;
    tex_object.owns = 0;
#else
    if (tex_array != nullptr)
        cudaFreeArray(tex_array);
    if (tex_object != 0)
        cudaDestroyTextureObject(tex_object);
    tex_array = nullptr;
    tex_object = 0;
#endif
}

float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    uint64 N = params->projectionData_numberOfElements();

    int stride    = params->projectionDataStride > 0 ? params->projectionDataStride : params->numRows;
    int3 srcShape = make_int3(params->numAngles, stride, params->numCols);
    int3 srcPos   = make_int3(0, params->projectionDataFirstRow, 0);
    int3 N_txt    = make_int3(params->numAngles, params->numRows, params->numCols);

    // The copy region [srcPos, srcPos + N_txt) must lie inside srcShape on every axis.
    // Out-of-bounds here means cudaMemcpy3D would silently read past one slab into the
    // next (e.g. firstRow + numRows > stride spills into the next angle's data).
    if (g == nullptr ||
        srcPos.x < 0 || srcPos.y < 0 || srcPos.z < 0 ||
        srcPos.x + N_txt.z > srcShape.z ||   // x of cudaPos addresses the "cols" axis
        srcPos.y + N_txt.y > srcShape.y ||
        srcPos.z + N_txt.x > srcShape.x)
    {
        printf("copyProjectionDataToGPU: srcPos/N_txt out of bounds for srcShape "
               "(srcPos=(%d,%d,%d) N_txt=(%d,%d,%d) srcShape=(%d,%d,%d))\n",
               srcPos.x, srcPos.y, srcPos.z,
               N_txt.x, N_txt.y, N_txt.z,
               srcShape.x, srcShape.y, srcShape.z);
        return nullptr;
    }
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g, N * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(projection[%d,%d,%d]) failed!\n", params->numAngles, params->numRows, params->numCols);
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}
    cudaMemcpy3DParms copy_params_;
    memset(&copy_params_, 0, sizeof(cudaMemcpy3DParms));

    // No cudaArray participates in this copy (both src and dst are linear/pitched
    // memory), so cudaMemcpy3D interprets the width axis (extent.width, pos.x, and
    // the pitched-ptr xsize) in BYTES, not elements. The row/slice axes stay in rows.
    copy_params_.dstPtr = make_cudaPitchedPtr((void*)dev_g, N_txt.z * sizeof(float), N_txt.z * sizeof(float), N_txt.y);
    copy_params_.dstPos = make_cudaPos(0, 0, 0);
    copy_params_.extent = make_cudaExtent(N_txt.z * sizeof(float), N_txt.y, N_txt.x);
    copy_params_.srcPos = make_cudaPos(srcPos.x * sizeof(float), srcPos.y, srcPos.z);
    copy_params_.kind = cudaMemcpyHostToDevice;
    copy_params_.srcPtr = make_cudaPitchedPtr((void*)g, srcShape.z * sizeof(float), N_txt.z * sizeof(float), srcShape.y);
    if ((cudaStatus = cudaMemcpy3D(&copy_params_)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy3D(projection) failed!\n");
        printf("cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_g);
		return NULL;
	}

	return dev_g;
}

bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    int stride    = params->projectionDataStride > 0 ? params->projectionDataStride : params->numRows;
    int3 dstShape = make_int3(params->numAngles, stride, params->numCols);
    int3 dstPos   = make_int3(0, params->projectionDataFirstRow, 0);
    int3 N_txt    = make_int3(params->numAngles, params->numRows, params->numCols);

    cudaMemcpy3DParms copy_params_;
    memset(&copy_params_, 0, sizeof(cudaMemcpy3DParms));

    // dev_g is contiguous (numAngles, numRows, numCols); host g may be strided.
    copy_params_.srcPtr = make_cudaPitchedPtr((void*)dev_g, N_txt.z * sizeof(float), N_txt.z * sizeof(float), N_txt.y);
    copy_params_.srcPos = make_cudaPos(0, 0, 0);
    copy_params_.extent = make_cudaExtent(N_txt.z * sizeof(float), N_txt.y, N_txt.x);
    copy_params_.dstPos = make_cudaPos(dstPos.x * sizeof(float), dstPos.y, dstPos.z);
    copy_params_.kind   = cudaMemcpyDeviceToHost;
    copy_params_.dstPtr = make_cudaPitchedPtr((void*)g, dstShape.z * sizeof(float), N_txt.z * sizeof(float), dstShape.y);

    if ((cudaStatus = cudaMemcpy3D(&copy_params_)) != cudaSuccess)
    {
        fprintf(stderr, "failed to copy projection data back to host!\n");
        fprintf(stderr, "cudaMemcpy3D Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }
    return true;
}

float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    uint64 N = params->volumeData_numberOfElements();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_f = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_f, N * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}
    if ((cudaStatus = cudaMemcpy(dev_f, f, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy(volume) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_f);
		return NULL;
	}

	return dev_f;
}

bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU, bool skipFIHT)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    bool do_clipping = true;

    //*
    if (params->doDBP && skipFIHT == false && params->DBPparameter >= 0.0)
    {
        float scalar = 1.0;
        //float sampleShift = 0.0;
        float sampleShift = -1.0;
        finiteInverseHilbert(dev_f, params, false, scalar, sampleShift);
        if (params->DBPparameter <= 0.0)
            do_clipping = false;
        else if (params->rFOV() < params->furthestFromCenter())
            windowFOV_gpu(dev_f, params);
    }
    //*/

    if (do_clipping && params->clipWeightedBackprojection == true && (params->doDBP == true || params->doWeightedBackprojection == true) && params->inconsistencyReconstruction == false)
    {
        cudaStatus = clip(dev_f, make_int3(params->numZ, params->numY, params->numX), whichGPU);
        if (cudaSuccess != cudaStatus)
        {
            fprintf(stderr, "volume clipping failed!\n");
            fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
            fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            return false;
        }
    }

    if (f == dev_f)
    {
        cudaDeviceSynchronize();
        return true;
    }
    uint64 N = params->volumeData_numberOfElements();
	cudaStatus = cudaMemcpy(f, dev_f, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy volume data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copy3DdataToGPU(float* g, int3 N, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

	uint64 N_prod = uint64(N.x) * uint64(N.y) * uint64(N.z);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g, N_prod * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
		return NULL;
	}
    if ((cudaStatus = cudaMemcpy(dev_g, g, N_prod * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy(3Ddata) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_g);
		return NULL;
	}

	return dev_g;
}

extern float* copy1DdataToGPU(float* x, int N, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy volume data to GPU
    float* dev_x = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(1D) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(1D) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_x);
        return NULL;
    }

    return dev_x;
}

extern float3* copy1Dfloat3ToGPU(float3* x, int N, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy volume data to GPU
    float3* dev_x = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float3))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(1D) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(float3), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(1D) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_x);
        return NULL;
    }

    return dev_x;
}

extern float4* copy1Dfloat4ToGPU(float4* x, int N, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy volume data to GPU
    float4* dev_x = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float4))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(1D) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(float4), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(1D) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_x);
        return NULL;
    }

    return dev_x;
}

extern bool* copy1DbooleanToGPU(bool* x, int N, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Copy volume data to GPU
    bool* dev_x = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(bool))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(1D) failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(bool), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(1D) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_x);
        return NULL;
    }

    return dev_x;
}

bool pull1DdataFromGPU(float* x, int N, float* dev_x, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

	if ((cudaStatus = cudaMemcpy(x, dev_x, N * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "failed to copy 1D data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

bool pull3DdataFromGPU(float* g, int3 N, float* dev_g, int whichGPU)
{
    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    uint64 N_prod = uint64(N.x) * uint64(N.y) * uint64(N.z);
	if ((cudaStatus = cudaMemcpy(g, dev_g, N_prod * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "failed to copy 3D data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copyAngleArrayToGPU(parameters* params)
{
    if (params == NULL)
    {
        printf("Error: copyAngleArrayToGPU: invalid argument!\n");
        return NULL;
    }

    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(params->whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return nullptr;
    }

    //cudaError_t cudaStatus;
    float* dev_phis = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "copyAngleArrayToGPU: cudaMalloc failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return NULL;
    }
    if ((cudaStatus = cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "copyAngleArrayToGPU: cudaMemcpy(phis) failed!\n");
        printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_phis);
        return NULL;
    }
    return dev_phis;
}

bool setProjectionGPUparams(parameters* params, int4& N, float4& T, float4& startVals, bool doNormalize)
{
    if (params == NULL)
    {
        printf("Error: setProjectionGPUparams: invalid argument!\n");
        return false;
    }
    else
    {
        N.x = params->numAngles; N.y = params->numRows; N.z = params->numCols;
        T.x = params->T_phi(); T.y = params->pixelHeight; T.z = params->pixelWidth;
        startVals.x = params->phi_0(); startVals.y = params->v_0(); startVals.z = params->u_0();
        //*
        if (params->geometry == parameters::MODULAR)
        {
            startVals.y = -(params->centerRow + params->rowShiftFromFilter) * params->pixelHeight;
            startVals.z = -(params->centerCol + params->colShiftFromFilter) * params->pixelWidth;
        }
        //*/

        if (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL)
        {
            N.w = params->numAngles;
            T.w = params->helicalPitch;
            startVals.w = params->z_source_offset;
        }
        else
        {
            N.w = params->numAngles;
            T.w = 0.0;
            startVals.w = 0.0;
        }

        if (params->geometry == parameters::CONE && params->detectorType == parameters::CURVED)
        {
            T.z = atan(T.z / params->sdd);
            if (doNormalize)
            {
                T.y = T.y / params->sdd;
                startVals.y = startVals.y / params->sdd;
            }
        }
        else if (doNormalize)
        {
            if (params->geometry == parameters::CONE)
            {
                T.y = T.y / params->sdd;
                startVals.y = startVals.y / params->sdd;
                
                T.z = T.z / params->sdd;
                startVals.z = startVals.z / params->sdd;
            }
            else if (params->geometry == parameters::CONE_PARALLEL)
            {
                T.y = T.y / params->sdd;
                startVals.y = startVals.y / params->sdd;

                //T.z = T.z / params->sdd;
                //startVals.z = startVals.z / params->sdd;
            }
            else if (params->geometry == parameters::FAN)
            {
                T.z = T.z / params->sdd;
                startVals.z = startVals.z / params->sdd;
            }
        }

        return true;
    }
}

bool setVolumeGPUparams(parameters* params, int4& N, float4& T, float4& startVals)
{
    if (params == NULL)
    {
        printf("Error: setVolumeGPUparams: invalid argument!\n");
        return false;
    }
    else
    {
        N.x = params->numX; N.y = params->numY; N.z = params->numZ;
        T.x = params->voxelWidth; T.y = params->voxelWidth; T.z = params->voxelHeight;
        startVals.x = params->x_0(); startVals.y = params->y_0(); startVals.z = params->z_0();
        return true;
    }
}

bool windowFOV_gpu(float* f, parameters* params)
{
    if (params == NULL)
    {
        printf("Error: windowFOV_gpu: invalid argument!\n");
        return false;
    }

    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(params->whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    float rFOVsq = params->rFOV() * params->rFOV();

    int4 N; float4 T; float4 startVal;
    if (setVolumeGPUparams(params, N, T, startVal) == false)
        return false;

    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);
    windowFOVKernel <<< dimGrid, dimBlock >>> (f, N, T, startVal, rFOVsq, params->volumeDimensionOrder);
    cudaStatus = cudaDeviceSynchronize();

    return true;
}

bool copy_volume_data_to_mask_gpu(float* f, float* mask, parameters* params, bool do_forward)
{
    if (f ==NULL || mask == NULL || params == NULL)
    {
        printf("Error: copy_volume_data_to_mask_gpu: invalid argument!\n");
        return false;
    }

    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(params->whichGPU)) != cudaSuccess)
    {
        printf("cudaSetDevice Error: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    int4 N; float4 T; float4 startVal;
    if (setVolumeGPUparams(params, N, T, startVal) == false)
        return false;

    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);
    copyVolumeDataToMaskKernel <<< dimGrid, dimBlock >>> (f, mask, N, do_forward);
    cudaStatus = cudaDeviceSynchronize();

    return true;
}

bool applyTransferFunction_gpu(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, int whichGPU, bool data_on_cpu)
{
    if (x == NULL) return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    float* dev_LUT = 0;
    if (data_on_cpu)
    {
        dev_f = copy3DdataToGPU(x, N, whichGPU);

        if (cudaSuccess != cudaMalloc((void**)&dev_LUT, numSamples * sizeof(float)))
            fprintf(stderr, "applyTransferFunction_gpu: cudaMalloc failed!\n");
        if (cudaMemcpy(dev_LUT, LUT, numSamples * sizeof(float), cudaMemcpyHostToDevice))
            fprintf(stderr, "applyTransferFunction_gpu: cudaMemcpy(LUT) failed!\n");
    }
    else
    {
        dev_f = x;
        dev_LUT = LUT;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    TransferFunctionKernel <<<dimGrid, dimBlock >>> (dev_f, dev_LUT, N, firstSample, sampleRate, numSamples);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        pull3DdataFromGPU(x, N, dev_f, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
        if (dev_LUT != 0)
            cudaFree(dev_LUT);
    }

    return true;
}

bool applyDualTransferFunction_gpu(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, int whichGPU, bool data_on_cpu)
{
    if (x == NULL || y == NULL) return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_x = 0;
    float* dev_y = 0;
    float* dev_LUT = 0;
    if (data_on_cpu)
    {
        dev_x = copy3DdataToGPU(x, N, whichGPU);
        dev_y = copy3DdataToGPU(y, N, whichGPU);

        if (cudaSuccess != cudaMalloc((void**)&dev_LUT, 2*numSamples*numSamples * sizeof(float)))
            fprintf(stderr, "applyDualTransferFunction_gpu: cudaMalloc failed!\n");
        if (cudaMemcpy(dev_LUT, LUT, 2 * numSamples * numSamples * sizeof(float), cudaMemcpyHostToDevice))
            fprintf(stderr, "applyDualTransferFunction_gpu: cudaMemcpy(LUT) failed!\n");
    }
    else
    {
        dev_x = x;
        dev_y = y;
        dev_LUT = LUT;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    DualTransferFunctionKernel <<< dimGrid, dimBlock >>> (dev_x, dev_y, dev_LUT, N, firstSample, sampleRate, numSamples);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        pull3DdataFromGPU(x, N, dev_x, whichGPU);
        pull3DdataFromGPU(y, N, dev_y, whichGPU);

        if (dev_x != 0)
            cudaFree(dev_x);
        if (dev_y != 0)
            cudaFree(dev_y);
        if (dev_LUT != 0)
            cudaFree(dev_LUT);
    }

    return true;
}

// For non-GPU versions of these functions, see cpu_utils.cpp. 
// Put there instead to avoid linking errors on CPU builds.
#endif
