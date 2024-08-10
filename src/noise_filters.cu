////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CUDA-based thresholded median filter and 3D low pass filter calculations
////////////////////////////////////////////////////////////////////////////////
#include "noise_filters.cuh"

#include <math.h>

#include "cuda_utils.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "parameters.h"

__constant__ bool d_DO_HIGH_PASS;

__global__ void azimuthalBlurKernel(float* f, float* f_filtered, const int3 N, const float3 T, const float3 startVal, const int N_phi_max, const float filterWidth, const float rFOVsq)
{
    // return;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float x = i * T.x + startVal.x;
    const float y = j * T.y + startVal.y;

    if (x * x + y * y > rFOVsq)
    {
        f[uint64(k) * uint64(N.x * N.y) + uint64(j * N.x + i)] = 0.0f;
        return;
    }

    // const float z = k * T.z + startVal.z;

    const float r = sqrt(x * x + y * y);
    const float one_over_Tx = 1.0f / T.x;
    int N_phi;
    float val = 0.0f;

    const float* f_slice = &f[uint64(k) * uint64(N.x * N.y)];

    const int N_xy = N.x * N.y;

    // do filtering
    if (filterWidth >= 360.0f)
    {
        float T_phi = atan(T.x / r);
        N_phi = max(4, min(N_phi_max, 2 * (int)(ceil(3.141592653589793f / T_phi))));
        T_phi = 2.0f * 3.141592653589793f / ((float)N_phi);
        for (int l = 0; l < N_phi; l += 2)
        {
            const float phi = T_phi * l + 0.5f * T_phi;
            const int ix_A = int(0.5f + (r * cos(phi) - startVal.x) * one_over_Tx);
            const int iy_A = int(0.5f + (r * sin(phi) - startVal.y) * one_over_Tx);
            const int ix_B = int(0.5f + (r * cos(phi + T_phi) - startVal.x) * one_over_Tx);
            const int iy_B = int(0.5f + (r * sin(phi + T_phi) - startVal.y) * one_over_Tx);

            const int ind_A = iy_A * N.x + ix_A;
            const int ind_B = iy_B * N.x + ix_B;
            if (0 <= ind_A && ind_A < N_xy)
                val += f_slice[ind_A];
            if (0 <= ind_B && ind_B < N_xy)
                val += f_slice[ind_B];
            // val += read_imagef(f, sampler, (float2)((r * sin(phi) - clf_y_0(f_info)) * one_over_Tx + 0.5f, (r *
            // cos(phi) - clf_x_0(f_info)) * one_over_Tx + 0.5f))
            //+ read_imagef(f, sampler, (float2)((r * sin(phi + T_phi) - clf_y_0(f_info)) * one_over_Tx + 0.5f, (r *
            //cos(phi + T_phi) - clf_x_0(f_info)) * one_over_Tx + 0.5f));
        }
    }
    else
    {
        float T_phi = atan(T.x / r);
        N_phi = max(4, min(N_phi_max, 2 * (int)(ceil((3.141592653589793f / 360.0f) * filterWidth / T_phi))));
        T_phi = (3.141592653589793f / 180.0f) * filterWidth / ((float)N_phi);
        const float psi = atan2(y, x) + 0.5f * T_phi * N_phi;
        for (int l = 0; l < N_phi; l += 2)
        {
            const float phi = T_phi * l + 0.5f * T_phi - psi;
            const int ix_A = int(0.5f + (r * cos(phi) - startVal.x) * one_over_Tx);
            const int iy_A = int(0.5f + (-r * sin(phi) - startVal.y) * one_over_Tx);
            const int ix_B = int(0.5f + (r * cos(phi + T_phi) - startVal.x) * one_over_Tx);
            const int iy_B = int(0.5f + (-r * sin(phi + T_phi) - startVal.y) * one_over_Tx);
            const int ind_A = iy_A * N.x + ix_A;
            const int ind_B = iy_B * N.x + ix_B;
            if (0 <= ind_A && ind_A < N_xy)
                val += f_slice[ind_A];
            if (0 <= ind_B && ind_B < N_xy)
                val += f_slice[ind_B];

            // val += read_imagef(f, sampler, (float2)((-r * sin(phi) - clf_y_0(f_info)) * one_over_Tx + 0.5f, (r *
            // cos(phi) - clf_x_0(f_info)) * one_over_Tx + 0.5f))
            //+ read_imagef(f, sampler, (float2)((-r * sin(phi + T_phi) - clf_y_0(f_info)) * one_over_Tx + 0.5f, (r *
            //cos(phi + T_phi) - clf_x_0(f_info)) * one_over_Tx + 0.5f));
        }
    }

    f_filtered[uint64(k) * uint64(N.x * N.y) + uint64(j * N.x + i)] = val / ((float)N_phi);
}

__global__ void badPixelCorrectionKernel(float* g, float* badPixelMap, const int3 N, const int windowRadius)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    if (badPixelMap[j*N.z+k] != 1.0f) // pixel is good; do nothing
        return;

    uint64 iProj = uint64(i) * uint64(N.y) * uint64(N.z);
    float* aProj = &g[iProj];

    float v[49];
    int ind = 0;
    for (int dj = -windowRadius; dj <= windowRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        for (int dk = -windowRadius; dk <= windowRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            if (badPixelMap[j_shift * N.z + k_shift] != 1.0f) // pixel is good, store it
            {
                v[ind] = aProj[j_shift * N.z + k_shift];
                ind += 1;
            }
        }
    }

    if (ind == 1)
    {
        aProj[j * N.z + k] = v[0];
    }
    else if (ind == 2)
    {
        aProj[j * N.z + k] = 0.5f * (v[0] + v[1]);
    }
    else if (ind > 2)
    {
        // 3 ==> 2
        // 4 ==> 3
        // 5 ==> 3
        // 6 ==> 4 (need 2 and 3)
        // 7 ==> 4
        // 8 ==> 5
        // 9 ==> 5
        const int ind_mid = (ind - (ind % 2)) / 2 + 1;

        // bubble-sort for first half of samples
        for (int i = 0; i < ind_mid; i++)
        {
            for (int j = i + 1; j < ind; j++)
            {
                if (v[i] > v[j])
                {  // swap?
                    const float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }
        if (ind % 2 == 0)
            aProj[j * N.z + k] = 0.5f * (v[ind_mid - 1] + v[ind_mid - 2]);
        else
            aProj[j * N.z + k] = v[ind_mid-1];
    }
}

__global__ void medianFilter2DKernel(float* f, float* f_filtered, const int3 N, const float threshold, const int windowRadius, const float signalThreshold)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    uint64 i_slice = uint64(i) * uint64(N.y) * uint64(N.z);
    float* f_slice = &f[i_slice];
    if (windowRadius == 1)
    {
        float v[9];
        int ind = 0;
        for (int dj = -windowRadius; dj <= windowRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -windowRadius; dk <= windowRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f_slice[j_shift * N.z + k_shift];
                ind += 1;
            }
        }
        const float curVal = v[4];
        if (signalThreshold > 0.0f && curVal > signalThreshold)
        {
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
            return;
        }

        // bubble-sort for first 5 samples
        for (int i = 0; i < 5; i++)
        {
            for (int j = i + 1; j < 9; j++)
            {
                if (v[i] > v[j])
                {  // swap?
                    const float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }
        if (fabs(curVal - v[4]) >= threshold * fabs(v[4]))
            f_filtered[i_slice + uint64(j * N.z + k)] = v[4];
        else
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
    }
    else if (windowRadius == 2)
    {
        float v[25];
        int ind = 0;
        for (int dj = -windowRadius; dj <= windowRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -windowRadius; dk <= windowRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f_slice[j_shift * N.z + k_shift];
                ind += 1;
            }
        }
        const float curVal = v[12];
        if (signalThreshold > 0.0f && curVal > signalThreshold)
        {
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
            return;
        }

        // bubble-sort for first 13 samples
        for (int i = 0; i < 13; i++)
        {
            for (int j = i + 1; j < 25; j++)
            {
                if (v[i] > v[j])
                {  // swap?
                    const float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }
        if (fabs(curVal - v[12]) >= threshold * fabs(v[12]))
            f_filtered[i_slice + uint64(j * N.z + k)] = v[12];
        else
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
    }
    else if (windowRadius == 3)
    {
        float v[49];
        int ind = 0;
        for (int dj = -windowRadius; dj <= windowRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -windowRadius; dk <= windowRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f_slice[j_shift * N.z + k_shift];
                ind += 1;
            }
        }
        const float curVal = v[24];
        if (signalThreshold > 0.0f && curVal > signalThreshold)
        {
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
            return;
        }

        // bubble-sort for first 25 samples
        for (int i = 0; i < 25; i++)
        {
            for (int j = i + 1; j < 49; j++)
            {
                if (v[i] > v[j])
                {  // swap?
                    const float tmp = v[i];
                    v[i] = v[j];
                    v[j] = tmp;
                }
            }
        }
        if (fabs(curVal - v[24]) >= threshold * fabs(v[24]))
            f_filtered[i_slice + uint64(j * N.z + k)] = v[24];
        else
            f_filtered[i_slice + uint64(j * N.z + k)] = curVal;
    }
}

__global__ void meanFilterKernel(float* f, float* f_filtered, int3 N, const int r, int sliceStart, int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    const int di_min = -min(i, r);
    const int di_max = min(N.x - 1 - i, r);

    const int dj_min = -min(j, r);
    const int dj_max = min(N.y - 1 - j, r);

    const int dk_min = -min(k, r);
    const int dk_max = min(N.z - 1 - k, r);

    float x = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                x += f[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
            }
        }
    }
    f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = x / float((di_max - di_min + 1) * (dj_max - dj_min + 1) * (dk_max - dk_min + 1));
}

__global__ void varianceFilterKernel(float* f, float* f_filtered, int3 N, const int r, int sliceStart, int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    const int di_min = -min(i, r);
    const int di_max = min(N.x - 1 - i, r);

    const int dj_min = -min(j, r);
    const int dj_max = min(N.y - 1 - j, r);

    const int dk_min = -min(k, r);
    const int dk_max = min(N.z - 1 - k, r);

    const float weight = 1.0f / float((di_max - di_min + 1) * (dj_max - dj_min + 1) * (dk_max - dk_min + 1));

    float x = 0.0f;
    float xx = 0.0f;
    for (int di = di_min; di <= di_max; di++)
    {
        for (int dj = dj_min; dj <= dj_max; dj++)
        {
            for (int dk = dk_min; dk <= dk_max; dk++)
            {
                const float curVal = f[uint64(i + di) * uint64(N.z * N.y) + uint64((j + dj) * N.z + (k + dk))];
                x += curVal;
                xx += curVal * curVal;
            }
        }
    }

    const float meanI = x * weight;
    const float varI = xx * weight - meanI * meanI;

    f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = varI;
}

__global__ void medianFilterKernel(float* f, float* f_filtered, int3 N, float threshold, const float signalThreshold, int sliceStart, int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    float v[27];
    int ind = 0;
    for (int di = -1; di <= 1; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -1; dj <= 1; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -1; dk <= 1; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j_shift * N.z + k_shift)];
                ind += 1;
            }
        }
    }
    const float curVal = v[13];
    if (signalThreshold > 0.0f && curVal > signalThreshold)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = curVal;
        return;
    }

    // bubble-sort for first 14 samples
    for (int i = 0; i < 14; i++)
    {
        for (int j = i + 1; j < 27; j++)
        {
            if (v[i] > v[j])
            {  // swap?
                const float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // fabs(curVal-v[13])/fabs(curVal) > threshold
    if (fabs(curVal - v[13]) >= threshold * fabs(v[13]))
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = v[13];
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = curVal;
}

__global__ void medianFilterKernel_5x5(float* f, float* f_filtered, int3 N, float threshold, const float signalThreshold, int sliceStart, int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }


    //f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = f[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)];
    //return;

    /*
    float v[125];
    int ind = 0;
    for (int di = -2; di <= 2; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -2; dj <= 2; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -2; dk <= 2; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j_shift * N.z + k_shift)];
                ind += 1;
            }
        }
    }
    const float curVal = v[62];

    // bubble-sort for first 63 samples
    for (int i = 0; i < 63; i++)
    {
        for (int j = i + 1; j < 125; j++)
        {
            if (v[i] > v[j])
            {  // swap?
                const float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // fabs(curVal-v[62])/fabs(curVal) > threshold
    if (fabs(curVal - v[62]) >= threshold * fabs(v[62]))
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = v[62];
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = curVal;
    //*/

    //*
    float v[75];
    int ind = 0;
    for (int di = -1; di <= 1; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -2; dj <= 2; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -2; dk <= 2; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j_shift * N.z + k_shift)];
                ind += 1;
            }
        }
    }
    const float curVal = v[37];
    if (signalThreshold > 0.0f && curVal > signalThreshold)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = curVal;
        return;
    }

    // bubble-sort for first 38 samples
    for (int i = 0; i < 38; i++)
    {
        for (int j = i + 1; j < 75; j++)
        {
            if (v[i] > v[j])
            {  // swap?
                const float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // fabs(curVal-v[37])/fabs(curVal) > threshold
    if (fabs(curVal - v[37]) >= threshold * fabs(v[37]))
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = v[37];
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = curVal;
    //*/
}

__global__ void BlurFilterKernel(float* f, float* f_filtered, int3 N, float FWHM, const int sliceStart, const int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        //if (i + di < 0 || i + di > N.x - 1)
        //    continue;
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            //if (j + dj < 0 || j + dj > N.y - 1)
            //    continue;
            for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                //if (k + dk < 0 || k + dk > N.z - 1)
                //    continue;

                const float theWeight = 0.5f +
                    0.5f * cosf(3.141592653589793f* min(sqrtf(float(di * di + dj * dj + dk * dk)) * denom, 1.0f));

                if (theWeight > 0.0001f)
                {
                    val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j_shift * N.z + k_shift)];
                    sum += theWeight;
                }
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = f[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}

__global__ void BlurFilter2DKernel(float* f, float* f_filtered, const int3 N, const float FWHM, const int axis)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;

    float* f_slice = &f[uint64(i) * uint64(N.y * N.z)];
    for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        //const float j_dist_sq = float((j - j_shift) * (j - j_shift));
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            //const float k_dist_sq = float((k - k_shift) * (k - k_shift));

            //const float theWeight = exp(-denom * (j_dist_sq + k_dist_sq));

            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(sqrtf(float(dj * dj + dk * dk)) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                val += theWeight * f_slice[uint64(j_shift * N.z + k_shift)];
                sum += theWeight;
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = f[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}

__global__ void BlurFilter1DKernel(float* f, float* f_filtered, const int3 N, const float FWHM, const int axis)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;
    if (axis == 0)
    {
        for (int di = -pixelRadius; di <= pixelRadius; di++)
        {
            const int i_shift = max(0, min(i + di, N.x - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(di) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j * N.z + k)];
                sum += theWeight;
            }
        }
    }
    else if (axis == 1)
    {
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(dj) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                val += theWeight * f[uint64(i) * uint64(N.y * N.z) + uint64(j_shift * N.z + k)];
                sum += theWeight;
            }
        }
    }
    else //if (axis == 2)
    {
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(dk) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                val += theWeight * f[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k_shift)];
                sum += theWeight;
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = f[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}

//########################################################################################################################################################
__global__ void BlurFilterKernel_txt(cudaTextureObject_t f, float* f_filtered, int3 N, float FWHM, const int sliceStart, const int sliceEnd)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    if (i < sliceStart || i > sliceEnd)
    {
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = 0.0f;
        return;
    }

    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        //const int i_shift = i + di;
        //if (i + di < 0 || i + di > N.x - 1)
        //    continue;
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            //const int j_shift = j + dj;
            //if (j + dj < 0 || j + dj > N.y - 1)
            //    continue;
            for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                //const int k_shift = k + dk;
                //if (k + dk < 0 || k + dk > N.z - 1)
                //    continue;

                const float theWeight = 0.5f +
                    0.5f * cosf(3.141592653589793f * min(sqrtf(float(di * di + dj * dj + dk * dk)) * denom, 1.0f));

                if (theWeight > 0.0001f)
                {
                    //val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j_shift * N.z + k_shift)];
                    val += theWeight * tex3D<float>(f, k_shift, j_shift, i_shift);
                    sum += theWeight;
                }
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = tex3D<float>(f, k, j, i) - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}

__global__ void BlurFilter2DKernel_txt(cudaTextureObject_t f, float* f_filtered, const int3 N, const float FWHM, const int axis)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;

    //float* f_slice = &f[uint64(i) * uint64(N.y * N.z)];
    for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        //const float j_dist_sq = float((j - j_shift) * (j - j_shift));
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            //const float k_dist_sq = float((k - k_shift) * (k - k_shift));

            //const float theWeight = exp(-denom * (j_dist_sq + k_dist_sq));

            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(sqrtf(float(dj * dj + dk * dk)) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                //val += theWeight * f_slice[uint64(j_shift * N.z + k_shift)];
                val += theWeight * tex3D<float>(f, k_shift, j_shift, i);
                sum += theWeight;
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = tex3D<float>(f, k, j, i) - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}

__global__ void BlurFilter1DKernel_txt(cudaTextureObject_t f, float* f_filtered, const int3 N, const float FWHM, const int axis)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    //const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    //const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    //const float denom = 1.0f / (2.0f * sigma * sigma);
    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0f;
    float sum = 0.0f;
    if (axis == 0)
    {
        for (int di = -pixelRadius; di <= pixelRadius; di++)
        {
            const int i_shift = max(0, min(i + di, N.x - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(di) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                //val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j * N.z + k)];
                val += theWeight * tex3D<float>(f, k, j, i_shift);
                sum += theWeight;
            }
        }
    }
    else if (axis == 1)
    {
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(dj) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                //val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j * N.z + k)];
                val += theWeight * tex3D<float>(f, k, j_shift, i);
                sum += theWeight;
            }
        }
    }
    else //if (axis == 2)
    {
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));

            //const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));
            const float theWeight = 0.5f +
                0.5f * cosf(3.141592653589793f * min(fabsf(dk) * denom, 1.0f));

            if (theWeight > 0.0001f)
            {
                //val += theWeight * f[uint64(i_shift) * uint64(N.y * N.z) + uint64(j * N.z + k)];
                val += theWeight * tex3D<float>(f, k_shift, j, i);
                sum += theWeight;
            }
        }
    }

    if (d_DO_HIGH_PASS)
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = tex3D<float>(f, k, j, i) - val / sum;
    else
        f_filtered[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] = val / sum;
}
//########################################################################################################################################################

void setConstantMemoryParameters(const bool doHighPass)
{
    cudaMemcpyToSymbol(d_DO_HIGH_PASS, &doHighPass, sizeof(bool));
}

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    cudaSetDevice(whichGPU);
    setConstantMemoryParameters(false);
    return lowOrHighPassFilter(f, N_1, N_2, N_3, FWHM, numDims, axis, data_on_cpu, whichGPU, sliceStart, sliceEnd, f_out);
}

bool blurFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    cudaSetDevice(whichGPU);
    setConstantMemoryParameters(false);
    return lowOrHighPassFilter_txt(f, N_1, N_2, N_3, FWHM, numDims, axis, data_on_cpu, whichGPU, sliceStart, sliceEnd, f_out);
}

bool highPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    cudaSetDevice(whichGPU);
    setConstantMemoryParameters(true);
    return lowOrHighPassFilter(f, N_1, N_2, N_3, FWHM, numDims, axis, data_on_cpu, whichGPU, sliceStart, sliceEnd, f_out);
}

bool highPassFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    cudaSetDevice(whichGPU);
    setConstantMemoryParameters(true);
    return lowOrHighPassFilter_txt(f, N_1, N_2, N_3, FWHM, numDims, axis, data_on_cpu, whichGPU, sliceStart, sliceEnd, f_out);
}

bool lowOrHighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    return lowOrHighPassFilter_txt(f, N_1, N_2, N_3, FWHM, numDims, axis, data_on_cpu, whichGPU, sliceStart, sliceEnd, f_out);
    if (f == NULL) return false;

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

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (f_out != NULL && data_on_cpu == false)
    {
        dev_Df = f_out;
    }
    else
    {
        if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
            return false;
        }
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    if (numDims == 1)
        BlurFilter1DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM, axis);
    else if (numDims == 2)
        BlurFilter2DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM, axis);
    else
        BlurFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM, sliceStart, sliceEnd);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        if (f_out == NULL)
        {
            // copy dev_Df to dev_f
            cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
            //cudaDeviceSynchronize();
        }
    }
    if (dev_Df != 0 && (f_out == NULL || data_on_cpu == true))
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool lowOrHighPassFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    if (f == NULL) return false;

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

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N, false, false);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (f_out != NULL && data_on_cpu == false)
    {
        dev_Df = f_out;
    }
    else
    {
        if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
            return false;
        }
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    if (numDims == 1)
        BlurFilter1DKernel_txt <<<dimGrid, dimBlock >>> (d_data_txt, dev_Df, N, FWHM, axis);
    else if (numDims == 2)
        BlurFilter2DKernel_txt <<<dimGrid, dimBlock >>> (d_data_txt, dev_Df, N, FWHM, axis);
    else
        BlurFilterKernel_txt <<<dimGrid, dimBlock >>> (d_data_txt, dev_Df, N, FWHM, sliceStart, sliceEnd);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        if (f_out == NULL)
        {
            // copy dev_Df to dev_f
            cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
            //cudaDeviceSynchronize();
        }
    }
    if (dev_Df != 0 && (f_out == NULL || data_on_cpu == true))
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool momentFilter(float* f, int N_1, int N_2, int N_3, int r, int order, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    if (f == NULL) return false;

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    r = max(1, min(r, 100));

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    if (order == 1)
    {
        meanFilterKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, r, sliceStart, sliceEnd);
    }
    else
    {
        varianceFilterKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, r, sliceStart, sliceEnd);
    }

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu, int whichGPU, int sliceStart, int sliceEnd, float* f_out)
{
    if (f == NULL) return false;

    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;

    int windowRadius = max(1, min(2, (w - 1) / 2));

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    if (windowRadius == 2)
    {
        medianFilterKernel_5x5 <<<dimGrid, dimBlock >>> (dev_f, dev_Df, N, threshold, signalThreshold, sliceStart, sliceEnd);
    }
    else
    {
        medianFilterKernel <<<dimGrid, dimBlock >>> (dev_f, dev_Df, N, threshold, signalThreshold, sliceStart, sliceEnd);
    }

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool medianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu, int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    int windowRadius = max(1, min(3, (w-1)/2));

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    medianFilter2DKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, threshold, windowRadius, signalThreshold);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}

bool badPixelCorrection_gpu(float* g, parameters* params, float* badPixelMap, int w, bool data_on_cpu)
{
    if (g  == NULL || params == NULL || badPixelMap == NULL) return false;

    cudaSetDevice(params->whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
    float* dev_g = 0;
    float* dev_badPixelMap = 0;
    if (data_on_cpu)
    {
        dev_g = copy3DdataToGPU(g, N, params->whichGPU);
        dev_badPixelMap = copy3DdataToGPU(badPixelMap, make_int3(1, params->numRows, params->numCols), params->whichGPU);
    }
    else
    {
        dev_g = g;
        dev_badPixelMap = badPixelMap;
    }

    int windowRadius = max(1, min(3, (w - 1) / 2));

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    badPixelCorrectionKernel <<< dimGrid, dimBlock >>> (dev_g, dev_badPixelMap, N, windowRadius);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        pull3DdataFromGPU(g, N, dev_g, params->whichGPU);

        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_badPixelMap != 0)
            cudaFree(dev_badPixelMap);
    }

    return true;
}

bool azimuthalBlur(float* f, parameters* params, float filterWidth, bool data_on_cpu, float* f_out)
{
    if (f == NULL) return false;

    cudaSetDevice(params->whichGPU);
    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(params->numX, params->numY, params->numZ);
    float3 T = make_float3(params->voxelWidth, params->voxelWidth, params->voxelHeight);
    float3 startVal = make_float3(params->x_0(), params->y_0(), params->z_0());
    float* dev_f = 0;
    if (data_on_cpu)
    {
        //dev_f = copy3DdataToGPU(f, N, params->whichGPU);
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    }
    else
        dev_f = f;

    int N_phi_max = max(4, 2 * int(double(max(N.x, N.y))));

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N.x, N.y, N.z);
        return false;
    }

    float rFOVsq = params->rFOV() * params->rFOV();

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    azimuthalBlurKernel <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, T, startVal, N_phi_max, filterWidth, rFOVsq);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        if (f_out != NULL)
            pull3DdataFromGPU(f_out, N, dev_Df, params->whichGPU);
        else
            pullVolumeDataFromGPU(f, params, dev_Df, params->whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }

    return true;
}
