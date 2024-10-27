////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for projectors of the Attenuated Radon Transform
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors_attenuated.cuh"
#include "cuda_utils.h"
//using namespace std;

__device__ float lineIntegral(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
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
            for (int iy = iy_start; iy <= iy_max; iy++)
                val += tex3D<float>(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
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

__device__ float lineIntegral_XYZ(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
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

__global__ void attenuatedWeightedBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, cudaTextureObject_t mu, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
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

    const float x_shift = 0.5f * T_f.x;
    //const float x_shift = T_f.x;
    //const float x_shift = 2.0f * T_f.x;

    /*
    if (i == 0 && j == 0 && k == 0)
    {
        const float x = -140.0f;
        const float y = 0.0f;
        const float z = 0.0f;
        const float D = T_f.x * max(N_f.x, N_f.y);
        const float cos_phi = -1.0f;
        const float sin_phi = 0.0f;
        float test = lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x, y, z), make_float3(x - D * cos_phi, y - D * sin_phi, z));
        //float test = lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x - D * cos_phi, y - D * sin_phi, z), make_float3(x, y, z));

        const float muCoeff = 0.01f;
        const float muRadius = 150.0f;
        const float s = -sin_phi * x + cos_phi * y;
        const float t = -x * cos_phi - y * sin_phi;
        float expTerm = 0.0f;
        if (s*s+t*t < muRadius* muRadius)
        {
            //expTerm = sqrt(muRadius * muRadius - s * s);
            //expTerm = max(0.0f, expTerm + min(expTerm, -x * cos_phi - y * sin_phi));
            expTerm = sqrt(muRadius * muRadius - s * s) - t;
        }
        //expTerm = expf(-muCoeff * expTerm);

        printf("T = %f, %f, %f\n", T_f.x, T_f.y, T_f.z);
        printf("(%f, %f, %f) to (%f, %f, %f)\n", x, y, z, x - D * cos_phi, y - D * sin_phi, z);
        printf("test = %f ?= %f (true)\n", test/0.01f, expTerm);
    }
    //*/

    if (sqrt(x * x + y * y) >= sqrt(rFOVsq) - x_shift)
    {
        f[ind] = 0.0f;
        return;
    }

    const float K = float(k) + 0.5f;
    const float D = -T_f.x * max(N_f.x, N_f.y);

    float val = 0.0;
    // loop over projection angles
    for (int l = 0; l < N_g.x; l++)
    {
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);
        const float L = float(l) + 0.5f;

        /*
        float expTerm = 0.0;
        if (volumeDimensionOrder == 1)
            expTerm = lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x, y, z), make_float3(x - D * cos_phi, y - D * sin_phi, z));
        else
            expTerm = lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x, y, z), make_float3(x - D * cos_phi, y - D * sin_phi, z));
        expTerm = expf(expTerm);

        const float s_arg = y * cos_phi - x * sin_phi;
        const float s_ind = (s_arg - startVals_g.z) / T_g.z;

        const float s_ind_x_shift = -sin_phi * x_shift / T_g.z;
        const float s_ind_y_shift = cos_phi * x_shift / T_g.z;

        val += (-sin_phi * (tex3D<float>(g, s_ind + s_ind_x_shift, K, L) - tex3D<float>(g, s_ind - s_ind_x_shift, K, L))
            + cos_phi * (tex3D<float>(g, s_ind + s_ind_y_shift, K, L) - tex3D<float>(g, s_ind - s_ind_y_shift, K, L))) * expTerm;
        //*/

        //*
        float4 expTerms;
        if (volumeDimensionOrder == 1)
        {
            expTerms.w = expf(lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x + x_shift, y, z), make_float3(x + x_shift - D * cos_phi, y - D * sin_phi, z)));
            expTerms.x = expf(lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x - x_shift, y, z), make_float3(x - x_shift - D * cos_phi, y - D * sin_phi, z)));
            expTerms.y = expf(lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x, y + x_shift, z), make_float3(x - D * cos_phi, y + x_shift - D * sin_phi, z)));
            expTerms.z = expf(lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x, y - x_shift, z), make_float3(x - D * cos_phi, y - x_shift - D * sin_phi, z)));
        }
        else
        {
            expTerms.w = expf(lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x + x_shift, y, z), make_float3(x + x_shift - D * cos_phi, y - D * sin_phi, z)));
            expTerms.x = expf(lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x - x_shift, y, z), make_float3(x - x_shift - D * cos_phi, y - D * sin_phi, z)));
            expTerms.y = expf(lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x, y + x_shift, z), make_float3(x - D * cos_phi, y + x_shift - D * sin_phi, z)));
            expTerms.z = expf(lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x, y - x_shift, z), make_float3(x - D * cos_phi, y - x_shift - D * sin_phi, z)));
        }

        const float s_arg = y * cos_phi - x * sin_phi;
        const float s_ind = (s_arg - startVals_g.z) / T_g.z;

        const float s_ind_x_shift = -sin_phi * x_shift / T_g.z;
        const float s_ind_y_shift = cos_phi * x_shift / T_g.z;

        val += -sin_phi * (tex3D<float>(g, s_ind + s_ind_x_shift, K, L) * expTerms.w - tex3D<float>(g, s_ind - s_ind_x_shift, K, L) * expTerms.x)
            + cos_phi * (tex3D<float>(g, s_ind + s_ind_y_shift, K, L) * expTerms.y - tex3D<float>(g, s_ind - s_ind_y_shift, K, L) * expTerms.z);
        //*/
    }

    f[ind] = val * T_f.x / (4.0f * x_shift);
}

__global__ void attenuatedBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, cudaTextureObject_t mu, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
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

    const float T_u_inv = 1.0f / T_g.z;
    const float C_num = 0.5f * T_u_inv * T_f.x;
    const float C_num_T_x = T_f.x * C_num;
    const float x_mult = x * T_u_inv;
    const float y_mult = y * T_u_inv;
    const float s_shift = -startVals_g.z * T_u_inv;
    float cos_phi, sin_phi, C, s_arg, ds;

    const float D = -T_f.x * max(N_f.x, N_f.y);

    float val = 0.0;
    // loop over projection angles
    int l = 0;
    while (l < N_g.x)
    {
        /*
        if (l + 3 < N_g.x)
        {
            const int l1 = l + 1;
            const int l2 = l + 2;
            const int l3 = l + 3;

            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            sin_phi = sin(phis[l1]);
            cos_phi = cos(phis[l1]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_B = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            sin_phi = sin(phis[l2]);
            cos_phi = cos(phis[l2]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_C = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            sin_phi = sin(phis[l3]);
            cos_phi = cos(phis[l3]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_D = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            // Do texture mapping
            val += tex3D<float>(g, s_ind_A, float(k) + 0.5f, float(l) + 0.5f)
                + tex3D<float>(g, s_ind_B, float(k) + 0.5f, float(l1) + 0.5f)
                + tex3D<float>(g, s_ind_C, float(k) + 0.5f, float(l2) + 0.5f)
                + tex3D<float>(g, s_ind_D, float(k) + 0.5f, float(l3) + 0.5f);
            l += 4;
        }
        else if (l + 1 < N_g.x)
        {
            int l1 = l + 1;

            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            sin_phi = sin(phis[l1]);
            cos_phi = cos(phis[l1]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_B = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            val += tex3D<float>(g, s_ind_A, float(k) + 0.5f, float(l) + 0.5f)
                + tex3D<float>(g, s_ind_B, float(k) + 0.5f, float(l1) + 0.5f);
            l += 2;
        }
        else //if (l+1 < N_g.x)
        { //*/
            sin_phi = sin(phis[l]);
            cos_phi = cos(phis[l]);
            C = C_num * max(fabs(cos_phi), fabs(sin_phi));
            s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
            ds = modf(s_arg, &s_arg);
            const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

            float expTerm = 0.0;
            if (volumeDimensionOrder == 1)
            {
                expTerm = lineIntegral(mu, N_f, T_f, startVals_f, make_float3(x,y,z), make_float3(x-D*cos_phi,y-D*sin_phi,z));
            }
            else
            {
                expTerm = lineIntegral_XYZ(mu, N_f, T_f, startVals_f, make_float3(x, y, z), make_float3(x - D * cos_phi, y - D * sin_phi, z));
            }

            val += tex3D<float>(g, s_ind_A, float(k) + 0.5f, float(l) + 0.5f) * expf(-expTerm);
            l += 1;
        //}
    }

    f[ind] = val * maxWeight;
}

__global__ void attenuatedProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, cudaTextureObject_t mu, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    //const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const float iz = float(m) + 0.5f;
    const float Tx_inv = 1.0f / T_f.x;

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

    float attenExpCoeff = 0.0f;
    float g_output = 0.0f;
    if (fabs(cos_phi) > fabs(sin_phi))
    {
        const float cos_phi_inv = 1.0f / cos_phi;

        const float ds_ind_dj_inv = 1.0f / ds_ind_dj;
        float shiftConstant;
        if (ds_ind_dj > 0.0f)
            shiftConstant = (n_minus_half - C) * ds_ind_dj_inv;
        else
            shiftConstant = (n_plus_half + C) * ds_ind_dj_inv;

        float attenExpCoeff_prev = 0.0f;
        for (int ii = 0; ii < N_f.x; ii++)
        {
            int i = ii;
            if (cos_phi > 0.0f)
                i = N_f.x - 1 - ii;

            // Terms for attenuation factor
            const float ix = float(i) + 0.5f;
            const float x = float(i) * T_f.x + startVals_f.x;
            float attenExpCoeff_inc;
            if (volumeDimensionOrder == 0)
                attenExpCoeff_inc = l_phi * tex3D<float>(mu, iz, ((u + x * sin_phi) * cos_phi_inv - startVals_f.y) * Tx_inv + 0.5f, ix);
            else
                attenExpCoeff_inc = l_phi * tex3D<float>(mu, ix, ((u + x * sin_phi) * cos_phi_inv - startVals_f.y) * Tx_inv + 0.5f, iz);

            if (ii == 0)
                attenExpCoeff = 0.5f * (attenExpCoeff_inc + attenExpCoeff_prev);
            else
                attenExpCoeff += 0.5f * (attenExpCoeff_inc + attenExpCoeff_prev);
            attenExpCoeff_prev = attenExpCoeff_inc;
            // End Terms for attenuation factor

            const float s_ind_base = (float)i * ds_ind_di + s_ind_offset;
            const int j_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_dj_inv);
            const float s_ind_A = s_ind_base + (float)j_min_A * ds_ind_dj;

            if (x * x + ((float)j_min_A * T_f.y + startVals_f.y) * ((float)j_min_A * T_f.y + startVals_f.y) > rFOVsq)
                continue;

            if (volumeDimensionOrder == 0)
            {
                g_output += (max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, m, j_min_A, i)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C)) * tex3D<float>(f, m, j_min_A + 1, i)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, m, j_min_A + 2, i))
                    * expf(-attenExpCoeff);
            }
            else
            {
                g_output += (max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, i, j_min_A, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C)) * tex3D<float>(f, i, j_min_A + 1, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, i, j_min_A + 2, m))
                    * expf(-attenExpCoeff);
            }
        }
    }
    else
    {
        const float sin_phi_inv = 1.0f / sin_phi;

        const float ds_ind_di_inv = 1.0f / ds_ind_di;
        float shiftConstant;
        if (ds_ind_di > 0.0f)
            shiftConstant = (n_minus_half - C) * ds_ind_di_inv;
        else
            shiftConstant = (n_plus_half + C) * ds_ind_di_inv;
        float attenExpCoeff_prev = 0.0f;
        for (int jj = 0; jj < N_f.y; jj++)
        {
            int j = jj;
            if (sin_phi > 0.0f)
                j = N_f.y - 1 - jj;

            // Terms for attenuation factor
            const float iy = float(j) + 0.5f;
            const float y = float(j) * T_f.y + startVals_f.y;
            float attenExpCoeff_inc;
            if (volumeDimensionOrder == 0)
                attenExpCoeff_inc = l_phi * tex3D<float>(mu, iz, iy, ((y * cos_phi - u) * sin_phi_inv - startVals_f.x) * Tx_inv + 0.5f);
            else
                attenExpCoeff_inc = l_phi * tex3D<float>(mu, ((y * cos_phi - u) * sin_phi_inv - startVals_f.x) * Tx_inv + 0.5f, iy, iz);

            if (jj == 0)
                attenExpCoeff = 0.5f * (attenExpCoeff_inc + attenExpCoeff_prev);
            else
                attenExpCoeff += 0.5f * (attenExpCoeff_inc + attenExpCoeff_prev);
            attenExpCoeff_prev = attenExpCoeff_inc;
            // End Terms for attenuation factor

            const float s_ind_base = (float)j * ds_ind_dj + s_ind_offset;
            const int i_min_A = (int)ceil(shiftConstant - s_ind_base * ds_ind_di_inv);
            const float s_ind_A = s_ind_base + (float)i_min_A * ds_ind_di;

            if (((float)i_min_A * T_f.x + startVals_f.x) * ((float)i_min_A * T_f.x + startVals_f.x) + ((float)j * T_f.y + startVals_f.y) * ((float)j * T_f.y + startVals_f.y) > rFOVsq)
                continue;

            if (volumeDimensionOrder == 0)
            {
                g_output += (max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, m, j, i_min_A)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C)) * tex3D<float>(f, m, j, i_min_A + 1)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, m, j, i_min_A + 2))
                    * expf(-attenExpCoeff);
            }
            else
            {
                g_output += (max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C)) * tex3D<float>(f, i_min_A, j, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C)) * tex3D<float>(f, i_min_A + 1, j, m)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, i_min_A + 2, j, m))
                    * expf(-attenExpCoeff);
            }
        }
    }
    g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = l_phi * g_output;
}

__global__ void cylindricalAttenuatedBackprojectorKernel_SF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, const float muCoeff, const float muRadius, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
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

    if (x * x + y * y > rFOVsq)
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

    //const float D = T_f.x * max(N_f.x, N_f.y);

    float val = 0.0;
    // loop over projection angles
    int l = 0;
    while (l < N_g.x)
    {
        sin_phi = sin(phis[l]);
        cos_phi = cos(phis[l]);
        C = C_num * max(fabs(cos_phi), fabs(sin_phi));
        s_arg = s_shift - sin_phi * x_mult + cos_phi * y_mult;
        ds = modf(s_arg, &s_arg);
        const float s_ind_A = s_arg - (C_num_T_x / C * max(0.0f, (min(0.5f, C + ds) + min(0.5f, C - ds)))) * maxWeight_inv + 1.5f;

        const float s = -sin_phi * x + cos_phi * y;
        float expTerm = 0.0f;
        /*
        if (fabs(s) < muRadius)
        {
            expTerm = sqrt(muRadius * muRadius - s * s);
            expTerm = expTerm + min(expTerm, -x * cos_phi - y * sin_phi);
        }
        //*/
        const float t = x * cos_phi + y * sin_phi;
        if (s * s + t * t < muRadius * muRadius)
        {
            //expTerm = sqrt(muRadius * muRadius - s * s);
            //expTerm = max(0.0f, expTerm + min(expTerm, -x * cos_phi - y * sin_phi));
            expTerm = exp(-muCoeff*(sqrt(muRadius * muRadius - s * s) - t));
        }
        //expTerm = -x * cos_phi - y * sin_phi; // ERT
        
        val += tex3D<float>(g, s_ind_A, float(k) + 0.5f, float(l) + 0.5f) * expTerm;
        l += 1;
    }

    f[ind] = val * maxWeight;
}

__global__ void cylindricalAttenuatedProjectorKernel_SF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, const float muCoeff, const float muRadius, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    //const float v = m * T_g.y + startVals_g.y;
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

    float g_output = 0.0f;
    if (fabs(cos_phi) > fabs(sin_phi))
    {
        const float cos_phi_inv = 1.0f / cos_phi;

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

            if (((float)i * T_f.x + startVals_f.x) * ((float)i * T_f.x + startVals_f.x) + ((float)j_min_A * T_f.y + startVals_f.y) * ((float)j_min_A * T_f.y + startVals_f.y) > rFOVsq)
                continue;

            //u = -sin_phi * x + cos_phi * y
            const float x = float(i) * T_f.x + startVals_f.x;
            const float y = (u + sin_phi * x) * cos_phi_inv;

            // Terms for attenuation factor
            float expTerm = 0.0f;
            const float t = x * cos_phi + y * sin_phi;
            if (u * u + t * t < muRadius * muRadius)
                expTerm = exp(-muCoeff * (sqrt(muRadius * muRadius - u * u) - t));

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_dj + C) - max(n_minus_half, s_ind_A + ds_ind_dj - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += ((weight_0 + weight_1) * tex3D<float>(f, float(m) + 0.5f, float(j_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(i) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, float(m) + 0.5f, float(j_min_A + 2) + 0.5f, float(i) + 0.5f)) * expTerm;
            }
            else
            {
                g_output += ((weight_0 + weight_1) * tex3D<float>(f, float(i) + 0.5f, float(j_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_dj + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_dj - C)) * tex3D<float>(f, float(i) + 0.5f, float(j_min_A + 2) + 0.5f, float(m) + 0.5f)) * expTerm;
            }
        }
    }
    else
    {
        const float sin_phi_inv = 1.0f / sin_phi;

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

            if (((float)i_min_A * T_f.x + startVals_f.x) * ((float)i_min_A * T_f.x + startVals_f.x) + ((float)j * T_f.y + startVals_f.y) * ((float)j * T_f.y + startVals_f.y) > rFOVsq)
                continue;

            //u = -sin_phi * x + cos_phi * y
            const float y = float(j) * T_f.y + startVals_f.y;
            const float x = (cos_phi * y - u) * sin_phi_inv;

            // Terms for attenuation factor
            float expTerm = 0.0f;
            const float t = x * cos_phi + y * sin_phi;
            if (u * u + t * t < muRadius * muRadius)
                expTerm = exp(-muCoeff * (sqrt(muRadius * muRadius - u * u) - t));

            const float weight_0 = max(0.0f, min(n_plus_half, s_ind_A + C) - max(n_minus_half, s_ind_A - C));
            const float weight_1 = max(0.0f, min(n_plus_half, s_ind_A + ds_ind_di + C) - max(n_minus_half, s_ind_A + ds_ind_di - C));
            if (volumeDimensionOrder == 0)
            {
                g_output += ((weight_0 + weight_1) * tex3D<float>(f, float(m) + 0.5f, float(j) + 0.5f, float(i_min_A) + 0.5f + weight_1 / (weight_0 + weight_1))
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, float(m) + 0.5f, float(j) + 0.5f, float(i_min_A + 2) + 0.5f)) * expTerm;
            }
            else
            {
                g_output += ((weight_0 + weight_1) * tex3D<float>(f, float(i_min_A) + 0.5f + weight_1 / (weight_0 + weight_1), float(j) + 0.5f, float(m) + 0.5f)
                    + max(0.0f, min(n_plus_half, s_ind_A + 2.0f * ds_ind_di + C) - max(n_minus_half, s_ind_A + 2.0f * ds_ind_di - C)) * tex3D<float>(f, float(i_min_A + 2) + 0.5f, float(j) + 0.5f, float(m) + 0.5f)) * expTerm;
            }
        }
    }
    g[uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n)] = l_phi * g_output;
}

bool project_attenuated(float*& g, float* f, parameters* params, bool data_on_cpu)
{
    if (g == NULL || f == NULL || params == NULL || params->muSpecified() == false || params->allDefined() == false)
        return false;
    if (params->voxelSizeWorksForFastSF() == false)
    {
        //printf("using extended\n");
        //return project_eSF(g, f, params, data_on_cpu);
        return false;
    }
    if (params->geometry != parameters::PARALLEL)
    {
        printf("Error: attenuated backprojection only works for parallel-beam data!\n");
        return false;
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;
    float* dev_mu = 0;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float rFOVsq = params->rFOV() * params->rFOV();

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
    d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
    //*/
    //*
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
    else
        d_data_array = loadTexture(d_data_txt, f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
    //*/

    cudaTextureObject_t d_mu_txt = NULL;
    cudaArray* d_mu_array = NULL;
    if (params->mu != NULL)
    {
        if (data_on_cpu)
            dev_mu = copyVolumeDataToGPU(params->mu, params, params->whichGPU);
        else
            dev_mu = params->mu;

        d_mu_array = loadTexture(d_mu_txt, dev_mu, N_f, false, true, bool(params->volumeDimensionOrder == 1));
    }

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->mu != NULL)
        attenuatedProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, d_mu_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    else
        cylindricalAttenuatedProjectorKernel_SF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, params->muCoeff, params->muRadius, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);

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
    if (params->mu != NULL)
    {
        cudaFreeArray(d_mu_array);
        cudaDestroyTextureObject(d_mu_txt);
    }
    cudaFree(dev_phis);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
        if (dev_mu != 0)
            cudaFree(dev_mu);
    }

    return true;
}

bool backproject_attenuated(float* g, float*& f, parameters* params, bool data_on_cpu)
{
    if (g == NULL || f == NULL || params == NULL || params->muSpecified() == false || params->allDefined() == false)
        return false;
    if (params->geometry != parameters::PARALLEL)
    {
        printf("Error: attenuated backprojection only works for parallel-beam data!\n");
        return false;
    }
    if (params->voxelSizeWorksForFastSF() == false)
    {
        //printf("using extended\n");
        //return backproject_eSF_cone(g, f, params, data_on_cpu);
        return false;
    }

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;
    float* dev_mu = 0;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    float* dev_phis = copyAngleArrayToGPU(params);

    //if (params->doWeightedBackprojection)
    //    params->colShiftFromFilter = 0.5;
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);
    //if (params->doWeightedBackprojection)
    //    params->colShiftFromFilter = 0.0;

    float rFOVsq = params->rFOV() * params->rFOV();

    if (data_on_cpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    cudaTextureObject_t d_mu_txt = NULL;
    cudaArray* d_mu_array = NULL;
    if (params->mu != NULL)
    {
        if (data_on_cpu)
            dev_mu = copyVolumeDataToGPU(params->mu, params, params->whichGPU);
        else
            dev_mu = params->mu;

        d_mu_array = loadTexture(d_mu_txt, dev_mu, N_f, false, true, bool(params->volumeDimensionOrder == 1));
    }

    dim3 dimBlock_g = setBlockSize(N_g);
    dim3 dimGrid_g = setGridSize(N_g, dimBlock_g);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, true);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        dev_g = 0;
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
    if (params->mu != NULL)
    {
        if (params->doWeightedBackprojection)
            attenuatedWeightedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, d_mu_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
        else
            attenuatedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, d_mu_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
    }
    else
        cylindricalAttenuatedBackprojectorKernel_SF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, params->muCoeff, params->muRadius, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (data_on_cpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    if (params->mu != NULL)
    {
        cudaFreeArray(d_mu_array);
        cudaDestroyTextureObject(d_mu_txt);
    }
    cudaFree(dev_phis);

    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
        if (dev_mu != 0)
            cudaFree(dev_mu);
    }

    return true;
}
