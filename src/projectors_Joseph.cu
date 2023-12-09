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
#include "device_launch_parameters.h"
#include "projectors.h"
#include "projectors_Joseph.cuh"
#include "cuda_utils.h"

__device__ float lineIntegral_Joseph_ZYX(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
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

__global__ void modularBeamJosephBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder)
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

    const float T_x_inv = 1.0f / T_f.x;

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
        const float R = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

        const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
        const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
        const float D = -p_minus_c_dot_n / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);

        //<p_minus_c + lineLength*r, u>
        //<p_minus_c + lineLength*r, v>
        const float u_arg = (p_minus_c.x + D * r.x) * u_vec[0] + (p_minus_c.y + D * r.y) * u_vec[1] + (p_minus_c.z + D * r.z) * u_vec[2];
        const float v_arg = (p_minus_c.x + D * r.x) * v_vec[0] + (p_minus_c.y + D * r.y) * v_vec[1] + (p_minus_c.z + D * r.z) * v_vec[2];

        const int u_ind = int(floor(0.5f + (u_arg - startVals_g.z) / T_g.z));
        const int v_ind = int(floor(0.5f + (v_arg - startVals_g.y) / T_g.y));

        const int searchWidth_u = 1 + int(0.5f * T_f.x / (R / D * T_g.z));
        const int searchWidth_v = 1 + int(0.5f * T_f.z / (R / D * T_g.y));

        if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
        {
            const float rayWeight = R * T_f.x / fabs(r.x);

            for (int iv = v_ind - searchWidth_v; iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = u_ind - searchWidth_u; iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float t = r.x / (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);
                    const float yy = sourcePosition[1] + t * (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                    const float zz = sourcePosition[2] + t * (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);

                    const float dy = max(0.0f, 1.0f - fabs(y - yy) * T_x_inv);
                    const float dz = max(0.0f, 1.0f - fabs(z - zz) * T_x_inv);

                    val += rayWeight * dy * dz * tex3D<float>(g, iu, iv, iphi);
                }
            }
        }
        else if (fabs(r.y) >= fabs(r.z))
        {
            const float rayWeight = R * T_f.x / fabs(r.y);

            for (int iv = v_ind - searchWidth_v; iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = u_ind - searchWidth_u; iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float t = r.y / (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                    const float xx = sourcePosition[0] + t * (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);
                    const float zz = sourcePosition[2] + t * (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);

                    const float dx = max(0.0f, 1.0f - fabs(x - xx) * T_x_inv);
                    const float dz = max(0.0f, 1.0f - fabs(z - zz) * T_x_inv);

                    val += rayWeight * dx * dz * tex3D<float>(g, iu, iv, iphi);
                }
            }
        }
        else
        {
            const float rayWeight = R * T_f.x / fabs(r.z);
            for (int iv = v_ind - searchWidth_v; iv <= v_ind + searchWidth_v; iv++)
            {
                const float v = iv * T_g.y + startVals_g.y;
                for (int iu = u_ind - searchWidth_u; iu <= u_ind + searchWidth_u; iu++)
                {
                    const float u = iu * T_g.z + startVals_g.z;

                    const float t = r.z / (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);
                    const float yy = sourcePosition[1] + t * (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                    const float xx = sourcePosition[0] + t * (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);

                    const float dy = max(0.0f, 1.0f - fabs(y - yy) * T_x_inv);
                    const float dx = max(0.0f, 1.0f - fabs(x - xx) * T_x_inv);

                    val += rayWeight * dy * dx * tex3D<float>(g, iu, iv, iphi);
                }
            }
        }
    }
    f[ind] = val;
}

__global__ void modularBeamJosephProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
        return;

    float v = j * T_g.y + startVals_g.y;
    float u = k * T_g.z + startVals_g.z;

    float3 sourcePos;
    sourcePos.x = sourcePositions[3 * i + 0];
    sourcePos.y = sourcePositions[3 * i + 1];
    sourcePos.z = sourcePositions[3 * i + 2];

    float3 dst;
    dst.x = moduleCenters[3 * i + 0] + v * rowVectors[3 * i + 0] + u * colVectors[3 * i + 0];
    dst.y = moduleCenters[3 * i + 1] + v * rowVectors[3 * i + 1] + u * colVectors[3 * i + 1];
    dst.z = moduleCenters[3 * i + 2] + v * rowVectors[3 * i + 2] + u * colVectors[3 * i + 2];

    if (volumeDimensionOrder == 0)
        g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] = lineIntegral_Joseph_XYZ(f, N_f, T_f, startVals_f, sourcePos, dst);
    else
        g[uint64(i) * uint64(N_g.y * N_g.z) + uint64(j * N_g.z + k)] = lineIntegral_Joseph_ZYX(f, N_f, T_f, startVals_f, sourcePos, dst);
}

bool project_Joseph(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    if (params->geometry == parameters::MODULAR)
        return project_Joseph_modular(g, f, params, cpu_to_gpu);
    else
        return false;
}

bool backproject_Joseph(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    if (params->geometry == parameters::MODULAR)
        return backproject_Joseph_modular(g, f, params, cpu_to_gpu);
    else
        return false;
}

bool project_Joseph_modular(float*& g, float* f, parameters* params, bool cpu_to_gpu)
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

    if (cpu_to_gpu)
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

    if (cpu_to_gpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    modularBeamJosephProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);

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
    cudaFree(dev_sourcePositions);
    cudaFree(dev_moduleCenters);
    cudaFree(dev_rowVectors);
    cudaFree(dev_colVectors);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

bool backproject_Joseph_modular(float* g, float*& f, parameters* params, bool cpu_to_gpu)
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

    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
        }
    }
    else
        dev_f = f;

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

    if (cpu_to_gpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, false);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    modularBeamJosephBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);

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
    cudaFree(dev_sourcePositions);
    cudaFree(dev_moduleCenters);
    cudaFree(dev_rowVectors);
    cudaFree(dev_colVectors);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}
