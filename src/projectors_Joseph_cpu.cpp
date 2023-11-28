////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for cpu projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "cpu_utils.h"
#include "projectors_Joseph_cpu.h"

using namespace std;

bool project_Joseph_cpu(float* g, float* f, parameters* params)
{
    if (params->geometry == parameters::MODULAR)
        return project_Joseph_modular_cpu(g, f, params);
    else
        return false;
}

bool backproject_Joseph_cpu(float* g, float* f, parameters* params)
{
    if (params->geometry == parameters::MODULAR)
        return backproject_Joseph_modular_cpu(g, f, params);
    else
        return false;
}

bool project_Joseph_modular_cpu(float* g, float* f, parameters* params)
{
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[iphi*params->numRows*params->numCols];
        float3 p = make_float3(params->sourcePositions[3 * iphi + 0], params->sourcePositions[3 * iphi + 1], params->sourcePositions[3 * iphi + 2]);
        float* c = &(params->moduleCenters[3 * iphi]);
        float* u_vec = &(params->colVectors[3 * iphi]);
        float* v_vec = &(params->rowVectors[3 * iphi]);

        for (int iRow = 0; iRow < params->numRows; iRow++)
        {
            float v = params->pixelHeight * iRow + params->v_0();

            float* aLine = &aProj[iRow*params->numCols];
            for (int iCol = 0; iCol < params->numCols; iCol++)
            {
                float u = params->pixelWidth * iCol + params->u_0();
                float3 dst = make_float3(c[0] + u * u_vec[0] + v * v_vec[0], c[1] + u * u_vec[1] + v * v_vec[1], c[2] + u * u_vec[2] + v * v_vec[2]);

                aLine[iCol] = projectLine_Joseph(f, params, p, dst);
            }
        }
    }
    return true;
}

bool backproject_Joseph_modular_cpu(float* g, float* f, parameters* params)
{
    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float z = iz * params->voxelHeight + params->z_0();
            float* zSlice = &f[iz*params->numX*params->numY];
            for (int iphi = 0; iphi < params->numAngles; iphi++)
            {
                float* aProj = &g[iphi*params->numRows*params->numCols];

                float* sourcePosition = &(params->sourcePositions[3 * iphi]);
                float* moduleCenter = &(params->moduleCenters[3 * iphi]);
                float* v_vec = &(params->rowVectors[3 * iphi]);
                float* u_vec = &(params->colVectors[3 * iphi]);

                for (int iy = 0; iy < params->numY; iy++)
                {
                    float y = iy * params->voxelWidth + params->y_0();
                    float* xLine = &zSlice[iy * params->numX];
                    for (int ix = 0; ix < params->numX; ix++)
                    {
                        float x = ix * params->voxelWidth + params->x_0();
                        if (iphi == 0)
                            xLine[ix] = 0.0;
                        xLine[ix] += backproject_Joseph_modular_kernel(aProj, params, sourcePosition, moduleCenter, u_vec, v_vec, x, y, z);
                    }
                }
            }
        }
    }
    else
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int ix = 0; ix < params->numX; ix++)
        {
            float x = ix * params->voxelWidth + params->x_0();
            float* xSlice = &f[ix * params->numZ * params->numY];
            for (int iphi = 0; iphi < params->numAngles; iphi++)
            {
                float* aProj = &g[iphi * params->numRows * params->numCols];

                float* sourcePosition = &(params->sourcePositions[3 * iphi]);
                float* moduleCenter = &(params->moduleCenters[3 * iphi]);
                float* v_vec = &(params->rowVectors[3 * iphi]);
                float* u_vec = &(params->colVectors[3 * iphi]);

                for (int iy = 0; iy < params->numY; iy++)
                {
                    float y = iy * params->voxelWidth + params->y_0();
                    float* zLine = &xSlice[iy * params->numZ];
                    for (int iz = 0; iz < params->numZ; iz++)
                    {
                        float z = iz * params->voxelHeight + params->z_0();
                        if (iphi == 0)
                            zLine[iz] = 0.0;
                        zLine[iz] += backproject_Joseph_modular_kernel(aProj, params, sourcePosition, moduleCenter, u_vec, v_vec, x, y, z);
                    }
                }
            }
        }
    }
    return true;
}

float backproject_Joseph_modular_kernel(float* aProj, parameters* params, float* sourcePosition, float* moduleCenter, float* u_vec, float* v_vec, float x, float y, float z)
{
    float val = 0.0;

    const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
        u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
        u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

    float T_x_inv = 1.0 / params->voxelWidth;

    float3 r = make_float3(x - sourcePosition[0], y - sourcePosition[1], z - sourcePosition[2]);
    const float R = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);

    const float3 p_minus_c = make_float3(sourcePosition[0] - moduleCenter[0], sourcePosition[1] - moduleCenter[1], sourcePosition[2] - moduleCenter[2]);
    const float p_minus_c_dot_n = p_minus_c.x * detNormal.x + p_minus_c.y * detNormal.y + p_minus_c.z * detNormal.z;
    const float D = -p_minus_c_dot_n / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);

    //<p_minus_c + lineLength*r, u>
    //<p_minus_c + lineLength*r, v>
    const float u_arg = (p_minus_c.x + D * r.x) * u_vec[0] + (p_minus_c.y + D * r.y) * u_vec[1] + (p_minus_c.z + D * r.z) * u_vec[2];
    const float v_arg = (p_minus_c.x + D * r.x) * v_vec[0] + (p_minus_c.y + D * r.y) * v_vec[1] + (p_minus_c.z + D * r.z) * v_vec[2];

    const int u_ind = int(floor(0.5 + (u_arg - params->u_0()) / params->pixelWidth));
    const int v_ind = int(floor(0.5 + (v_arg - params->v_0()) / params->pixelHeight));

    const int searchWidth_u = 1 + int(0.5 * params->voxelWidth / (R / D * params->pixelWidth));
    const int searchWidth_v = 1 + int(0.5 * params->voxelWidth / (R / D * params->pixelHeight));

    if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
    {
        const float rayWeight = R * params->voxelWidth / fabs(r.x);

        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows-1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * params->pixelWidth + params->u_0();

                const float t = r.x / (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);
                const float yy = sourcePosition[1] + t * (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                const float zz = sourcePosition[2] + t * (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);

                const float dy = max(0.0, 1.0 - fabs(y - yy) * T_x_inv);
                const float dz = max(0.0, 1.0 - fabs(z - zz) * T_x_inv);

                val += rayWeight * dy * dz * aProj[iv*params->numCols+iu];
            }
        }
    }
    else if (fabs(r.y) >= fabs(r.z))
    {
        const float rayWeight = R * params->voxelWidth / fabs(r.y);

        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * params->pixelWidth + params->u_0();

                const float t = r.y / (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                const float xx = sourcePosition[0] + t * (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);
                const float zz = sourcePosition[2] + t * (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);

                const float dx = max(0.0, 1.0 - fabs(x - xx) * T_x_inv);
                const float dz = max(0.0, 1.0 - fabs(z - zz) * T_x_inv);

                val += rayWeight * dx * dz * aProj[iv * params->numCols + iu];
            }
        }
    }
    else
    {
        const float rayWeight = R * params->voxelWidth / fabs(r.z);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * params->pixelWidth + params->u_0();

                const float t = r.z / (-p_minus_c.z + u * u_vec[2] + v * v_vec[2]);
                const float yy = sourcePosition[1] + t * (-p_minus_c.y + u * u_vec[1] + v * v_vec[1]);
                const float xx = sourcePosition[0] + t * (-p_minus_c.x + u * u_vec[0] + v * v_vec[0]);

                const float dy = max(0.0, 1.0 - fabs(y - yy) * T_x_inv);
                const float dx = max(0.0, 1.0 - fabs(x - xx) * T_x_inv);

                val += rayWeight * dy * dx * aProj[iv * params->numCols + iu];
            }
        }
    }
    return val;
}

float projectLine_Joseph(float* mu, parameters* params, float3 p, float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    float3 ip = make_float3((p.x - params->x_0()) / params->voxelWidth, (p.y - params->y_0()) / params->voxelWidth,
        (p.z - params->z_0()) / params->voxelWidth);
    float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    float half = 0.5;

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction

        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(params->numX - 1, int(floor(0.5 + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0;
        if (r.x > 0.0)
        {
            if (ip.x >= float(params->numX) - 0.5) return 0.0;
            int ix_max = min(params->numX - 1, int(ceil((dst.x - params->x_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, float(ix_start), iy_start, iz_start, params) *
                ((float(ix_start) - half) - max(-half, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start);
            const float iz_offset = iz_start - ir.z * float(ix_start);

            for (int ix = ix_start; ix <= ix_max; ix++)
                val += tex3D_rev(mu, float(ix), iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix), params);
        }
        else
        {
            if (ip.x <= -0.5) return 0.0;
            int ix_min = max(0, int(floor((dst.x - params->x_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, float(ix_start), iy_start, iz_start, params) *
                (min((float(params->numX) - half), ip.x) - (float(ix_start) + half));

            const float iy_offset = iy_start + ir.y * float(ix_start);
            const float iz_offset = iz_start + ir.z * float(ix_start);
            for (int ix = ix_start; ix >= ix_min; ix--)
                val += tex3D_rev(mu, float(ix), iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix), params);
        }
        return val * sqrt(1.0 + ir.y * ir.y + ir.z * ir.z) * params->voxelWidth;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(params->numY - 1, int(floor(0.5 + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0;
        if (r.y > 0.0)
        {
            if (ip.y >= float(params->numY) - 0.5) return 0.0;
            int iy_max = min(params->numY - 1, int(ceil((dst.y - params->y_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, ix_start, float(iy_start), iz_start, params) *
                ((float(iy_start) - half) - max(-half, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start);
            const float iz_offset = iz_start - ir.z * float(iy_start);
            for (int iy = iy_start; iy <= iy_max; iy++)
                val += tex3D_rev(mu, ix_offset + ir.x * float(iy), float(iy), iz_offset + ir.z * float(iy), params);
        }
        else
        {
            if (ip.y <= -0.5) return 0.0;
            int iy_min = max(0, int(floor((dst.y - params->y_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, ix_start, iy_start, iz_start, params) *
                (min((float(params->numY) - half), ip.y) - (float(iy_start) + half));

            const float ix_offset = ix_start + ir.x * float(iy_start);
            const float iz_offset = iz_start + ir.z * float(iy_start);
            for (int iy = iy_start; iy >= iy_min; iy--)
                val += tex3D_rev(mu, ix_offset - ir.x * float(iy), float(iy), iz_offset - ir.z * float(iy), params);
        }
        return val * sqrt(1.0 + ir.x * ir.x + ir.z * ir.z) * params->voxelWidth;
    }
    else
    {
        // integral in z direction
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(params->numZ - 1, int(floor(0.5 + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0;
        if (r.z > 0.0)
        {
            if (ip.z >= float(params->numZ) - 0.5) return 0.0;
            int iz_max = min(params->numZ - 1, int(ceil((dst.z - params->z_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, ix_start, iy_start, iz_start, params) *
                ((float(iz_start) - half) - max(-half, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start);
            const float iy_offset = iy_start - ir.y * float(iz_start);
            for (int iz = iz_start; iz <= iz_max; iz++)
                val += tex3D_rev(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz), params);
        }
        else
        {
            if (ip.z <= -0.5) return 0.0;
            int iz_min = max(0, int(floor((dst.z - params->z_0()) / params->voxelWidth)));

            val = tex3D_rev(mu, ix_start, iy_start, float(iz_start), params) *
                (min((float(params->numZ) - half), ip.z) - (float(iz_start) + half));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5;
            for (int iz = iz_start; iz >= iz_min; iz--)
                val += tex3D_rev(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz), params);
        }
        return val * sqrt(1.0 + ir.x * ir.x + ir.y * ir.y) * params->voxelWidth;
    }
}
