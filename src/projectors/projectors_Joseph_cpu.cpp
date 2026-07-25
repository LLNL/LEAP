////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for Joseph CPU projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "cpu_utils.h"
#include "projectors_Joseph_cpu.h"

using namespace std;

#ifdef __USE_CPU

float3 make_float3(float x, float y, float z)
{
    float3 retVal;
    retVal.x = x;
    retVal.y = y;
    retVal.z = z;
    return retVal;
}

double3 make_double3(double x, double y, double z)
{
    double3 retVal;
    retVal.x = x;
    retVal.y = y;
    retVal.z = z;
    return retVal;
}
#endif

bool usingSFprojectorsForModularBeam(parameters* params)
{
    if (params->whichGPU < 0)
        return false;
    else
    {
        if (params->modularbeamIsAxiallyAligned() == true /* && params->voxelSizeWorksForFastSF() == true*/)
            return true;
        else
            return false;
    }
}

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
    double v_0 = double(params->v_0());
    double u_0 = double(params->u_0());

    double x_0 = double(params->x_0());
    double y_0 = double(params->y_0());
    double z_0 = double(params->z_0());

    double x_end = double(params->numX - 1) * double(params->voxelWidth) + x_0;
    double y_end = double(params->numY - 1) * double(params->voxelWidth) + y_0;
    double z_end = double(params->numZ - 1) * double(params->voxelHeight) + z_0;

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* sourcePos = &(params->sourcePositions[3 * iphi]);
        float* c = &(params->moduleCenters[3 * iphi]);
        float* u_vec = &(params->colVectors[3 * iphi]);
        float* v_vec = &(params->rowVectors[3 * iphi]);

        float* aProj = &g[uint64(iphi)* uint64(params->numRows*params->numCols)];
        float3 p = make_float3(sourcePos[0], sourcePos[1], sourcePos[2]);

        for (int iRow = 0; iRow < params->numRows; iRow++)
        {
            double v = double(params->pixelHeight) * double(iRow) + v_0;

            float* aLine = &aProj[iRow*params->numCols];
            for (int iCol = 0; iCol < params->numCols; iCol++)
            {
                double u = double(params->pixelWidth) * double(iCol) + u_0;
                float3 dst = make_float3(c[0] + u * u_vec[0] + v * v_vec[0], c[1] + u * u_vec[1] + v * v_vec[1], c[2] + u * u_vec[2] + v * v_vec[2]);

                const double3 r = make_double3(double(dst.x) - double(p.x), double(dst.y) - double(p.y), double(dst.z) - double(p.z));
                double t = 0.0f;
                if (fabs(r.x) > max(fabs(r.y), fabs(r.z)))
                {
                    if (r.x < 0.0f)
                        t = (x_end - double(p.x)) / r.x;
                    else
                        t = (x_0 - double(p.x)) / r.x;
                }
                else if (fabs(r.y) > fabs(r.z))
                {
                    if (r.y < 0.0f)
                        t = (y_end - double(p.y)) / r.y;
                    else
                        t = double(y_0 - double(p.y)) / r.y;
                }
                else
                {
                    if (r.z < 0.0f)
                        t = (z_end - double(p.z)) / r.z;
                    else
                        t = (z_0 - double(p.z)) / r.z;
                }
                const float3 edgePos = make_float3(float(double(p.x) + t * r.x), float(double(p.y) + t * r.y), float(double(p.z) + t * r.z));

                aLine[iCol] = projectLine_Joseph(f, params, edgePos, dst);
            }
        }
    }
    return true;
}

bool backproject_Joseph_modular_cpu(float* g, float* f, parameters* params)
{
    double alpha = max(atan(0.5 * (params->numCols - 1) * params->pixelWidth / params->sdd), atan(0.5 * (params->numRows - 1) * params->pixelHeight / params->sdd));
    double rFOV = max(fabs(params->furthestFromCenter()), max(fabs(params->z_0()), fabs(params->z_samples(params->numZ - 1))));
    double maxDivergence = tan(alpha) * (params->sdd + rFOV) - tan(alpha) * (params->sdd - rFOV);
    //double maxTravel = maxDivergence / min(params->pixelWidth, params->pixelHeight);
    double maxTravel = maxDivergence / params->voxelWidth;
    //printf("maxTravel = %f\n", maxTravel);
    
    bool doParallel = false;
    if (maxTravel < 0.25 && params->truncatedScan == false)
    {
        //printf("do parallel!\n");
        doParallel = true;
    }

    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float z = iz * params->voxelHeight + params->z_0();
            float* zSlice = &f[uint64(iz)* uint64(params->numX*params->numY)];
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
                        if (doParallel)
                            xLine[ix] += backproject_Joseph_modular_parallel_kernel(aProj, params, sourcePosition, moduleCenter, u_vec, v_vec, x, y, z);
                        else
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
            float* xSlice = &f[uint64(ix) * uint64(params->numZ * params->numY)];
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
                        if (doParallel)
                            zLine[iz] += backproject_Joseph_modular_parallel_kernel(aProj, params, sourcePosition, moduleCenter, u_vec, v_vec, x, y, z);
                        else
                            zLine[iz] += backproject_Joseph_modular_kernel(aProj, params, sourcePosition, moduleCenter, u_vec, v_vec, x, y, z);
                    }
                }
            }
        }
    }
    return true;
}

float backproject_Joseph_modular_parallel_kernel(float* aProj, parameters* params, float* sourcePosition, float* moduleCenter, float* u_vec, float* v_vec, float x, float y, float z)
{
    float T_u = params->pixelWidth;
    float T_v = params->pixelHeight;
    float u_0 = params->u_0();
    float v_0 = params->v_0();
    float T_x = params->voxelWidth;
    float T_x_inv = 1.0 / T_x;

    const float3 detNormal = make_float3(u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
        u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
        u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]);

    float3 r = make_float3(moduleCenter[0] - sourcePosition[0], moduleCenter[1] - sourcePosition[1], moduleCenter[2] - sourcePosition[2]);
    const float D = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

    const float odd = ((moduleCenter[0] - x) * detNormal.x + (moduleCenter[1] - y) * detNormal.y + (moduleCenter[2] - z) * detNormal.z) / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
    const float u_arg = (x + odd * r.x - moduleCenter[0]) * u_vec[0] + (y + odd * r.y - moduleCenter[1]) * u_vec[1] + (z + odd * r.z - moduleCenter[2]) * u_vec[2];
    const float v_arg = (x + odd * r.x - moduleCenter[0]) * v_vec[0] + (y + odd * r.y - moduleCenter[1]) * v_vec[1] + (z + odd * r.z - moduleCenter[2]) * v_vec[2];

    const int u_ind = int(floor(0.5f + (u_arg - u_0) / T_v));
    const int v_ind = int(floor(0.5f + (v_arg - v_0) / T_v));

    const int searchWidth_u = 1 + int(ceil(0.5 * T_x / T_u));
    const int searchWidth_v = 1 + int(ceil(0.5 * T_x / T_v));

    float val = 0.0;
    if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
    {
        const float r_x_inv = 1.0f / r.x;
        const float rayWeight = D * fabs(r_x_inv);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * T_v + v_0;
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * T_u + u_0;

                const float dx = (moduleCenter[0] - x + u * u_vec[0] + v * v_vec[0]) * r_x_inv;
                const float yy = moduleCenter[1] + u * u_vec[1] + v * v_vec[1] - dx * r.y;
                const float zz = moduleCenter[2] + u * u_vec[2] + v * v_vec[2] - dx * r.z;

                const float dy = max(0.0, 1.0 - fabs(y - yy) * T_x_inv);
                const float dz = max(0.0, 1.0 - fabs(z - zz) * T_x_inv);

                if (dy > 0.0 && dz > 0.0)
                {
                    val += rayWeight * dy * dz * aProj[iv * params->numCols + iu];
                    //val += dy * dz * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    else if (fabs(r.y) >= fabs(r.z))
    {
        const float r_y_inv = 1.0f / r.y;
        const float rayWeight = D * fabs(r_y_inv);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * T_v + v_0;
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * T_u + u_0;

                const float dy = (moduleCenter[1] - y + u * u_vec[1] + v * v_vec[1]) * r_y_inv;
                const float xx = moduleCenter[0] + u * u_vec[0] + v * v_vec[0] - dy * r.x;
                const float zz = moduleCenter[2] + u * u_vec[2] + v * v_vec[2] - dy * r.z;

                const float dx = max(0.0, 1.0 - fabs(x - xx) * T_x_inv);
                const float dz = max(0.0, 1.0 - fabs(z - zz) * T_x_inv);

                if (dx > 0.0 && dz > 0.0)
                {
                    val += rayWeight * dx * dz * aProj[iv * params->numCols + iu];
                    //val += dx * dz * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    else
    {
        const float r_z_inv = 1.0f / r.z;
        const float rayWeight = D * fabs(r_z_inv);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const float v = iv * T_v + v_0;
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const float u = iu * T_u + u_0;

                const float dz = (moduleCenter[2] - z + u * u_vec[2] + v * v_vec[2]) * r_z_inv;
                const float xx = moduleCenter[0] + u * u_vec[0] + v * v_vec[0] - dz * r.x;
                const float yy = moduleCenter[1] + u * u_vec[1] + v * v_vec[1] - dz * r.y;

                const float dx = max(0.0, 1.0 - fabs(x - xx) * T_x_inv);
                const float dy = max(0.0, 1.0 - fabs(y - yy) * T_x_inv);

                if (dx > 0.0 && dy > 0.0)
                {
                    val += rayWeight * dx * dy * aProj[iv * params->numCols + iu];
                    //val += dx * dy * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    return val * params->voxelWidth;
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

    //const int searchWidth_u = 1 + int(0.5 * params->voxelWidth / (R / D * params->pixelWidth));
    //const int searchWidth_v = 1 + int(0.5 * params->voxelWidth / (R / D * params->pixelHeight));
    const int searchWidth_u = 1 + int(ceil(0.5f * params->voxelWidth / params->pixelWidth * fabs(D)));
    const int searchWidth_v = 1 + int(ceil(0.5f * params->voxelWidth / params->pixelHeight * fabs(D)));

    const double3 rd = make_double3(double(r.x) * double(T_x_inv), double(r.y) * double(T_x_inv), double(r.z) * double(T_x_inv));
    if (fabs(r.x) >= max(fabs(r.y), fabs(r.z)))
    {
        const float rayWeight = R / fabs(r.x);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows-1, v_ind + searchWidth_v); iv++)
        {
            const double v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const double u = iu * params->pixelWidth + params->u_0();

                const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                const double trueRay_x_inv = 1.0f / trueRay_x;

                const float dy = max(0.0, 1.0 - fabs(rd.y - rd.x * trueRay_y * trueRay_x_inv));
                const float dz = max(0.0, 1.0 - fabs(rd.z - rd.x * trueRay_z * trueRay_x_inv));

                if (dy > 0.0 && dz > 0.0)
                {
                    //val += sqrtf(trueRay_x * trueRay_x + trueRay_y * trueRay_y + trueRay_z * trueRay_z) * fabs(trueRay_x_inv) * dy * dz * aProj[iv * params->numCols + iu];
                    val += rayWeight * dy * dz * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    else if (fabs(r.y) >= fabs(r.z))
    {
        const float rayWeight = R / fabs(r.y);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const double v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const double u = iu * params->pixelWidth + params->u_0();

                const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                const double trueRay_y_inv = 1.0f / trueRay_y;

                const float dx = max(0.0, 1.0 - fabs(rd.x - rd.y * trueRay_x * trueRay_y_inv));
                const float dz = max(0.0, 1.0 - fabs(rd.z - rd.y * trueRay_z * trueRay_y_inv));

                if (dx > 0.0 && dz > 0.0)
                {
                    //val += sqrtf(trueRay_y * trueRay_y + trueRay_x * trueRay_x + trueRay_z * trueRay_z) * fabs(trueRay_y_inv) * dx * dz * aProj[iv * params->numCols + iu];
                    val += rayWeight * dx * dz * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    else
    {
        const float rayWeight = R / fabs(r.z);
        for (int iv = max(0, v_ind - searchWidth_v); iv <= min(params->numRows - 1, v_ind + searchWidth_v); iv++)
        {
            const double v = iv * params->pixelHeight + params->v_0();
            for (int iu = max(0, u_ind - searchWidth_u); iu <= min(params->numCols - 1, u_ind + searchWidth_u); iu++)
            {
                const double u = iu * params->pixelWidth + params->u_0();

                const double trueRay_x = u * u_vec[0] + v * v_vec[0] - p_minus_c.x;
                const double trueRay_y = u * u_vec[1] + v * v_vec[1] - p_minus_c.y;
                const double trueRay_z = u * u_vec[2] + v * v_vec[2] - p_minus_c.z;
                const double trueRay_z_inv = 1.0f / trueRay_z;

                const float dy = max(0.0, 1.0 - fabs(rd.y - rd.z * trueRay_y * trueRay_z_inv));
                const float dx = max(0.0, 1.0 - fabs(rd.x - rd.z * trueRay_x * trueRay_z_inv));

                if (dx > 0.0 && dy > 0.0)
                {
                    //val += sqrtf(trueRay_z * trueRay_z + trueRay_x * trueRay_x + trueRay_y * trueRay_y) * fabs(trueRay_z_inv) * dx * dy * aProj[iv * params->numCols + iu];
                    val += rayWeight * dx * dy * aProj[iv * params->numCols + iu];
                }
            }
        }
    }
    return val * params->voxelWidth;
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
            {
                //printf("iy = %d: indices = %f, %f, %f; update: %f\n", iy, ix_offset + ir.x * float(iy), float(iy), iz_offset + ir.z * float(iy), tex3D_rev(mu, ix_offset + ir.x * float(iy), float(iy), iz_offset + ir.z * float(iy), params));
                val += tex3D_rev(mu, ix_offset + ir.x * float(iy), float(iy), iz_offset + ir.z * float(iy), params);
            }
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
