////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for cpu-based sensitivity calculation (P*1)
////////////////////////////////////////////////////////////////////////////////

#include "sensitivity_cpu.h"
#include <omp.h>
#include <stdlib.h>
#include <math.h>

bool sensitivity_CPU(float*& s, parameters* params)
{
    if (s == NULL || params == NULL)
        return false;
    else if (params->geometry == parameters::CONE)
        return sensitivity_cone_CPU(s, params);
    else if (params->geometry == parameters::FAN)
        return sensitivity_fan_CPU(s, params);
    else if (params->geometry == parameters::PARALLEL)
        return sensitivity_parallel_CPU(s, params);
    else
        return false;
}

bool sensitivity_cone_CPU(float*& s, parameters* params)
{
    if (s == NULL || params == NULL)
        return false;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = true;
    float u_min = params->u(0);
    float u_max = params->u(params->numCols-1);

    float v_min = params->v(0);
    float v_max = params->v(params->numRows - 1);
    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

    float magFactor = params->sod / params->sdd;
    float scalar = (params->voxelWidth * params->voxelWidth / (magFactor*params->pixelWidth)) * (params->voxelHeight / (magFactor*params->pixelHeight)) * params->sod * params->sod;

    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float z = iz * params->voxelHeight + params->z_0();
            float* zSlice = &s[uint64(iz)* uint64(params->numY*params->numX)];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float y = iy * params->voxelWidth + params->y_0();
                float* xLine = &zSlice[iy*params->numX];
                for (int ix = 0; ix < params->numX; ix++)
                {
                    float x = ix * params->voxelWidth + params->x_0();
                    double curVal = 0.0;
                    for (int iphi = 0; iphi < params->numAngles; iphi++)
                    {
                        float cos_phi = cos(params->phis[iphi]);
                        float sin_phi = sin(params->phis[iphi]);
                        float z_source = params->z_source(iphi);

                        if (params->detectorType == parameters::CURVED)
                        {
                            float u_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = atan(-sin_phi * x + cos_phi * y + params->tau) * u_denom_inv;

                            float dist_from_source_components_x = fabs(params->sod * cos_phi + params->tau * sin_phi - x);
                            float dist_from_source_components_y = fabs(params->sod * sin_phi - params->tau * cos_phi - y);
                            float v_denom_inv = 1.0/sqrt(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                        }
                        else
                        {
                            float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                        }
                    }
                    if (curVal == 0.0)
                        curVal = 1.0;
                    xLine[ix] = float(curVal);
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
            float* xSlice = &s[uint64(ix) * uint64(params->numY * params->numZ)];
            float x = ix*params->voxelWidth + params->x_0();
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* zLine = &xSlice[iy * params->numZ];
                float y = iy * params->voxelWidth + params->y_0();
                for (int iz = 0; iz < params->numZ; iz++)
                {
                    float z = iz * params->voxelHeight + params->z_0();
                    double curVal = 0.0;
                    for (int iphi = 0; iphi < params->numAngles; iphi++)
                    {
                        float cos_phi = cos(params->phis[iphi]);
                        float sin_phi = sin(params->phis[iphi]);
                        float z_source = params->z_source(iphi);

                        if (params->detectorType == parameters::CURVED)
                        {
                            float u_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = atan(-sin_phi * x + cos_phi * y + params->tau) * u_denom_inv;

                            float dist_from_source_components_x = fabs(params->sod * cos_phi + params->tau * sin_phi - x);
                            float dist_from_source_components_y = fabs(params->sod * sin_phi - params->tau * cos_phi - y);
                            float v_denom_inv = 1.0 / sqrt(dist_from_source_components_x * dist_from_source_components_x + dist_from_source_components_y * dist_from_source_components_y);
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                        }
                        else
                        {
                            float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_max <= v_arg)
                                curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                        }
                    }
                    if (curVal == 0.0)
                        curVal = 1.0;
                    zLine[iz] = float(curVal);
                }
            }
        }
    }
    return true;
}

bool sensitivity_fan_CPU(float*& s, parameters* params)
{
    if (s == NULL || params == NULL)
        return false;

    float u_min = params->u_0() / params->sdd;
    float u_max = ((params->numCols - 1) * params->pixelWidth + params->u_0()) / params->sdd;

    float magFactor = params->sod / params->sdd;
    float scalar = (params->voxelWidth * params->voxelWidth / (magFactor * params->pixelWidth)) * params->sod;

    float* sensitivitySlice = (float*)malloc(sizeof(float) * params->numX * params->numY);
    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iy = 0; iy < params->numY; iy++)
        {
            float y = iy * params->voxelWidth + params->y_0();
            float* xLine = &sensitivitySlice[iy * params->numX];
            for (int ix = 0; ix < params->numX; ix++)
            {
                float x = ix * params->voxelWidth + params->x_0();
                double curVal = 0.0;
                for (int iphi = 0; iphi < params->numAngles; iphi++)
                {
                    float cos_phi = cos(params->phis[iphi]);
                    float sin_phi = sin(params->phis[iphi]);

                    float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                    float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                    if (u_min <= u_arg && u_arg <= u_max)
                        curVal += scalar * sqrt(1.0 + u_arg * u_arg) * v_denom_inv;
                }
                if (curVal == 0.0)
                    curVal = 1.0;
                xLine[ix] = float(curVal);
            }
        }

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float* zSlice = &s[uint64(iz) * uint64(params->numY * params->numX)];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* xLine = &zSlice[iy * params->numX];
                float* xLine_in = &sensitivitySlice[iy * params->numX];
                for (int ix = 0; ix < params->numX; ix++)
                    xLine[ix] = xLine_in[ix];
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
            float* xLine = &sensitivitySlice[ix * params->numY];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float y = iy * params->voxelWidth + params->y_0();
                double curVal = 0.0;
                for (int iphi = 0; iphi < params->numAngles; iphi++)
                {
                    float cos_phi = cos(params->phis[iphi]);
                    float sin_phi = sin(params->phis[iphi]);

                    float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                    float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                    if (u_min <= u_arg && u_arg <= u_max)
                        curVal += scalar * sqrt(1.0 + u_arg * u_arg) * v_denom_inv;
                }
                if (curVal == 0.0)
                    curVal = 1.0;
                xLine[iy] = float(curVal);
            }
        }

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int ix = 0; ix < params->numX; ix++)
        {
            float* xSlice = &s[uint64(ix) * uint64(params->numY * params->numZ)];
            float* xLine_in = &sensitivitySlice[ix * params->numY];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* zLine = &xSlice[iy * params->numZ];
                float curVal = xLine_in[iy];
                for (int iz = 0; iz < params->numZ; iz++)
                    zLine[iz] = curVal;
            }
        }
    }
    free(sensitivitySlice);
    return true;
}

bool sensitivity_parallel_CPU(float*& s, parameters* params)
{
    if (s == NULL || params == NULL)
        return false;

    float u_min = params->u_0();
    float u_max = (params->numCols - 1) * params->pixelWidth + params->u_0();

    float scalar = (params->voxelWidth * params->voxelWidth / params->pixelWidth);

    float* sensitivitySlice = (float*)malloc(sizeof(float) * params->numX * params->numY);
    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iy = 0; iy < params->numY; iy++)
        {
            float y = iy * params->voxelWidth + params->y_0();
            float* xLine = &sensitivitySlice[iy * params->numX];
            for (int ix = 0; ix < params->numX; ix++)
            {
                float x = ix * params->voxelWidth + params->x_0();
                double curVal = 0.0;
                for (int iphi = 0; iphi < params->numAngles; iphi++)
                {
                    float cos_phi = cos(params->phis[iphi]);
                    float sin_phi = sin(params->phis[iphi]);

                    float u_arg = -sin_phi * x + cos_phi * y;
                    if (u_min <= u_arg && u_arg <= u_max)
                        curVal += scalar;
                }
                if (curVal == 0.0)
                    curVal = 1.0;
                xLine[ix] = float(curVal);
            }
        }

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float* zSlice = &s[uint64(iz) * uint64(params->numY * params->numX)];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* xLine = &zSlice[iy * params->numX];
                float* xLine_in = &sensitivitySlice[iy * params->numX];
                for (int ix = 0; ix < params->numX; ix++)
                    xLine[ix] = xLine_in[ix];
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
            float* xLine = &sensitivitySlice[ix * params->numY];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float y = iy * params->voxelWidth + params->y_0();
                double curVal = 0.0;
                for (int iphi = 0; iphi < params->numAngles; iphi++)
                {
                    float cos_phi = cos(params->phis[iphi]);
                    float sin_phi = sin(params->phis[iphi]);

                    float u_arg = -sin_phi * x + cos_phi * y;
                    if (u_min <= u_arg && u_arg <= u_max)
                        curVal += scalar;
                }
                if (curVal == 0.0)
                    curVal = 1.0;
                xLine[iy] = float(curVal);
            }
        }

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int ix = 0; ix < params->numX; ix++)
        {
            float* xSlice = &s[uint64(ix) * uint64(params->numY * params->numZ)];
            float* xLine_in = &sensitivitySlice[ix * params->numY];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* zLine = &xSlice[iy * params->numZ];
                float curVal = xLine_in[iy];
                for (int iz = 0; iz < params->numZ; iz++)
                    zLine[iz] = curVal;
            }
        }
    }
    free(sensitivitySlice);
    return true;
}
