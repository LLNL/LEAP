////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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
    else if (params->geometry == parameters::MODULAR)
        return sensitivity_modular_CPU(s, params);
    else
        return false;
}

bool sensitivity_modular_CPU(float*& s, parameters* params)
{
    if (s == NULL || params == NULL)
        return false;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = false;
    float u_min = params->u(0);
    float u_max = params->u(params->numCols - 1);

    float v_min = params->v(0);
    float v_max = params->v(params->numRows - 1);
    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

    //float magFactor = params->sod / params->sdd;
    //float scalar = (params->voxelWidth * params->voxelWidth / (magFactor * params->pixelWidth)) * (params->voxelHeight / (magFactor * params->pixelHeight)) * params->sod * params->sod;

    const float scalar = (params->voxelWidth * params->voxelWidth / params->pixelWidth) * (params->voxelHeight / params->pixelHeight);

    if (params->volumeDimensionOrder == parameters::ZYX)
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for schedule(dynamic)
        for (int iz = 0; iz < params->numZ; iz++)
        {
            float z = iz * params->voxelHeight + params->z_0();
            float* zSlice = &s[uint64(iz) * uint64(params->numY * params->numX)];
            for (int iy = 0; iy < params->numY; iy++)
            {
                float y = iy * params->voxelWidth + params->y_0();
                float* xLine = &zSlice[iy * params->numX];
                for (int ix = 0; ix < params->numX; ix++)
                {
                    float x = ix * params->voxelWidth + params->x_0();
                    double curVal = 0.0;
                    for (int iphi = 0; iphi < params->numAngles; iphi++)
                    {
                        float* sourcePosition = &(params->sourcePositions[3 * iphi]);
                        float* moduleCenter = &(params->moduleCenters[3 * iphi]);
                        float* v_vec = &(params->rowVectors[3 * iphi]);
                        float* u_vec = &(params->colVectors[3 * iphi]);
                        float detNormal[3];
                        detNormal[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
                        detNormal[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
                        detNormal[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

                        float c_minus_s[3];
                        c_minus_s[0] = moduleCenter[0] - sourcePosition[0];
                        c_minus_s[1] = moduleCenter[1] - sourcePosition[1];
                        c_minus_s[2] = moduleCenter[2] - sourcePosition[2];

                        const float D_sq = c_minus_s[0] * c_minus_s[0] + c_minus_s[1] * c_minus_s[1] + c_minus_s[2] * c_minus_s[2];
                        const float one_over_D = 1.0/sqrt(D_sq);

                        const float c_minus_s_dot_u = c_minus_s[0] * u_vec[0] + c_minus_s[1] * u_vec[1] + c_minus_s[2] * u_vec[2];
                        const float c_minus_s_dot_v = c_minus_s[0] * v_vec[0] + c_minus_s[1] * v_vec[1] + c_minus_s[2] * v_vec[2];
                        const float c_minus_s_dot_n = c_minus_s[0] * detNormal[0] + c_minus_s[1] * detNormal[1] + c_minus_s[2] * detNormal[2];

                        const float denom = (x - sourcePosition[0]) * detNormal[0] + (y - sourcePosition[1]) * detNormal[1] + (z - sourcePosition[2]) * detNormal[2];
                        const float t_C = c_minus_s_dot_n / denom;

                        const float u_arg = t_C * ((x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;
                        const float v_arg = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;

                        if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                            curVal += sqrtf(D_sq + u_arg * u_arg + v_arg * v_arg) * one_over_D * t_C * t_C;
                    }
                    if (curVal == 0.0)
                        curVal = 1.0;
                    else
                        curVal *= scalar;
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
            float x = ix * params->voxelWidth + params->x_0();
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
                        float* sourcePosition = &(params->sourcePositions[3 * iphi]);
                        float* moduleCenter = &(params->moduleCenters[3 * iphi]);
                        float* v_vec = &(params->rowVectors[3 * iphi]);
                        float* u_vec = &(params->colVectors[3 * iphi]);
                        float detNormal[3];
                        detNormal[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
                        detNormal[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
                        detNormal[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

                        float c_minus_s[3];
                        c_minus_s[0] = moduleCenter[0] - sourcePosition[0];
                        c_minus_s[1] = moduleCenter[1] - sourcePosition[1];
                        c_minus_s[2] = moduleCenter[2] - sourcePosition[2];

                        const float D_sq = c_minus_s[0] * c_minus_s[0] + c_minus_s[1] * c_minus_s[1] + c_minus_s[2] * c_minus_s[2];
                        const float one_over_D = 1.0 / sqrt(D_sq);

                        const float c_minus_s_dot_u = c_minus_s[0] * u_vec[0] + c_minus_s[1] * u_vec[1] + c_minus_s[2] * u_vec[2];
                        const float c_minus_s_dot_v = c_minus_s[0] * v_vec[0] + c_minus_s[1] * v_vec[1] + c_minus_s[2] * v_vec[2];
                        const float c_minus_s_dot_n = c_minus_s[0] * detNormal[0] + c_minus_s[1] * detNormal[1] + c_minus_s[2] * detNormal[2];

                        const float denom = (x - sourcePosition[0]) * detNormal[0] + (y - sourcePosition[1]) * detNormal[1] + (z - sourcePosition[2]) * detNormal[2];
                        const float t_C = c_minus_s_dot_n / denom;

                        const float u_arg = t_C * ((x - sourcePosition[0]) * u_vec[0] + (y - sourcePosition[1]) * u_vec[1] + (z - sourcePosition[2]) * u_vec[2]) - c_minus_s_dot_u;
                        const float v_arg = t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v;

                        if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                            curVal += sqrtf(D_sq + u_arg * u_arg + v_arg * v_arg) * one_over_D * t_C * t_C;
                    }
                    if (curVal == 0.0)
                        curVal = 1.0;
                    else
                        curVal *= scalar;
                    zLine[iz] = float(curVal);
                }
            }
        }
    }
    return true;
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
                            if (params->helicalPitch == 0.0)
                            {
                                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                                    curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                            else
                            {
                                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                    curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                        }
                        else
                        {
                            float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (params->helicalPitch == 0.0)
                            {
                                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                                    curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                            else
                            {
                                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                    curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
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
                            if (params->helicalPitch == 0.0)
                            {
                                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_arg <= v_max*/)
                                    curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                            else
                            {
                                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_arg <= v_max)
                                    curVal += scalar * sqrt(1.0 + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                        }
                        else
                        {
                            float v_denom_inv = 1.0 / (params->sod - x * cos_phi - y * sin_phi);
                            float u_arg = (-sin_phi * x + cos_phi * y + params->tau) * v_denom_inv;
                            float v_arg = (z - z_source) * v_denom_inv;
                            if (params->helicalPitch == 0.0)
                            {
                                if (u_min <= u_arg && u_arg <= u_max /*&& v_min <= v_arg && v_max <= v_arg*/)
                                    curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
                            else
                            {
                                if (u_min <= u_arg && u_arg <= u_max && v_min <= v_arg && v_max <= v_arg)
                                    curVal += scalar * sqrt(1.0 + u_arg * u_arg + v_arg * v_arg) * v_denom_inv * v_denom_inv;
                            }
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
