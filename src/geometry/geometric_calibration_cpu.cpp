////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based geometric calibration routines
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include "cpu_utils.h"
#include "geometric_calibration_cpu.h"

using namespace std;

bool backprojection_sweep_cpu(float* g, parameters* params, float* shifts, int numShifts, float* tilts, int numTilts, int which_param, float* costValues)
{
    if (g == NULL || params == NULL || shifts == NULL || tilts == NULL || numShifts <= 0 || numTilts <= 0 || costValues == NULL || params->geometry != parameters::CONE || params->detectorType != parameters::FLAT)
        return false;
    if (params->numZ != 1)
        return false;
    if (params->helicalPitch != 0.0)
        return false;

    float T_u = params->pixelWidth / params->sdd;
    float T_v = params->pixelHeight / params->sdd;
    float u_0 = params->col(0) / params->sdd;
    float v_0 = params->row(0) / params->sdd;

    float y_0 = params->y_0();
    float x_0 = params->x_0();
    float z = params->z_0();
    float rFOVsq = params->rFOV() * params->rFOV();

    float cos_pitch = cos(params->pitchAngle*PI/180.0);
    float sin_pitch = sin(params->pitchAngle*PI/180.0);

    float v_min_ind = -10;
    float v_max_ind = params->numRows-1+10;
    float v_min = v_min_ind*T_v + v_0;
    float v_max = v_max_ind*T_v + v_0;

    int N_v_minus_one = params->numRows - 1;
    int N_u_minus_one = params->numCols - 1;

    float backproject_scalar = (params->voxelWidth * params->voxelWidth * params->voxelHeight) / (T_u * T_v);

    memset(costValues, 0, sizeof(float) * numTilts * numShifts);

    int num_threads = num_cpu_threads();

    for (int itilt = 0; itilt < numTilts; itilt++)
    {
        float tiltAngle = (tilts[itilt] + params->tiltAngle) * PI/180.0;
        float cos_tilt = cos(tiltAngle);
        float sin_tilt = sin(tiltAngle);

        float u_vec_0[3] = {sin_pitch*sin_tilt, cos_tilt, cos_pitch*sin_tilt};
        float v_vec_0[3] = {sin_pitch*cos_tilt, -sin_tilt, cos_pitch*cos_tilt};

        float* accum_threads = new float[num_threads*numShifts];
        memset(accum_threads, 0, sizeof(float)*num_threads*numShifts);

        omp_set_num_threads(num_threads);
        #pragma omp parallel for
        for (int iy = 0; iy < params->numY; iy++)
        {
            float y = y_0 + iy*params->voxelWidth;

            float* accum_thread = &accum_threads[omp_get_thread_num()*numShifts];

            for (int ishift = 0; ishift < numShifts; ishift++)
            {
                float tau_shift = 0.0;
                float u_shift = u_0;
                if (which_param == 0)
                {
                    u_shift = u_0 - shifts[ishift]*T_u;
                    tau_shift = 0.0;
                }
                else
                {
                    u_shift = u_0;
                    tau_shift = shifts[ishift];
                }

                double accum = 0.0;
                for (int ix = 0; ix < params->numX; ix++)
                {
                    float x = x_0 + ix*params->voxelWidth;
                    if (x * x + y * y <= rFOVsq)
                    {
                        double backprojected_value = 0.0;
                        for (int iphi = 0; iphi < params->numAngles; iphi++)
                        {
                            float cos_phi = cos(params->phis[iphi]);
                            float sin_phi = sin(params->phis[iphi]);
                            float* aProj = &g[uint64(iphi)*uint64(params->numRows)*uint64(params->numCols)];

                            float dist_from_source_0 = (params->sod - x * cos_phi - y * sin_phi)*cos_pitch;

                            float v_vec_dot_x = x * (cos_phi*v_vec_0[0] - sin_phi*v_vec_0[1]) + y * (sin_phi*v_vec_0[0] + cos_phi*v_vec_0[1]);
                            float v_val_num_shift = params->sod*v_vec_0[0] - (params->tau + tau_shift)*v_vec_0[1];

                            float dist_from_source_inv = 1.0f / (dist_from_source_0 + sin_pitch*z);

                            float v_val = (v_vec_dot_x + z * v_vec_0[2] - v_val_num_shift) * dist_from_source_inv;

                            if (v_min <= v_val && v_val <= v_max)
                            {
                                float u_vec_dot_x = x * (cos_phi*u_vec_0[0] - sin_phi*u_vec_0[1]) + y * (sin_phi*u_vec_0[0] + cos_phi*u_vec_0[1]);
                                float u_val_num_shift = params->sod*u_vec_0[0] - (params->tau + tau_shift)*u_vec_0[1];
                                float u_val = (u_vec_dot_x + z * u_vec_0[2] - u_val_num_shift) * dist_from_source_inv;

                                float v_ind = (v_val - v_0) / T_v;
                                v_ind = max(float(0.0), min(float(N_v_minus_one), v_ind));

                                float u_ind = (u_val - u_shift) / T_u;
                                u_ind = max(float(0.0), min(float(N_u_minus_one), u_ind));

                                int v_lo = max(0, min(N_v_minus_one, int(v_ind)));
                                int v_hi = max(0, min(N_v_minus_one, v_lo+1));

                                int u_lo = max(0, min(N_u_minus_one, int(u_ind)));
                                int u_hi = max(0, min(N_u_minus_one, u_lo+1));

                                float dv = v_ind - float(v_lo);
                                float du = u_ind - float(u_lo);

                                float cur_val = (1.0-dv)*((1.0-du)*aProj[v_lo*params->numCols + u_lo] + du*aProj[v_lo*params->numCols + u_hi]) + dv*(((1.0-du)*aProj[v_hi*params->numCols + u_lo] + du*aProj[v_hi*params->numCols + u_hi]));
                                backprojected_value += cur_val * dist_from_source_inv * dist_from_source_inv * sqrtf(1.0f + u_val * u_val + v_val * v_val);
                            }
                        }
                        backprojected_value *= backproject_scalar;
                        backprojected_value *= backprojected_value;
                        backprojected_value *= 1.0f / float(params->numX * params->numY);

                        accum += backprojected_value;
                    }
                }

                accum_thread[ishift] += accum;
            }
        }

        for (int ithread = 0; ithread < num_threads; ithread++)
        {
            for (int ishift = 0; ishift < numShifts; ishift++)
                costValues[numShifts*itilt+ishift] += accum_threads[ithread*numShifts + ishift];
        }
        delete [] accum_threads;
    }
    return true;
}
