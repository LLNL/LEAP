////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for the primary CPU projectors models in LEAP
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors_SF_cpu.h"
#include "cpu_utils.h"

using namespace std;

//########################################################################################################################################################################
//########################################################################################################################################################################
//### Separable Footprint (SF) Projectors
//########################################################################################################################################################################
//########################################################################################################################################################################
bool CPUproject_SF_ZYX(float* g, float* f, parameters* params)
{
    params->setToZero(g, params->projectionData_numberOfElements());
    int numZ_save = params->numZ;
    float offsetZ_save = params->offsetZ;
    float z_0_save = params->z_0();

    params->volumeDimensionOrder = parameters::XYZ;

    int chunkSize = 16;
    int numChunks = max(1,int(ceil(double(params->numZ) / double(chunkSize))));
    for (int ichunk = 0; ichunk < numChunks; ichunk++)
    {
        params->numZ = numZ_save;
        params->offsetZ = offsetZ_save;

        int sliceStart = ichunk * chunkSize;
        int sliceEnd = min(params->numZ - 1, sliceStart + chunkSize - 1);
        if (sliceStart >= params->numZ)
            break;

        float* f_XYZ = reorder_ZYX_to_XYZ(f, params, sliceStart, sliceEnd);
        params->numZ = sliceEnd - sliceStart + 1;
        //params->offsetZ = offsetZ_save + sliceStart * params->voxelHeight;
        params->offsetZ += sliceStart * params->voxelHeight + z_0_save - params->z_0();
        if (params->geometry == parameters::CONE)
            CPUproject_SF_cone(g, f_XYZ, params, false);
        else if (params->geometry == parameters::FAN)
            CPUproject_SF_fan(g, f_XYZ, params, false);
        else
            CPUproject_SF_parallel(g, f_XYZ, params, false);
        free(f_XYZ);
    }
    if (params->geometry == parameters::CONE)
        applyInversePolarWeight(g, params);
    params->numZ = numZ_save;
    params->offsetZ = offsetZ_save;
    params->volumeDimensionOrder = parameters::ZYX;
    return true;
}

bool CPUbackproject_SF_ZYX(float* g, float* f, parameters* params)
{
    int numZ_save = params->numZ;
    float offsetZ_save = params->offsetZ;
    float z_0_save = params->z_0();

    params->volumeDimensionOrder = parameters::XYZ;

    //if (params->geometry == parameters::CONE)
    //    applyInversePolarWeight(g, params);

    int chunkSize = 16;
    //chunkSize = params->numZ;
    int numChunks = max(1,int(ceil(double(params->numZ) / double(chunkSize))));
    for (int ichunk = 0; ichunk < numChunks; ichunk++)
    {
        params->numZ = numZ_save;
        params->offsetZ = offsetZ_save;

        int sliceStart = ichunk * chunkSize;
        int sliceEnd = min(params->numZ - 1, sliceStart + chunkSize - 1);
        if (sliceStart >= params->numZ)
            break;

        int numZ_new = (sliceEnd - sliceStart + 1);
        float* f_XYZ = (float*)malloc(sizeof(float) * params->numX * params->numY * numZ_new);
        params->numZ = numZ_new;
        //params->offsetZ = offsetZ_save + sliceStart * params->voxelHeight;
        params->offsetZ += sliceStart * params->voxelHeight + z_0_save - params->z_0();
        //printf("z: %f to %f\n", params->z_samples(0), params->z_samples(params->numZ-1));

        if (params->geometry == parameters::CONE)
            CPUbackproject_SF_cone(g, f_XYZ, params);
        else if (params->geometry == parameters::FAN)
            CPUbackproject_SF_fan(g, f_XYZ, params);
        else
            CPUbackproject_SF_parallel(g, f_XYZ, params);

        params->numZ = numZ_save;
        params->offsetZ = offsetZ_save;
        for (int iz = sliceStart; iz <= sliceEnd; iz++)
        {
            //float* xSlice_out = &f_XYZ[ix * params->numZ * params->numY];
            float* zSlice_out = &f[uint64(iz) * uint64(params->numX * params->numY)]; // ZYX
            for (int iy = 0; iy < params->numY; iy++)
            {
                float* xLine_out = &zSlice_out[iy * params->numX];
                for (int ix = 0; ix < params->numX; ix++)
                {
                    xLine_out[ix] = f_XYZ[uint64(ix) * uint64(numZ_new * params->numY) + uint64(iy * numZ_new + iz-sliceStart)];
                }
            }
        }
        free(f_XYZ);
    }
    //if (params->geometry == parameters::CONE)
    //    applyPolarWeight(g, params);
    params->numZ = numZ_save;
    params->offsetZ = offsetZ_save;
    params->volumeDimensionOrder = parameters::ZYX;
    return true;
}

//#####################################################################################################################
bool CPUproject_SF_fan(float* g, float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (setToZero)
        params->setToZero(g, params->projectionData_numberOfElements());
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUproject_SF_ZYX(g, f, params);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[uint64(iphi) * uint64(params->numCols * params->numRows)];

        for (int ix = 0; ix < params->numX; ix++)
        {
            float* xSlice = &f[uint64(ix) * uint64(params->numY * params->numZ)];
            CPUproject_SF_fan_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
    return true;
}

bool CPUbackproject_SF_fan(float* g, float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (setToZero)
        params->setToZero(f, params->numX * params->numY * params->numZ);
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUbackproject_SF_ZYX(g, f, params);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int ix = 0; ix < params->numX; ix++)
    {
        float* xSlice = &f[uint64(ix) * uint64(params->numY * params->numZ)];
        for (int iphi = 0; iphi < params->numAngles; iphi++)
        {
            float* aProj = &g[uint64(iphi) * uint64(params->numCols * params->numRows)];
            CPUbackproject_SF_fan_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
    return true;
}

bool CPUproject_SF_fan_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float u_0 = params->u_0() / params->sdd;
    float v_0 = params->v_0();
    float T_u = params->pixelWidth / params->sdd;
    float T_v = params->pixelHeight;

    float T_z = params->voxelHeight;
    float z_0 = params->z_0();

    //const float iv = (z - startVals_g.y) / T_g.y;

    const float cos_phi = cos(params->phis[iphi]);
    const float sin_phi = sin(params->phis[iphi]);
    const float x = params->voxelWidth * ix + params->x_0();

    float A_x, B_x, A_y, B_y;
    float T_x_over_2 = params->voxelWidth / 2.0;
    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }

    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;

    //float v_denom;//, v_phi_x;
    float pitch_mult_phi_plus_startZ = 0.0;

    //float t_neg, t_pos, t_1, t_2;
    //int t_ind_min, t_ind_max;
    //float A_ind;
    float T_z_over_2 = params->voxelHeight / 2.0;

    //float T_z_over_2T_v_v_denom;

    float rFOVsq = params->rFOV() * params->rFOV();

    int vBounds[2];
    vBounds[0] = 0;
    vBounds[1] = params->numRows - 1;
    dist_from_source_components[0] = fabs(params->sod * cos_phi + params->tau * sin_phi - x);
    for (int iy = 0; iy < params->numY; iy++)
    {
        const float y = iy * params->voxelWidth + params->y_0();
        if (x * x + y * y <= rFOVsq)
        {
            float* zLine = &xSlice[iy * params->numZ];
            x_dot_theta = x * cos_phi + y * sin_phi;
            x_dot_theta_perp = -sin_phi * x + cos_phi * y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod * sin_phi - params->tau * cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0] * dist_from_source_components[0] + dist_from_source_components[1] * dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);

            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg * cos_phi - sin_phi) > fabs(u_arg * sin_phi + cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            //v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low + 0.5); // first detector index
            ind_last = int(tauInd_high + 0.5); // last detector index

            if (tauInd_low >= double(params->numCols) - 0.5 || tauInd_high <= -0.5)
                break;

            ind_diff = ind_last - ind_first;

            /*
            T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);

            double v_phi_x_step = 2.0 * T_z_over_2T_v_v_denom;
            t_neg = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;
            //*/

            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;

                for (int i = 0; i < params->numZ; i++)
                {
                    if (zLine[i] != 0.0)
                    {
                        int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                        aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                    }
                } // z
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first) + 0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last) - 0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                                aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                                aProj[L * params->numCols + ind_last] += zLine[i] * lastWeight * theWeight;
                            }
                        } // z
                    }
                    else
                    {
                        // do first
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                                aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                            }
                        } // z
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                            aProj[L * params->numCols + ind_last] += zLine[i] * lastWeight * theWeight;
                        }
                    } // z
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first) + 0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last) - 0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                                aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                                aProj[L * params->numCols + ind_middle] += zLine[i] * middleWeight * theWeight;
                                aProj[L * params->numCols + ind_last] += zLine[i] * lastWeight * theWeight;
                            } // support check
                        } // z
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                                aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                                aProj[L * params->numCols + ind_middle] += zLine[i] * middleWeight * theWeight;
                            } // support check
                        } // z
                    }
                    else
                    {
                        // do first only
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                                aProj[L * params->numCols + ind_first] += zLine[i] * firstWeight * theWeight;
                            } // support check
                        } // z
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                            aProj[L * params->numCols + ind_middle] += zLine[i] * middleWeight * theWeight;
                            aProj[L * params->numCols + ind_last] += zLine[i] * lastWeight * theWeight;
                        } // support check
                    } // z
                }
                else
                {
                    // do last only
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            int L = int(0.5 + (i * T_z + z_0 - v_0) / T_v);
                            aProj[L * params->numCols + ind_last] += zLine[i] * lastWeight * theWeight;
                        } // support check
                    } // z
                }
            } // number of contributions (1, 2, or 3)
        }
    }
    return true;
}

bool CPUbackproject_SF_fan_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float u_0 = params->u_0() / params->sdd;
    float v_0 = params->v_0();
    float T_u = params->pixelWidth / params->sdd;
    float T_v = params->pixelHeight;

    float T_z = params->voxelHeight;
    float z_0 = params->z_0();

    float rFOVsq = params->rFOV() * params->rFOV();

    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;

    float phi = params->phis[iphi];
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    //float v_denom, v_phi_x;
    float pitch_mult_phi_plus_startZ = 0.0;

    //float t_neg, t_pos; //, t_1, t_2;
    //int t_ind_min, t_ind_max;
    float T_z_over_2 = params->voxelHeight / 2.0;

    float T_x_over_2 = params->voxelWidth / 2.0;
    float A_x, B_x;
    float A_y, B_y;

    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }

    int i;

    //double T_z_over_2T_v_v_denom;

    int v_arg_bounds[2];
    v_arg_bounds[0] = 0;
    v_arg_bounds[1] = params->numRows - 1;
    float* interpolatedLineOut = (float*)calloc(size_t(params->numRows), sizeof(float));

    const float x = ix * params->voxelWidth + params->x_0();
    dist_from_source_components[0] = fabs(params->sod * cos_phi + params->tau * sin_phi - x);
    for (int iy = 0; iy < params->numY; iy++)
    {
        const float y = iy * params->voxelWidth + params->y_0();
        float* zLine = &xSlice[iy * params->numZ];
        if (x * x + y * y <= rFOVsq)
        {
            x_dot_theta = x * cos_phi + y * sin_phi;
            x_dot_theta_perp = -sin_phi * x + cos_phi * y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod * sin_phi - params->tau * cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0] * dist_from_source_components[0] + dist_from_source_components[1] * dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);

            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg * cos_phi - sin_phi) > fabs(u_arg * sin_phi + cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            //v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;
            if (params->doWeightedBackprojection)
                theWeight *= params->sod / R_minus_x_dot_theta;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low + 0.5); // first detector index
            ind_last = int(tauInd_high + 0.5); // last detector index

            //if (ind_first > params->numCols-1 || ind_last < 0)
            //    break;
            if (tauInd_low >= double(params->numCols) - 0.5 || tauInd_high <= -0.5)
                break;

            ind_diff = ind_last - ind_first;

            //T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);

            //g->v_arg_bounds(f, pitch_mult_phi_plus_startZ, v_denom, &v_arg_bounds[0]);
            //v_arg_bounds[0] = max(v_arg_bounds[0], vBounds[0]);
            //v_arg_bounds[1] = min(v_arg_bounds[1], vBounds[1]);


            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;

                for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                    interpolatedLineOut[i] = aProj[i * params->numCols + ind_first] * firstWeight * theWeight;
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first) + 0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last) - 0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i * params->numCols + ind_first] + lastWeight * aProj[i * params->numCols + ind_last]) * theWeight;
                    }
                    else
                    {
                        // do first
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i * params->numCols + ind_first] * theWeight;
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i * params->numCols + ind_last] * theWeight;
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first) + 0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last) - 0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i * params->numCols + ind_first] + middleWeight * aProj[i * params->numCols + ind_middle] + lastWeight * aProj[i * params->numCols + ind_last]) * theWeight;
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i * params->numCols + ind_first] + middleWeight * aProj[i * params->numCols + ind_middle]) * theWeight;
                    }
                    else
                    {
                        // do first only
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i * params->numCols + ind_first] * theWeight;
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = (middleWeight * aProj[i * params->numCols + ind_middle] + lastWeight * aProj[i * params->numCols + ind_last]) * theWeight;
                }
                else
                {
                    // do last only
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i * params->numCols + ind_last] * theWeight;
                }
            } // number of contributions (1, 2, or 3)

            for (int iz = 0; iz < params->numZ; iz++)
            {
                int L = int(0.5 + (iz * T_z + z_0 - v_0) / T_v);
                zLine[iz] += interpolatedLineOut[L];
            }

            /*
            v_phi_x = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v;
            double v_phi_x_step = 2.0 * T_z_over_2T_v_v_denom;
            t_neg = v_phi_x - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;

            i = 0;
            L = int(ceil(t_neg - 0.5)); // enforce: t_neg <= L+0.5
            if (L < v_arg_bounds[0])
                L = v_arg_bounds[0];
            double L_plus_half = double(L) + 0.5;
            double previousBoundary = double(L) - 0.5;

            // Extrapolation off bottom of detector
            while (t_pos < previousBoundary && i < params->numZ)
            {
                //if ((supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //    zLine[i] += interpolatedLineOut[L]*v_phi_x_step;
                t_neg = t_pos;
                t_pos += v_phi_x_step;
                i += 1;
            }
            if (t_neg < previousBoundary)
            {
                // known: t_neg < previousBoundary <= t_pos
                //if (i < params->numZ && (supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //if (i < params->numZ)
                //    zLine[i] += interpolatedLineOut[L]*(previousBoundary - t_neg);
            }
            else
                previousBoundary = t_neg;

            while (i < params->numZ && L < params->numRows)
            {
                if (t_pos <= L_plus_half)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                    zLine[i] += interpolatedLineOut[L] * (t_pos - previousBoundary);
                    previousBoundary = t_pos;
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                    i += 1;
                }
                else // L_plus_half < t_pos
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                    zLine[i] += interpolatedLineOut[L] * (L_plus_half - previousBoundary);
                    previousBoundary = L_plus_half;
                    L_plus_half += 1.0;
                    L += 1;
                }
            }
            //*/
        } // ROI check
    } // y
    free(interpolatedLineOut);

    return true;
}
//#####################################################################################################################

bool CPUproject_SF_parallel(float* g, float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    
    if (setToZero)
	    params->setToZero(g, params->projectionData_numberOfElements());
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUproject_SF_ZYX(g, f, params);
    double u_0 = params->u_0();
    
    float rFOVsq = params->rFOV()*params->rFOV();

    int iv_shift = int(floor(0.5 + (params->z_0() - params->v_0()) / params->pixelHeight));

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[uint64(iphi)* uint64(params->numCols*params->numRows)];
        const float cos_phi = cos(params->phis[iphi]);
        const float sin_phi = sin(params->phis[iphi]);
        
        float cos_phi_over_T_s = cos_phi / params->pixelWidth;
        float sin_phi_over_T_s = sin_phi / params->pixelWidth;
        float T_y_mult_cos_phi_over_T_s = params->voxelWidth*cos_phi_over_T_s;
        float N_s_minus_one = float(params->numCols-1);

        float l_phi = 1.0 / max(fabs(cos_phi), fabs(sin_phi));

        float maxWeight = params->voxelWidth*params->voxelWidth / params->pixelWidth; // cm
        float A = 0.5*(1.0 - params->voxelWidth/(params->pixelWidth*l_phi));
        float one_minus_A = 1.0 - A;
        float T_x_mult_l_phi = params->voxelWidth * l_phi;

        float s_arg, ds, ds_conj;
        int s_low, s_high;
        
        float s_0_over_T_s = u_0 / params->pixelWidth;
        for (int ix = 0; ix < params->numX; ix++)
        {
            const float x = ix * params->voxelWidth + params->x_0();
            float* xSlice = &f[uint64(ix)* uint64(params->numY*params->numZ)];
        
            s_arg = params->y_0()*cos_phi_over_T_s - x*sin_phi_over_T_s - s_0_over_T_s;
            for (int iy = 0; iy < params->numY; iy++)
            {
                const float y = iy * params->voxelWidth + params->y_0();
                if (x*x + y*y <= rFOVsq && s_arg > -1.0 && s_arg < double(params->numCols))
                {
                    float expAttenWeight = 1.0;
                    if (params->muCoeff != 0.0)
                    {
                        float x_dot_theta_perp = -sin_phi * x + cos_phi * y; // u
                        float x_dot_theta = x * cos_phi + y * sin_phi; // t
                        if (x_dot_theta_perp * x_dot_theta_perp + x_dot_theta * x_dot_theta < params->muRadius * params->muRadius)
                            expAttenWeight = exp(-params->muCoeff * (sqrt(params->muRadius * params->muRadius - x_dot_theta_perp * x_dot_theta_perp) - x_dot_theta));
                        else
                            expAttenWeight = 0.0;
                    }

                    float* zLine = &xSlice[iy*params->numZ];
                    
                    s_low = int(s_arg);
                    if (s_arg < 0.0 || s_arg > N_s_minus_one)
                    {
                        ds_conj = maxWeight * expAttenWeight;
                        for (int k = 0; k < params->numZ; k++)
                            aProj[(k+ iv_shift)*params->numCols+s_low] += ds_conj*zLine[k];
                    }
                    else
                    {
                        s_high = s_low+1;
                        ds = s_arg - double(s_low);
                        if (A > ds)
                        {
                            ds_conj = maxWeight * expAttenWeight;
                            for (int k = 0; k < params->numZ; k++)
                                aProj[(k + iv_shift)*params->numCols+s_low] += ds_conj*zLine[k];
                        }
                        else if (ds > one_minus_A)
                        {
                            ds = maxWeight * expAttenWeight;
                            for (int k = 0; k < params->numZ; k++)
                                aProj[(k + iv_shift)*params->numCols+s_high] += ds*zLine[k];
                        }
                        else
                        {
                            ds_conj = T_x_mult_l_phi*(one_minus_A - ds);
                            ds = maxWeight - ds_conj;
                            ds_conj *= expAttenWeight;
                            ds *= expAttenWeight;

                            for (int k = 0; k < params->numZ; k++)
                            {
                                aProj[(k + iv_shift)*params->numCols+s_low] += ds_conj*zLine[k];
                                aProj[(k + iv_shift)*params->numCols+s_high] += ds*zLine[k];
                            }
                        }
                    }
                }
                s_arg += T_y_mult_cos_phi_over_T_s;
            }
        }
    }
    return true;
}

bool CPUbackproject_SF_parallel(float* g , float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (setToZero)
    	params->setToZero(f, params->volumeData_numberOfElements());
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUbackproject_SF_ZYX(g, f, params);
    float u_0 = params->u_0();
    
    float rFOVsq = params->rFOV()*params->rFOV();

    int iv_shift = int(floor(0.5 + (params->z_0() - params->v_0()) / params->pixelHeight));

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int ix = 0; ix < params->numX; ix++)
    {
        const float x = ix * params->voxelWidth + params->x_0();
        float* xSlice = &f[uint64(ix)* uint64(params->numY*params->numZ)];

        for (int iphi = 0; iphi < params->numAngles; iphi++)
        {
            float* aProj = &g[uint64(iphi)* uint64(params->numCols*params->numRows)];
            const float cos_phi = cos(params->phis[iphi]);
            const float sin_phi = sin(params->phis[iphi]);
            
            float cos_phi_over_T_s = cos_phi / params->pixelWidth;
            float sin_phi_over_T_s = sin_phi / params->pixelWidth;
            float T_y_mult_cos_phi_over_T_s = params->voxelWidth*cos_phi_over_T_s;
            float N_s_minus_one = float(params->numCols-1);

            float l_phi = 1.0 / max(fabs(cos_phi), fabs(sin_phi));

            float maxWeight = params->voxelWidth*params->voxelWidth / params->pixelWidth; // cm
            float A = 0.5*(1.0 - params->voxelWidth/(params->pixelWidth*l_phi));
            float one_minus_A = 1.0 - A;
            float T_x_mult_l_phi = params->voxelWidth * l_phi;

            float s_arg, ds, ds_conj;
            int s_low, s_high;
            
            float s_0_over_T_s = u_0 / params->pixelWidth;
            s_arg = params->y_0()*cos_phi_over_T_s - x*sin_phi_over_T_s - s_0_over_T_s;
            for (int iy = 0; iy < params->numY; iy++)
            {
                const float y = iy * params->voxelWidth + params->y_0();
                if (x*x + y*y <= rFOVsq && s_arg > -1.0 && s_arg < double(params->numCols))
                {
                    float expAttenWeight = 1.0;
                    if (params->muCoeff != 0.0)
                    {
                        float x_dot_theta_perp = -sin_phi * x + cos_phi * y; // u
                        float x_dot_theta = x * cos_phi + y * sin_phi; // t
                        if (x_dot_theta_perp * x_dot_theta_perp + x_dot_theta * x_dot_theta < params->muRadius * params->muRadius)
                            expAttenWeight = exp(-params->muCoeff * (sqrt(params->muRadius * params->muRadius - x_dot_theta_perp * x_dot_theta_perp) - x_dot_theta));
                        else
                            expAttenWeight = 0.0;
                    }

                    float* zLine = &xSlice[iy*params->numZ];
                    
                    s_low = int(s_arg);
                    if (s_arg < 0.0 || s_arg > N_s_minus_one)
                    {
                        ds_conj = maxWeight * expAttenWeight;
                        for (int k = 0; k < params->numZ; k++)
                            zLine[k] += ds_conj*aProj[(k+iv_shift)*params->numCols+s_low];
                    }
                    else
                    {
                        s_high = s_low+1;
                        ds = s_arg - double(s_low);
                        if (A > ds)
                        {
                            ds_conj = maxWeight * expAttenWeight;
                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds_conj*aProj[(k + iv_shift)*params->numCols+s_low];
                        }
                        else if (ds > one_minus_A)
                        {
                            ds = maxWeight * expAttenWeight;
                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds*aProj[(k + iv_shift)*params->numCols+s_high];
                        }
                        else
                        {
                            ds_conj = T_x_mult_l_phi*(one_minus_A - ds);
                            ds = maxWeight - ds_conj;

                            ds_conj *= expAttenWeight;
                            ds *= expAttenWeight;

                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds_conj*aProj[(k + iv_shift)*params->numCols+s_low] + ds*aProj[(k + iv_shift)*params->numCols+s_high];
                        }
                    }
                }
                s_arg += T_y_mult_cos_phi_over_T_s;
            }
        }
    }
    return true;
}

bool CPUproject_SF_cone(float* g, float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (setToZero)
    	params->setToZero(g, params->projectionData_numberOfElements());
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUproject_SF_ZYX(g, f, params);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[uint64(iphi)* uint64(params->numCols*params->numRows)];
        
        for (int ix = 0; ix < params->numX; ix++)
        {
            float* xSlice = &f[uint64(ix)* uint64(params->numY*params->numZ)];
            CPUproject_SF_cone_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
    if (setToZero)
	    applyInversePolarWeight(g, params);
    return true;
}

bool CPUbackproject_SF_cone(float* g, float* f, parameters* params, bool setToZero)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
	//applyInversePolarWeight(g, params);
    if (setToZero)
    	params->setToZero(f, params->volumeData_numberOfElements());
    if (params->volumeDimensionOrder == parameters::ZYX)
        return CPUbackproject_SF_ZYX(g, f, params);
    applyInversePolarWeight(g, params);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int ix = 0; ix < params->numX; ix++)
    {
        float* xSlice = &f[uint64(ix)* uint64(params->numY*params->numZ)];
        for (int iphi = 0; iphi < params->numAngles; iphi++)
        {
            float* aProj = &g[uint64(iphi)* uint64(params->numCols*params->numRows)];
            CPUbackproject_SF_cone_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
	applyPolarWeight(g, params); // do this so projection data is not altered by backprojection operation
    return true;
}

bool CPUproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float T_u, u_0;
    float T_v = params->pixelHeight / params->sdd;
    float v_0 = params->v_0() / params->sdd;
    if (params->detectorType == parameters::CURVED)
    {
        T_u = atan(params->pixelWidth / params->sdd);
        u_0 = params->u_0();
    }
    else
    {
        u_0 = params->u_0() / params->sdd;
        T_u = params->pixelWidth / params->sdd;
    }
    
    const float cos_phi = cos(params->phis[iphi]);
    const float sin_phi = sin(params->phis[iphi]);
    const float x = params->voxelWidth*ix + params->x_0();

    const float z_source = params->z_source(iphi);
    
	float A_x, B_x, A_y, B_y;
	float T_x_over_2 = params->voxelWidth / 2.0;
    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }
    
    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;
    
    float v_denom;//, v_phi_x;
    float pitch_mult_phi_plus_startZ = z_source;

    float t_neg, t_pos, t_1, t_2;
    int t_ind_min, t_ind_max;
    float A_ind;
    float T_z_over_2 = params->voxelHeight/2.0;

    float T_z_over_2T_v_v_denom;
    
    float rFOVsq = params->rFOV()*params->rFOV();

	int vBounds[2];
	vBounds[0] = 0;
	vBounds[1] = params->numRows - 1;
	dist_from_source_components[0] = fabs(params->sod*cos_phi + params->tau*sin_phi - x);
    for (int iy = 0; iy < params->numY; iy++)
    {
        const float y = iy * params->voxelWidth + params->y_0();
        if (x*x + y*y <= rFOVsq)
        {
            float* zLine = &xSlice[iy*params->numZ];
            x_dot_theta = x*cos_phi + y*sin_phi;
            x_dot_theta_perp = -sin_phi*x + cos_phi*y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod*sin_phi - params->tau*cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0]*dist_from_source_components[0] + dist_from_source_components[1]*dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);
            
            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg*cos_phi-sin_phi) > fabs(u_arg*sin_phi+cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            if (params->detectorType == parameters::CURVED)
            {
                tau[0] = atan(tau[0]);
                tau[1] = atan(tau[1]);
                v_denom = dist_from_source;
            }
            else
                v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low+0.5); // first detector index
            ind_last = int(tauInd_high+0.5); // last detector index
            
            if (tauInd_low >= double(params->numCols)-0.5 || tauInd_high <= -0.5)
                break;
            
            ind_diff = ind_last - ind_first;

            T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);
            
            double v_phi_x_step = 2.0*T_z_over_2T_v_v_denom;
            t_neg = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;

            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;

                for (int i = 0; i < params->numZ; i++)
                {
                    if (zLine[i] != 0.0)
                    {
                        t_ind_min = int(t_neg + 0.5);
                        t_ind_max = int(t_pos + 0.5);
                        if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                        if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                        for (int L=t_ind_min; L <= t_ind_max; L++)
                        {
                            // [t_1   t_2  ]
                            // [t_neg t_pos]
                            t_1 = double(L) - 0.5;
                            t_2 = t_1 + 1.0;
                            if (t_pos < t_2) {t_2 = t_pos;}
                            if (t_neg > t_1) {t_1 = t_neg;}
                            if (t_2 > t_1)
                            {
                                A_ind = (t_2 - t_1) * theWeight;
                                aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                            }
                        } // L
                    }
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                } // z
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                    }
                                } // L
                            }
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else
                    {
                        // do first
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                    }
                                } // L
                            }
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        }
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                        aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else
                    {
                        // do first only
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        } // support check
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
                else
                {
                    // do last only
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        } // support check
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
            } // number of contributions (1, 2, or 3)
        }
    }
    return true;
}

bool CPUbackproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float T_u, u_0;
    float T_v = params->pixelHeight / params->sdd;
    float v_0 = params->v_0() / params->sdd;
    if (params->detectorType == parameters::CURVED)
    {
        T_u = atan(params->pixelWidth / params->sdd);
        u_0 = params->u_0();
    }
    else
    {
        u_0 = params->u_0() / params->sdd;
        T_u = params->pixelWidth / params->sdd;
    }
    
    float rFOVsq = params->rFOV()*params->rFOV();
    
    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;
    
    float phi = params->phis[iphi];
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float v_denom, v_phi_x;
    float pitch_mult_phi_plus_startZ = params->z_source(iphi);

    float t_neg, t_pos; //, t_1, t_2;
    //int t_ind_min, t_ind_max;
    float T_z_over_2 = params->voxelHeight/2.0;

    float T_x_over_2 = params->voxelWidth / 2.0;
    float A_x, B_x;
    float A_y, B_y;
    
    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }

    int i, L;

    double T_z_over_2T_v_v_denom;

    int v_arg_bounds[2];
	v_arg_bounds[0] = 0;
	v_arg_bounds[1] = params->numRows - 1;
    float* interpolatedLineOut = (float*) calloc(size_t(params->numRows), sizeof(float));

    const float x = ix*params->voxelWidth + params->x_0();
    dist_from_source_components[0] = fabs(params->sod*cos_phi + params->tau*sin_phi - x);
    for (int iy=0; iy < params->numY; iy++)
    {
        const float y = iy*params->voxelWidth + params->y_0();
        float* zLine = &xSlice[iy*params->numZ];
        if (x*x + y*y <= rFOVsq)
        {
            x_dot_theta = x*cos_phi + y*sin_phi;
            x_dot_theta_perp = -sin_phi*x + cos_phi*y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod*sin_phi - params->tau*cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0]*dist_from_source_components[0] + dist_from_source_components[1]*dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);

            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg*cos_phi-sin_phi) > fabs(u_arg*sin_phi+cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            if (params->detectorType == parameters::CURVED)
            {
                tau[0] = atan(tau[0]);
                tau[1] = atan(tau[1]);
                v_denom = dist_from_source;
            }
            else
                v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low+0.5); // first detector index
            ind_last = int(tauInd_high+0.5); // last detector index
            
            //if (ind_first > params->numCols-1 || ind_last < 0)
            //    break;
            if (tauInd_low >= double(params->numCols)-0.5 || tauInd_high <= -0.5)
                break;
            
            ind_diff = ind_last - ind_first;

            T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);

            //g->v_arg_bounds(f, pitch_mult_phi_plus_startZ, v_denom, &v_arg_bounds[0]);
            //v_arg_bounds[0] = max(v_arg_bounds[0], vBounds[0]);
            //v_arg_bounds[1] = min(v_arg_bounds[1], vBounds[1]);

            
            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;
                
                for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                    interpolatedLineOut[i] = aProj[i*params->numCols + ind_first] * firstWeight * theWeight;
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i*params->numCols+ind_first] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                    }
                    else
                    {
                        // do first
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i*params->numCols+ind_first] * theWeight;
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i*params->numCols+ind_last] * theWeight;
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight*aProj[i*params->numCols+ind_first] + middleWeight*aProj[i*params->numCols+ind_middle] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i*params->numCols+ind_first] + middleWeight*aProj[i*params->numCols+ind_middle]) * theWeight;
                    }
                    else
                    {
                        // do first only
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i*params->numCols+ind_first] * theWeight;
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = (middleWeight * aProj[i*params->numCols+ind_middle] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                }
                else
                {
                    // do last only
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i*params->numCols+ind_last] * theWeight;
                }
            } // number of contributions (1, 2, or 3)

            v_phi_x = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v;
            double v_phi_x_step = 2.0*T_z_over_2T_v_v_denom;
            t_neg = v_phi_x - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;

            i = 0;
            L = int(ceil(t_neg - 0.5)); // enforce: t_neg <= L+0.5
            if (L < v_arg_bounds[0])
                L = v_arg_bounds[0];
            double L_plus_half = double(L) + 0.5;
            double previousBoundary = double(L) - 0.5;

            // Extrapolation off bottom of detector
            while (t_pos < previousBoundary && i < params->numZ)
            {
                //if ((supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //    zLine[i] += interpolatedLineOut[L]*v_phi_x_step;
                t_neg = t_pos;
                t_pos += v_phi_x_step;
                i += 1;
            }
            if (t_neg < previousBoundary)
            {
                // known: t_neg < previousBoundary <= t_pos
                //if (i < params->numZ && (supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //if (i < params->numZ)
                //    zLine[i] += interpolatedLineOut[L]*(previousBoundary - t_neg);
            }
            else
                previousBoundary = t_neg;

            while (i < params->numZ && L < params->numRows)
            {
                if (t_pos <= L_plus_half)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(t_pos - previousBoundary);
                    previousBoundary = t_pos;
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                    i += 1;
                }
                else // L_plus_half < t_pos
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(L_plus_half - previousBoundary);
                    previousBoundary = L_plus_half;
                    L_plus_half += 1.0;
                    L += 1;
                }
            }

            // now either: i == params->numZ || L == g->N_v
            // Extrapolation off top of detector
			/*
            if (i < params->numZ && doExtrapolation == true)
            {
                L = params->numRows-1;
                L_plus_half = double(L) + 0.5;
                if (t_neg < L_plus_half)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(t_pos - L_plus_half);
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                    i += 1;
                }
                // now: L_plus_half < t_neg < t_pos
                while (i < params->numZ)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*v_phi_x_step;
                    i += 1;
                }
            }
			//*/
        } // ROI check
    } // y
    free(interpolatedLineOut);

    return true;
}

bool applyPolarWeight(float* g, parameters* params)
{
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < params->numAngles; i++)
	{
		float* aProj = &g[uint64(i)* uint64(params->numRows*params->numCols)];
		for (int j = 0; j < params->numRows; j++)
		{
            float v = params->v(j) / params->sdd;
			float temp = 1.0 / sqrt(1.0 + v*v);
			float* zLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
				zLine[k] *= temp;
		}
	}
	return true;
}

bool applyInversePolarWeight(float* g, parameters* params)
{
	omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
	for (int i = 0; i < params->numAngles; i++)
	{
		float* aProj = &g[uint64(i)* uint64(params->numRows*params->numCols)];
		for (int j = 0; j < params->numRows; j++)
		{
            float v = params->v(j) / params->sdd;
			float temp = sqrt(1.0 + v*v);
			float* zLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
				zLine[k] *= temp;
		}
	}
	return true;
}
