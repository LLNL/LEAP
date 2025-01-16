////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include "rebin.h"
#include "leap_defines.h"
#include "log.h"
#include "ramp_filter_cpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <cstring>
#include <omp.h>

using namespace std;

rebin::rebin()
{
    params = NULL;
    fanAngles = NULL;
}

rebin::~rebin()
{
    params = NULL;
    fanAngles = NULL;
}

float* rebin::rebin_parallel_singleProjection(float* g, parameters* params_in, int order, float desiredAngle)
{
    params = params_in;
    if (g == NULL || params == NULL || params->numCols <= 2)
        return NULL;
    if (params->geometry != parameters::CONE && params->geometry != parameters::FAN)
    {
        printf("Error: rebin_parallel: input data must be specified as fan-beam or cone-beam\n");
        return NULL;
    }
    if (params->anglesAreEquispaced() == false && order > 2)
    {
        order = 2;
        LOG(logWARNING, "rebin", "rebin_parallel") << "Warning: non-equi-spaced angles can only be rebinned with bilinear interpolation" << std::endl;
        //LOG(logERROR, "rebin", "rebin_parallel") << "Error: current implementation requires angles to be equi-spaced.  If this feature is of interest please submit a feature request." << std::endl;
        //return NULL;
    }
    float* outProj = new float[params->numRows * params->numCols];

    order = order + (order % 2);
    order = max(2, min(order, 6));

    T_phi = params->T_phi();
    phi_0 = params->phis[0];
    N_phi = params->numAngles;

    float angularRange_mod_360 = fabs(params->angularRange) - floor(0.5 + fabs(params->angularRange) / 360.0) * 360.0;
    //printf("angularRange_mod_360 = %f\n", angularRange_mod_360);
    //printf("fabs(params->T_phi())*180.0/PI = %f\n", fabs(params->T_phi()) * 180.0 / PI);

    bool doWrapAround = true;
    if (params->helicalPitch != 0.0)
        doWrapAround = false;
    else if (fabs(params->angularRange) < min(359.0, 360.0 - fabs(T_phi) * 180.0 / PI)) //if (angularRange_mod_360 > fabs(params->T_phi())*180.0/PI)
        doWrapAround = false;

    if (doWrapAround)
    {
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = TRUE" << std::endl;
    }
    else
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = FALSE" << std::endl;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = true;

    double tau = params->tau;
    double R = params->sod;
    double R_tau = sqrt(R * R + tau * tau);
    double atan_A = atan(2.0 * tau * R / (R * R - tau * tau));
    double asin_tau_R_tau = asin(tau / R_tau);

    float rotationDirection = 1.0;
    if (T_phi < 0.0)
        rotationDirection = -1.0;

    N_phi_new = N_phi;
    phi_0_new = phi_0;

    //double T_u = params->pixelWidth * params->sod / params->sdd;
    double T_u = params->u(1) - params->u(0);
    double u_0 = params->u(0);
    double u_end = params->u(params->numCols - 1);

    double u_mid = 0.5 * (u_end + u_0);
    double u_mid_ind = (u_mid - u_0) / T_u;// T_u* i + u_0

    double T_s = params->pixelWidth * R_tau / params->sdd;
    //double s_0 = -u_mid_ind * T_s;//u_mid_ind*T_s + s_0 = 0
    double s_0 = -params->centerCol * T_s;

    if (doWrapAround == false)
    {
        double fanAngle_low, fanAngle_high;
        // Need to change N_phi and phi_0
        if (params->detectorType == parameters::FLAT)
        {
            fanAngle_low = atan(params->u(0));
            fanAngle_high = atan(params->u(params->numCols - 1));
        }
        else
        {
            fanAngle_low = params->u(0);
            fanAngle_high = params->u(params->numCols - 1);
        }
        //float phi_start = params->phis[0] - fanAngle_low * rotationDirection;
        //float phi_end = params->phis[params->numAngles - 1] - fanAngle_high * rotationDirection;
        float phi_start, phi_end;
        if (T_phi >= 0.0)
        {
            phi_start = params->phis[0] - fanAngle_low;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_high;
        }
        else
        {
            phi_start = params->phis[0] - fanAngle_high;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_low;
        }

        //phi_start = T_phi*? + phi_0
        int phi_0_new_ind = int(ceil((phi_start - phi_0) / (T_phi)));
        int phi_end_new_ind = int(floor((phi_end - phi_0) / (T_phi)));

        phi_0_new = phi_0_new_ind * T_phi + phi_0;
        N_phi_new = phi_end_new_ind - phi_0_new_ind + 1;
    }
    //double beta_min = min(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    //double beta_max = max(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    double beta_min = min(phi_0, T_phi * (N_phi - 1) + phi_0);
    double beta_max = max(phi_0, T_phi * (N_phi - 1) + phi_0);

    int iAngle_min, iAngle_max;
    double phi = desiredAngle;
    for (int iCol = 0; iCol < params->numCols; iCol++)
    {
        double s = iCol * T_s + s_0;
        double alpha = asin(s / R_tau) + asin_tau_R_tau;
        double beta = phi + alpha;

        if (doWrapAround)
        {
            if (beta < beta_min)
                beta = beta + 2.0 * PI;
            else if (beta > beta_max)
                beta = beta - 2.0 * PI;
        }
        else
        {
            if (beta < beta_min)
            {
                double alpha_new = -alpha + atan_A;
                beta = beta - (alpha - alpha_new) + PI;
                alpha = alpha_new;
            }
            else if (beta > beta_max)
            {
                double alpha_new = -alpha + atan_A;
                beta = beta - (alpha - alpha_new) - PI;
                alpha = alpha_new;
            }
        }

        double lateral = alpha;
        if (params->detectorType == parameters::FLAT)
            lateral = tan(alpha);

        float beta_ind = params->phi_inv(beta);
        if (iCol == 0)
        {
            iAngle_min = int(floor(beta_ind));
            iAngle_max = int(ceil(beta_ind));
        }
        else
        {
            iAngle_min = min(iAngle_min, int(floor(beta_ind)));
            iAngle_max = max(iAngle_max, int(ceil(beta_ind)));
        }
    }
    iAngle_min = max(0, min(params->numAngles - 1, iAngle_min - order));
    iAngle_max = max(0, min(params->numAngles - 1, iAngle_max + order));

    // Perform fan to parallel rebinning
    LOG(logDEBUG, "rebin", "rebin_parallel") << "performing fan to parallel rebinning..." << std::endl;
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int iRow = 0; iRow < params->numRows; iRow++)
    {
        // Make a copy of the sinogram
        float* sino = new float[params->numCols * params->numAngles];
        for (int iAngle = iAngle_min; iAngle <= iAngle_max; iAngle++)
        {
            uint64 ind_offset = uint64(iAngle) * uint64(params->numRows * params->numCols) + uint64(iRow) * uint64(params->numCols);
            float* g_offset = &g[ind_offset];

            for (int iCol = 0; iCol < params->numCols; iCol++)
                sino[iAngle * params->numCols + iCol] = g_offset[iCol];
        }

        double phi = desiredAngle;
        for (int iCol = 0; iCol < params->numCols; iCol++)
        {
            double s = iCol * T_s + s_0;
            double alpha = asin(s / R_tau) + asin_tau_R_tau;
            double beta = phi + alpha;

            if (doWrapAround)
            {
                if (beta < beta_min)
                    beta = beta + 2.0 * PI;
                else if (beta > beta_max)
                    beta = beta - 2.0 * PI;
            }
            else
            {
                if (beta < beta_min)
                {
                    double alpha_new = -alpha + atan_A;
                    beta = beta - (alpha - alpha_new) + PI;
                    alpha = alpha_new;
                }
                else if (beta > beta_max)
                {
                    double alpha_new = -alpha + atan_A;
                    beta = beta - (alpha - alpha_new) - PI;
                    alpha = alpha_new;
                }
            }

            double lateral = alpha;
            if (params->detectorType == parameters::FLAT)
                lateral = tan(alpha);

            float beta_ind = params->phi_inv(beta);
            outProj[uint64(iRow)*uint64(params->numCols) + uint64(iCol)] = LagrangeInterpolation13(sino, params, beta_ind, (lateral - u_0) / T_u, iRow, doWrapAround, order);
        }

        delete[] sino;
    }

    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

    return outProj;
}

int rebin::rebin_parallel_singleSinogram(float* g, parameters* params_in, float* parallel_sinogram, int order, int desiredRow, bool reduce180)
{
    //*
    params = params_in;
    if (g == NULL || params == NULL || params->numCols <= 2 || parallel_sinogram == NULL)
        return 0;
    if (params->geometry != parameters::CONE && params->geometry != parameters::FAN)
    {
        printf("Error: rebin_parallel_singleSinogram: input data must be specified as fan-beam or cone-beam\n");
        return 0;
    }
    if (params->anglesAreEquispaced() == false && order > 2)
    {
        order = 2;
        LOG(logWARNING, "rebin", "rebin_parallel_singleSinogram") << "Warning: non-equi-spaced angles can only be rebinned with bilinear interpolation" << std::endl;
        //LOG(logERROR, "rebin", "rebin_parallel_singleSinogram") << "Error: current implementation requires angles to be equi-spaced.  If this feature is of interest please submit a feature request." << std::endl;
        //return NULL;
    }

    if (desiredRow < 0 || desiredRow >= params->numRows)
        desiredRow = max(0, min(params->numRows-1, int(floor(0.5 + params->centerRow))));

    reduce180 = true;
    bool padOnLeft;
    int N_add = 0;
    if (reduce180)
        N_add = zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);
    if (N_add <= 0)
        reduce180 = false;

    order = order + (order % 2);
    order = max(2, min(order, 6));

    T_phi = params->T_phi();
    phi_0 = params->phis[0];
    N_phi = params->numAngles;

    float angularRange_mod_360 = fabs(params->angularRange) - floor(0.5 + fabs(params->angularRange) / 360.0) * 360.0;
    //printf("angularRange_mod_360 = %f\n", angularRange_mod_360);
    //printf("fabs(params->T_phi())*180.0/PI = %f\n", fabs(params->T_phi()) * 180.0 / PI);

    bool doWrapAround = true;
    if (params->helicalPitch != 0.0)
        doWrapAround = false;
    else if (fabs(params->angularRange) < min(359.0, 360.0 - fabs(T_phi) * 180.0 / PI)) //if (angularRange_mod_360 > fabs(params->T_phi())*180.0/PI)
        doWrapAround = false;

    if (doWrapAround)
    {
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = TRUE" << std::endl;
    }
    else
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = FALSE" << std::endl;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = true;

    double tau = params->tau;
    double R = params->sod;
    double R_tau = sqrt(R * R + tau * tau);
    double atan_A = atan(2.0 * tau * R / (R * R - tau * tau));
    double asin_tau_R_tau = asin(tau / R_tau);

    float rotationDirection = 1.0;
    if (T_phi < 0.0)
        rotationDirection = -1.0;

    N_phi_new = N_phi;
    phi_0_new = phi_0;

    //double T_u = params->pixelWidth * params->sod / params->sdd;
    double T_u = params->u(1) - params->u(0);
    double u_0 = params->u(0);
    double u_end = params->u(params->numCols - 1);

    double u_mid = 0.5*(u_end + u_0);
    double u_mid_ind = (u_mid-u_0)/T_u;// T_u* i + u_0

    int N_s = params->numCols;
    double T_s = params->pixelWidth * R_tau / params->sdd;
    //double s_0 = -u_mid_ind * T_s;//u_mid_ind*T_s + s_0 = 0
    double s_0 = -params->centerCol * T_s;

    if (doWrapAround == false)
    {
        double fanAngle_low, fanAngle_high;
        // Need to change N_phi and phi_0
        if (params->detectorType == parameters::FLAT)
        {
            fanAngle_low = atan(params->u(0));
            fanAngle_high = atan(params->u(params->numCols-1));
        }
        else
        {
            fanAngle_low = params->u(0);
            fanAngle_high = params->u(params->numCols - 1);
        }
        //float phi_start = params->phis[0] - fanAngle_low * rotationDirection;
        //float phi_end = params->phis[params->numAngles - 1] - fanAngle_high * rotationDirection;
        float phi_start, phi_end;
        if (T_phi >= 0.0)
        {
            phi_start = params->phis[0] - fanAngle_low;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_high;
        }
        else
        {
            phi_start = params->phis[0] - fanAngle_high;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_low;
        }

        //phi_start = T_phi*? + phi_0
        int phi_0_new_ind = int(ceil((phi_start - phi_0) / (T_phi)));
        int phi_end_new_ind = int(floor((phi_end - phi_0) / (T_phi)));

        phi_0_new = phi_0_new_ind * T_phi + phi_0;
        N_phi_new = phi_end_new_ind - phi_0_new_ind + 1;
    }
    //double beta_min = min(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    //double beta_max = max(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    double beta_min = min(phi_0, T_phi * (N_phi-1) + phi_0);
    double beta_max = max(phi_0, T_phi * (N_phi-1) + phi_0);
    //printf("beta_max = %f, beta_min = %f\n", beta_max, beta_min);

    float* sino_input = new float[params->numCols * params->numAngles];
    if (params->geometry == parameters::CONE && params->detectorType == parameters::FLAT && params->numRows > 1)
    {
        float cos_tilt = cos(params->tiltAngle * PI / 180.0);
        float sin_tilt = sin(params->tiltAngle * PI / 180.0);

        LOG(logDEBUG, "rebin", "rebin_parallel_singleSinogram") << "performing detector column stretching..." << std::endl;
        double T_v = params->v(1) - params->v(0);
        double v_0 = params->v(0);
        double T_u = params->u(1) - params->u(0);

        //*
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int i = 0; i < params->numAngles; i++)
        {
            float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];
            for (int k = 0; k < params->numCols; k++)
            {
                double u = params->u(k);
                double u_centered = T_u * (double(k) - 0.5 * double(params->numCols - 1));
                float lineLength = sqrt(1.0 + u * u);

                int j = desiredRow;
                //for (int j = 0; j < params->numRows; j++)
                {
                    double v_new = params->v(j);
                    double v_centered = T_v * (double(j) - 0.5 * double(params->numRows - 1));

                    v_new = -sin_tilt * u_centered + cos_tilt * v_centered + v_new-v_centered;
                    //v_new = sin_tilt * u_centered + cos_tilt * v_centered + v_new-v_centered;

                    //v_new = v_new / lineLength;
                    v_new = v_new * lineLength;
                    
                    /*
                    float col_A = cos_alpha * col + sin_alpha * row - col_0 + col_0_centered;
                    float row_A = -sin_alpha * col + cos_alpha * row;
                    //*/

                    double ind = (v_new - v_0) / T_v;
                    int ind_closest = int(floor(0.5 + ind));
                    if (fabs(ind - double(ind_closest)) < 1.0e-8 || ind_closest < 0 || ind_closest > params->numRows-1)
                    {
                        ind_closest = max(0, min(params->numRows - 1, ind_closest));
                        sino_input[i * params->numCols + k] = aProj[ind_closest*params->numCols + k];
                    }
                    else
                    {
                        // Determine interpolation polynomial order and placement
                        int ind_lowest, M;
                        int M_default_local = order;
                        if (fabs(ind - double(ind_closest)) < 0.25)
                        {
                            // use odd length filter
                            M = min(M_default_local - ((M_default_local + 1) % 2), 1 + min(params->numCols - 1 - ind_closest, ind_closest));
                            if (M % 2 == 0)
                                M -= 1;
                            ind_lowest = ind_closest - (M - 1) / 2;
                        }
                        else
                        {
                            M = min(M_default_local - (M_default_local % 2), min(params->numCols - int(ceil(ind)), 1 + int(ind)));
                            if (M % 2 == 1)
                                M -= 1;
                            ind_lowest = int(ind) - M / 2 + 1;
                        }
                        if (M <= 1)
                        {
                            ind_closest = max(0, min(params->numRows - 1, ind_closest));
                            sino_input[i * params->numCols + k] = aProj[ind_closest*params->numCols + k];
                        }
                        else
                        {
                            double val = 0.0;
                            for (int l = 0; l < M; l++)
                            {
                                double h = 1.0;
                                for (int m = 0; m < M; m++)
                                {
                                    if (m != l)
                                        h *= double(ind - m - ind_lowest) / double(l - m);
                                }
                                int row_ind = max(0, min(params->numRows-1, l + ind_lowest));
                                val += h * aProj[row_ind*params->numCols + k];
                            }
                            sino_input[i * params->numCols + k] = val;
                        }
                    }
                }
            }
        }
        //*/
    }
    else
    {
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int i = 0; i < params->numAngles; i++)
        {
            float* aLine = &g[uint64(i) * uint64(params->numRows * params->numCols) + uint64(desiredRow*params->numCols)];
            for (int k = 0; k < params->numCols; k++)
                sino_input[i * params->numCols + k] = aLine[k];
        }
    }

    if (N_add > 0)
    {
        /*
        if (padOnLeft) // params->centerCol += N_add;
            s_0 = -(params->centerCol+N_add) * T_s;
        N_s += N_add;
        N_phi_new = params->numAngles/2;
        //*/
        N_s = params->numCols*2;
        s_0 = -0.5*(N_s-1)*T_s;
        N_phi_new = params->numAngles/2;
    }

    // Perform fan to parallel rebinning
    int iRow = desiredRow;

    LOG(logDEBUG, "rebin", "rebin_parallel") << "performing fan to parallel rebinning..." << std::endl;

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int iAngle = 0; iAngle < N_phi_new; iAngle++)
    {
        uint64 ind_offset = uint64(iAngle) * uint64(N_s);
        double phi = iAngle*T_phi + phi_0_new;
        for (int iCol = 0; iCol < N_s; iCol++)
        {
            double s = iCol*T_s + s_0;
            double alpha = asin(s / R_tau) + asin_tau_R_tau;
            double beta = phi + alpha;

            double lateral = alpha;
            if (params->detectorType == parameters::FLAT)
                lateral = tan(alpha);

            if (reduce180)
            {
                if ((lateral < u_0 && padOnLeft) || (lateral > u_end && padOnLeft == false))
                {
                    alpha = asin(-s / R_tau) + asin_tau_R_tau;
                    beta = phi + alpha + PI;
                }
            }

            if (doWrapAround)
            {
                if (beta < beta_min)
                    beta = beta + 2.0 * PI;
                else if (beta > beta_max)
                    beta = beta - 2.0 * PI;
                //if (beta < beta_min || beta > beta_max)
                //    printf("beta error!\n");
            }
            else
            {
                //if (iRow == params->numRows / 2 && (beta < beta_min || beta > beta_max))
                //    printf("using conjugate ray: phi = %f, beta = %f, alpha = %f\n", phi, beta, alpha);
                if (beta < beta_min)
                {
                    double alpha_new = -alpha + atan_A;
                    beta = beta - (alpha - alpha_new) + PI;
                    alpha = alpha_new;
                }
                else if (beta > beta_max)
                {
                    double alpha_new = -alpha + atan_A;
                    beta = beta - (alpha - alpha_new) - PI;
                    alpha = alpha_new;
                }
            }

            lateral = alpha;
            if (params->detectorType == parameters::FLAT)
                lateral = tan(alpha);

            float beta_ind = params->phi_inv(beta);
            parallel_sinogram[ind_offset + uint64(iCol)] = LagrangeInterpolation13(sino_input, params, beta_ind, (lateral-u_0)/T_u, iRow, doWrapAround, order);
        }
    }
    delete [] sino_input;

    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
    //*/

    /* Modify CT geometry parameters
    if (params->geometry == parameters::CONE)
        params->geometry = parameters::CONE_PARALLEL;
    else if (params->geometry == parameters::FAN)
        params->geometry = parameters::PARALLEL;
    //*/

    if (N_phi_new != N_phi || phi_0_new != phi_0)
    {
        /*
        float* phis = new float[N_phi_new];
        for (int i = 0; i < N_phi_new; i++)
            phis[i] = (T_phi * i + phi_0_new) * 180.0 / PI + 90.0;
        params->set_angles(phis, N_phi_new);
        delete[] phis;
        //*/

        uint64 offs = uint64(N_phi_new) * uint64(params->numCols);
        float* g_crop = &g[offs];
        if (reduce180 == false)
        {
            params->setToZero(g_crop, uint64(N_phi - N_phi_new) * uint64(params->numCols));
            LOG(logWARNING, "rebin", "rebin_parallel") << "WARNING: Reducing number of angles from " << N_phi << " to " << N_phi_new << std::endl;
        }
        //LOG(logWARNING, "rebin", "rebin_parallel") << "New angular coverage: " << params->angularRange << " degrees" << std::endl;
    }

    /*
    params->numCols = N_s;
    params->centerCol = -s_0/T_s; // i* T_s + s_0 == 0
    params->tau = 0.0;
    params->sod = R_tau;
    params->pixelWidth = T_s;
    params->detectorType = parameters::FLAT;
    //*/

    return N_s;
}

bool rebin::rebin_parallel(float* g, parameters* params_in, int order)
{
    //*
    params = params_in;
    if (g == NULL || params == NULL || params->numCols <= 2)
        return false;
    if (params->geometry != parameters::CONE && params->geometry != parameters::FAN)
    {
        printf("Error: rebin_parallel: input data must be specified as fan-beam or cone-beam\n");
        return false;
    }
    if (params->anglesAreEquispaced() == false && order > 2)
    {
        order = 2;
        LOG(logWARNING, "rebin", "rebin_parallel") << "Warning: non-equi-spaced angles can only be rebinned with bilinear interpolation" << std::endl;
        //LOG(logERROR, "rebin", "rebin_parallel") << "Error: current implementation requires angles to be equi-spaced.  If this feature is of interest please submit a feature request." << std::endl;
        //return NULL;
    }

    order = order + (order % 2);
    order = max(2, min(order, 6));

    T_phi = params->T_phi();
    phi_0 = params->phis[0];
    N_phi = params->numAngles;

    float angularRange_mod_360 = fabs(params->angularRange) - floor(0.5 + fabs(params->angularRange) / 360.0) * 360.0;
    //printf("angularRange_mod_360 = %f\n", angularRange_mod_360);
    //printf("fabs(params->T_phi())*180.0/PI = %f\n", fabs(params->T_phi()) * 180.0 / PI);

    bool doWrapAround = true;
    if (params->helicalPitch != 0.0)
        doWrapAround = false;
    else if (fabs(params->angularRange) < min(359.0, 360.0 - fabs(T_phi) * 180.0 / PI)) //if (angularRange_mod_360 > fabs(params->T_phi())*180.0/PI)
        doWrapAround = false;

    if (doWrapAround)
    {
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = TRUE" << std::endl;
    }
    else
        LOG(logDEBUG, "rebin", "rebin_parallel") << "doWrapAround = FALSE" << std::endl;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = true;

    double tau = params->tau;
    double R = params->sod;
    double R_tau = sqrt(R * R + tau * tau);
    double atan_A = atan(2.0 * tau * R / (R * R - tau * tau));
    double asin_tau_R_tau = asin(tau / R_tau);

    float rotationDirection = 1.0;
    if (T_phi < 0.0)
        rotationDirection = -1.0;

    N_phi_new = N_phi;
    phi_0_new = phi_0;

    //double T_u = params->pixelWidth * params->sod / params->sdd;
    double T_u = params->u(1) - params->u(0);
    double u_0 = params->u(0);
    double u_end = params->u(params->numCols - 1);

    double u_mid = 0.5*(u_end + u_0);
    double u_mid_ind = (u_mid-u_0)/T_u;// T_u* i + u_0

    double T_s = params->pixelWidth * R_tau / params->sdd;
    //double s_0 = -u_mid_ind * T_s;//u_mid_ind*T_s + s_0 = 0
    double s_0 = -params->centerCol * T_s;

    if (doWrapAround == false)
    {
        double fanAngle_low, fanAngle_high;
        // Need to change N_phi and phi_0
        if (params->detectorType == parameters::FLAT)
        {
            fanAngle_low = atan(params->u(0));
            fanAngle_high = atan(params->u(params->numCols-1));
        }
        else
        {
            fanAngle_low = params->u(0);
            fanAngle_high = params->u(params->numCols - 1);
        }
        //float phi_start = params->phis[0] - fanAngle_low * rotationDirection;
        //float phi_end = params->phis[params->numAngles - 1] - fanAngle_high * rotationDirection;
        float phi_start, phi_end;
        if (T_phi >= 0.0)
        {
            phi_start = params->phis[0] - fanAngle_low;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_high;
        }
        else
        {
            phi_start = params->phis[0] - fanAngle_high;
            phi_end = params->phis[params->numAngles - 1] - fanAngle_low;
        }

        //phi_start = T_phi*? + phi_0
        int phi_0_new_ind = int(ceil((phi_start - phi_0) / (T_phi)));
        int phi_end_new_ind = int(floor((phi_end - phi_0) / (T_phi)));

        phi_0_new = phi_0_new_ind * T_phi + phi_0;
        N_phi_new = phi_end_new_ind - phi_0_new_ind + 1;
    }
    //double beta_min = min(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    //double beta_max = max(phi_0_new, T_phi * (N_phi_new-1) + phi_0_new);
    double beta_min = min(phi_0, T_phi * (N_phi-1) + phi_0);
    double beta_max = max(phi_0, T_phi * (N_phi-1) + phi_0);
    //printf("beta_max = %f, beta_min = %f\n", beta_max, beta_min);

    if (params->geometry == parameters::CONE && params->detectorType == parameters::FLAT && params->numRows > 1)
    {
        LOG(logDEBUG, "rebin", "rebin_parallel") << "performing detector column stretching..." << std::endl;
        double T_v = params->v(1) - params->v(0);
        double v_0 = params->v(0);

        //*
        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int i = 0; i < params->numAngles; i++)
        {
            float* vertLine = new float[params->numRows];
            float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];
            for (int k = 0; k < params->numCols; k++)
            {
                for (int j = 0; j < params->numRows; j++)
                    vertLine[j] = aProj[j * params->numCols + k];
                float lineLength = sqrt(1.0 + params->u(k) * params->u(k));

                for (int j = 0; j < params->numRows; j++)
                {
                    //double v_new = params->v(j) / lineLength;
                    double v_new = params->v(j) * lineLength;

                    double ind = (v_new - v_0) / T_v;
                    int ind_closest = int(floor(0.5 + ind));
                    if (fabs(ind - double(ind_closest)) < 1.0e-8 || ind_closest < 0 || ind_closest > params->numRows-1)
                    {
                        ind_closest = max(0, min(params->numRows - 1, ind_closest));
                        aProj[j * params->numCols + k] = vertLine[ind_closest];
                    }
                    else
                    {
                        // Determine interpolation polynomial order and placement
                        int ind_lowest, M;
                        int M_default_local = order;
                        if (fabs(ind - double(ind_closest)) < 0.25)
                        {
                            // use odd length filter
                            M = min(M_default_local - ((M_default_local + 1) % 2), 1 + min(params->numCols - 1 - ind_closest, ind_closest));
                            if (M % 2 == 0)
                                M -= 1;
                            ind_lowest = ind_closest - (M - 1) / 2;
                        }
                        else
                        {
                            M = min(M_default_local - (M_default_local % 2), min(params->numCols - int(ceil(ind)), 1 + int(ind)));
                            if (M % 2 == 1)
                                M -= 1;
                            ind_lowest = int(ind) - M / 2 + 1;
                        }
                        if (M <= 1)
                        {
                            ind_closest = max(0, min(params->numRows - 1, ind_closest));
                            aProj[j * params->numCols + k] = vertLine[ind_closest];
                        }
                        else
                        {
                            double val = 0.0;
                            for (int l = 0; l < M; l++)
                            {
                                double h = 1.0;
                                for (int m = 0; m < M; m++)
                                {
                                    if (m != l)
                                        h *= double(ind - m - ind_lowest) / double(l - m);
                                }
                                val += h * vertLine[max(0, min(params->numRows-1, l + ind_lowest))];
                            }
                            aProj[j * params->numCols + k] = val;
                        }
                    }
                }
            }
            delete[] vertLine;
        }
        //*/
    }

    //return true;

    // Perform fan to parallel rebinning
    LOG(logDEBUG, "rebin", "rebin_parallel") << "performing fan to parallel rebinning..." << std::endl;
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int iRow = 0; iRow < params->numRows; iRow++)
    {
        // Make a copy of the sinogram
        float* sino = new float[params->numCols * params->numAngles];
        for (int iAngle = 0; iAngle < params->numAngles; iAngle++)
        {
            uint64 ind_offset = uint64(iAngle) * uint64(params->numRows * params->numCols) + uint64(iRow) * uint64(params->numCols);
            float* g_offset = &g[ind_offset];

            for (int iCol = 0; iCol < params->numCols; iCol++)
                sino[iAngle*params->numCols + iCol] = g_offset[iCol];
        }

        for (int iAngle = 0; iAngle < N_phi_new; iAngle++)
        {
            uint64 ind_offset = uint64(iAngle) * uint64(params->numRows * params->numCols) + uint64(iRow) * uint64(params->numCols);
            double phi = iAngle*T_phi + phi_0_new;
            for (int iCol = 0; iCol < params->numCols; iCol++)
            {
                double s = iCol*T_s + s_0;
                double alpha = asin(s / R_tau) + asin_tau_R_tau;
                double beta = phi + alpha;

                if (doWrapAround)
                {
                    if (beta < beta_min)
                        beta = beta + 2.0 * PI;
                    else if (beta > beta_max)
                        beta = beta - 2.0 * PI;
                    //if (beta < beta_min || beta > beta_max)
                    //    printf("beta error!\n");
                }
                else
                {
                    //if (iRow == params->numRows / 2 && (beta < beta_min || beta > beta_max))
                    //    printf("using conjugate ray: phi = %f, beta = %f, alpha = %f\n", phi, beta, alpha);
                    if (beta < beta_min)
                    {
                        double alpha_new = -alpha + atan_A;
                        beta = beta - (alpha - alpha_new) + PI;
                        alpha = alpha_new;
                    }
                    else if (beta > beta_max)
                    {
                        double alpha_new = -alpha + atan_A;
                        beta = beta - (alpha - alpha_new) - PI;
                        alpha = alpha_new;
                    }
                }

                double lateral = alpha;
                if (params->detectorType == parameters::FLAT)
                    lateral = tan(alpha);

                float beta_ind = params->phi_inv(beta);
                g[ind_offset + uint64(iCol)] = LagrangeInterpolation13(sino, params, beta_ind, (lateral-u_0)/T_u, iRow, doWrapAround, order);
            }
        }

        delete[] sino;
    }

    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
    //*/

    // Modify CT geometry parameters
    if (params->geometry == parameters::CONE)
        params->geometry = parameters::CONE_PARALLEL;
    else if (params->geometry == parameters::FAN)
        params->geometry = parameters::PARALLEL;

    if (N_phi_new != N_phi || phi_0_new != phi_0)
    {
        float* phis = new float[N_phi_new];
        for (int i = 0; i < N_phi_new; i++)
            phis[i] = (T_phi * i + phi_0_new) * 180.0 / PI + 90.0;
        params->set_angles(phis, N_phi_new);
        delete[] phis;

        uint64 offs = uint64(N_phi_new) * uint64(params->numRows) * uint64(params->numCols);
        float* g_crop = &g[offs];
        params->setToZero(g_crop, uint64(N_phi - N_phi_new) * uint64(params->numRows) * uint64(params->numCols));
        LOG(logWARNING, "rebin", "rebin_parallel") << "WARNING: Reducing number of angles from " << N_phi << " to " << N_phi_new << std::endl;
        LOG(logWARNING, "rebin", "rebin_parallel") << "New angular coverage: " << params->angularRange << " degrees" << std::endl;
    }

    params->centerCol = -s_0/T_s; // i* T_s + s_0 == 0
    params->tau = 0.0;
    params->sod = R_tau;
    params->pixelWidth = T_s;
    params->detectorType = parameters::FLAT;

    return true;
}

float rebin::LagrangeInterpolation13(float* g, parameters* params, double x1, double x3, int j, bool doWrapAround, int order)
{
    double ind;
    int ind_closest, ind_lowest, M;

    // Specify 1st dimension interpolation parameters
    double h_1[8];
    int ind_lowest_1, M_1;

    /*
    if (x1 >= 0.0)
        ind = int(x1 + 0.5);
    else
    {
        //printf("x1 = %f\n", x1);
        if (doWrapAround == false)
            ind = 0;
        else
            ind = -int(-x1 + 0.5);
    }
    //*/

    ind = x1;
    ind_closest = int(floor(0.5 + ind));
    ind_lowest = ind_closest;
    M = order;
    if (fabs(ind - double(ind_closest)) < 1.0e-8 || ind_closest < 0 || ind_closest > params->numAngles - 1)
    {
        ind_lowest_1 = max(0, min(params->numAngles - 1, ind_closest));
        M_1 = 1;
        h_1[0] = 1.0;
    }
    else
    {
        // Determine interpolation polynomial order and placement
        int ind_lowest, M;
        int M_default_local = order;
        if (fabs(ind - double(ind_closest)) < 0.25)
        {
            // use odd length filter
            M = min(M_default_local - ((M_default_local + 1) % 2), 1 + min(params->numAngles - 1 - ind_closest, ind_closest));
            if (M % 2 == 0)
                M -= 1;
            ind_lowest = ind_closest - (M - 1) / 2;
        }
        else
        {
            M = min(M_default_local - (M_default_local % 2), min(params->numAngles - int(ceil(ind)), 1 + int(ind)));
            if (M % 2 == 1)
                M -= 1;
            ind_lowest = int(ind) - M / 2 + 1;
        }

        ind_lowest_1 = ind_lowest;
        M_1 = M;

        if (M <= 1)
        {
            M_1 = 1;
            ind_lowest_1 = max(0, min(params->numAngles - 1, ind_lowest_1));
            h_1[0] = 1.0;
        }
        else
        {
            for (int l = 0; l < M_1; l++)
            {
                double h = 1.0;
                for (int m = 0; m < M_1; m++)
                {
                    if (m != l)
                        h *= double(ind - m - ind_lowest_1) / double(l - m);
                }
                h_1[l] = h;
            }
        }
    }

    // Specify 3rd dimension interpolation parameters
    double h_3[8];
    int ind_lowest_3, M_3;

    ind = x3;
    ind_closest = int(floor(0.5 + ind));
    ind_lowest = ind_closest;
    M = order;
    if (fabs(ind - double(ind_closest)) < 1.0e-8 || ind_closest < 0 || ind_closest > params->numCols - 1)
    {
        ind_lowest_3 = max(0, min(params->numCols - 1, ind_closest));
        M_3 = 1;
        h_3[0] = 1.0;
    }
    else
    {
        // Determine interpolation polynomial order and placement
        int ind_lowest, M;
        int M_default_local = order;
        if (fabs(ind - double(ind_closest)) < 0.25)
        {
            // use odd length filter
            M = min(M_default_local - ((M_default_local + 1) % 2), 1 + min(params->numCols - 1 - ind_closest, ind_closest));
            if (M % 2 == 0)
                M -= 1;
            ind_lowest = ind_closest - (M - 1) / 2;
        }
        else
        {
            M = min(M_default_local - (M_default_local % 2), min(params->numCols - int(ceil(ind)), 1 + int(ind)));
            if (M % 2 == 1)
                M -= 1;
            ind_lowest = int(ind) - M / 2 + 1;
        }

        ind_lowest_3 = ind_lowest;
        M_3 = M;

        if (M_3 <= 1)
        {
            M_3 = 1;
            ind_lowest_3 = max(0, min(params->numCols - 1, ind_lowest_3));
            h_3[0] = 1.0;
        }
        else
        {
            for (int l = 0; l < M_3; l++)
            {
                double h = 1.0;
                for (int m = 0; m < M_3; m++)
                {
                    if (m != l)
                        h *= double(ind - m - ind_lowest_3) / double(l - m);
                }
                h_3[l] = h;
            }
        }
    }

    double val = 0.0;
    for (int l1 = 0; l1 < M_1; l1++)
    {
        int i = l1 + ind_lowest_1;
        if (0 <= i && i < params->numAngles)
        {
            for (int l3 = 0; l3 < M_3; l3++)
            {
                int k = l3 + ind_lowest_3;
                if (0 <= k && k < params->numCols)
                {
                    val += h_1[l1] * h_3[l3] * g[i * params->numCols + k];
                }
            }
        }
    }

    return float(val);
}

/*
float rebin::LagrangeInterpolation13(float* g, parameters* params, double x1, double x3, int j, bool doWrapAround, int order)
{
    double x1_low, x3_low;

    if (x1 >= 0.0)
        x1_low = double(int(x1));
    else
        x1_low = double(int(x1) - 1);
    if (x3 >= 0.0)
        x3_low = double(int(x3));
    else
        x3_low = double(int(x3) - 1);

    // Find closest bin
    int x1_star, x3_star;

    if (x1 >= 0.0)
        x1_star = int(x1 + 0.5);
    else
    {
        if (doWrapAround == false)
            x1_star = 0;
        else
            x1_star = -int(-x1 + 0.5);
    }

    x3_star = int(x3 + 0.5);
    if (x3 < 0.0)
        x3_star = 0;
    else if (x3 > params->numCols - 1)
        x3_star = params->numCols - 1;

    int M_1 = order;
    int M_3 = order;

    if (doWrapAround == false)
    {
        if (x1 <= 0.0 || x1 >= double(params->numAngles - 1) || x1_low == x1)
            M_1 = 1;
        else if (int(x1_low) < M_1)
            M_1 = 2 * (int(x1_low) + 1);
        else if (params->numAngles - 1 - int(x1_low) < M_1)
            M_1 = 2 * (params->numAngles - 1 - int(x1_low));
    }

    if (x3 <= 0.05 || x3 >= double(params->numCols) - 1.05 || x3_low == x3)
        M_3 = 1;
    else if (int(x3_low) < M_3)
        M_3 = min(order, 2 * (int(x3_low) + 1));
    else if (params->numCols - 1 - int(x3_low) < M_3)
        M_3 = min(order, 2 * (params->numCols - 1 - int(x3_low)));

    // Switch between even and odd length filters, based on sample location
    if (fabs(x1 - x1_low - 0.5) > 0.25 && M_1 > 1)
        M_1 = M_1 - 1;
    if (fabs(x3 - x3_low - 0.5) > 0.25 && M_3 > 1)
        M_3 = M_3 - 1;

    // Set x1 interpolation coefficients
    double x1_shift;
    if (M_1 % 2 == 1)
        x1_shift = x1 - double(x1_star) + double(M_1 - 1) / 2.0;
    else
        x1_shift = x1 - x1_low + double(M_1 - 2) / 2.0;
    double* l_1 = LagrangeCoefficients(x1_shift, M_1);

    // Set x3 interpolation coefficients
    double x3_shift;
    if (M_3 % 2 == 1)
        x3_shift = x3 - double(x3_star) + double(M_3 - 1) / 2.0;
    else
        x3_shift = x3 - x3_low + double(M_3 - 2) / 2.0;
    if (M_3 == 1)
        x3_shift = 0;

    double* l_3 = LagrangeCoefficients(x3_shift, M_3);

    // Do interpolation
    int i_offset = x1 - x1_shift;
    int k_offset = x3 - x3_shift;
    int i_shift;

    double retVal = 0.0;
    for (int i = 0; i < M_1; i++)
    {
        double l_1_i = l_1[i];
        i_shift = i + i_offset;
        if (doWrapAround == true)
        {
            i_shift = i_shift % params->numAngles;
            if (i_shift < 0)
                i_shift += params->numAngles;
        }
        if (i_shift >= 0 && i_shift < params->numAngles)
        {
            uint64 ind_offset = uint64(i_shift) * uint64(params->numCols);
            float* aLine = &g[ind_offset];

            for (int k = 0; k < M_3; k++)
            {
                if (0 <= k + k_offset && k + k_offset < params->numCols)
                {
                    retVal += aLine[k + k_offset] * l_1_i * l_3[k];
                }
            }
        }
    }
    delete[] l_1;
    delete[] l_3;

    return float(retVal);
}
//*/

double* rebin::LagrangeCoefficients(double theShift, int N/* = 6*/, bool doFlip/* = false*/)
{
    N = max(N, 1);

    double* retVal = new double[N];

    for (int i = 0; i < N; i++)
    {
        int i_conj;
        if (doFlip == false)
            i_conj = i;
        else
            i_conj = N - 1 - i;
        retVal[i_conj] = 1.0;
        for (int m = 0; m < N; m++)
        {
            if (m != i)
                retVal[i_conj] *= (theShift - double(m)) / double(i - m);
        }
    }
    return retVal;
}

double rebin::fanAngles_inv(double val)
{
    int N = params->numCols;
    if (val <= fanAngles[0])
        return 0.0;
    else if (val >= fanAngles[N - 1])
        return double(N - 1);
    else
    {
        int ind = int(0.5 + (val - fanAngles[0]) / T_fanAngle);
        if (ind <= 0)
            return 0.0;
        else if (ind >= N-1)
            return double(N-1);
        else if (fanAngles[ind] < val)
        {
            while (fanAngles[ind] < val)
            {
                if (ind + 1 >= N-1)
                    return double(N-1);
                ind++;
            }
            // now columnAngles[ind-1] < val <= columnAngles[ind]
            return (val - fanAngles[ind - 1]) / (fanAngles[ind] - fanAngles[ind - 1]) + double(ind - 1);
        }
        else if (fanAngles[ind] > val)
        {
            while (fanAngles[ind] > val)
            {
                if (ind - 1 <= 0)
                    return 0.0;
                ind--;
            }
            // now columnAngles[ind] <= val < columnAngles[ind+1]
            return (val - fanAngles[ind]) / (fanAngles[ind + 1] - fanAngles[ind]) + double(ind);
        }
        else //if (columnAngles[ind] == val)
        {
            return double(ind);
        }
    }
}

bool rebin::rebin_curved(float* g, parameters* params_in, float* fanAngles_in, int order)
{
    fanAngles = fanAngles_in;
    params = params_in;
    if (fanAngles == NULL || g == NULL || params == NULL || params->numCols <= 2)
        return false;
    if (params->geometry != parameters::CONE || params->detectorType != parameters::CURVED)
    {
        printf("Error: rebin_curved: input data must be specified as curved cone-beam\n");
        return false;
    }

    T_fanAngle = 0.0;
    for (int i = 0; i < params->numCols-1; i++)
    {
        T_fanAngle += fabs(fanAngles[i+1] - fanAngles[i]);
    }
    T_fanAngle = T_fanAngle / double(params->numCols - 1);

    //*
    int* Ms = (int*) malloc(size_t(params->numCols)*sizeof(int));
    double** hs = (double**)malloc(size_t(params->numCols) * sizeof(double*));
    int* ind_lowests = (int*)malloc(size_t(params->numCols) * sizeof(int));
    //int Ms[1024];
    //double hs[1024][8];
    //int ind_lowests[1024];
    double maxDiff = 0.0;
    for (int k = 0; k < params->numCols; k++)
    {
        hs[k] = (double*)malloc(size_t(8) * sizeof(double));

        double targetAngle = 180.0 / PI * params->u(k);
        double ind = fanAngles_inv(targetAngle);

        int ind_closest = int(ind + 0.5);

        maxDiff = max(fabs(targetAngle - fanAngles[ind_closest]), maxDiff);

        if (fabs(ind - double(ind_closest)) < 1.0e-8)
        {
            Ms[k] = 1;
            hs[k][0] = 1;
            ind_lowests[k] = ind_closest;
        }
        else
        {
            int ind_lowest, M;
            int M_default_local = order;
            if (fabs(ind - double(ind_closest)) < 0.25)
            {
                // use odd length filter
                M = min(M_default_local - ((M_default_local + 1) % 2), 1 + min(params->numCols-1 - ind_closest, ind_closest));
                if (M % 2 == 0)
                    M -= 1;
                ind_lowest = ind_closest - (M - 1) / 2;
                Ms[k] = M;
                ind_lowests[k] = ind_lowest;
            }
            else
            {
                M = min(M_default_local - (M_default_local % 2), min(params->numCols - int(ceil(ind)), 1 + int(ind)));
                if (M % 2 == 1)
                    M -= 1;
                ind_lowest = int(ind) - M / 2 + 1;
                //int(ceil(ind)) + M/2-1 - int(ind) + M/2 + 1-1 = 2+M
                Ms[k] = M;
                ind_lowests[k] = ind_lowest;
            }
            //M = 1; // just for testing
            if (M <= 1)
            {
                Ms[k] = 1;
                hs[k][0] = 1;
                ind_lowests[k] = ind_closest;
            }
            else
            {
                for (int l = 0; l < M; l++)
                {
                    double h = 1.0;
                    for (int m = 0; m < M; m++)
                    {
                        if (m != l)
                            h *= (targetAngle - fanAngles[m + ind_lowest]) / (fanAngles[l + ind_lowest] - fanAngles[m + ind_lowest]);
                    }
                    hs[k][l] = h;
                }
            }
        }
    }

    //*
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < params->numAngles; i++)
    {
        float* targetProj = &g[uint64(i)*uint64(params->numRows*params->numCols)];
        float* sourceProj = (float*) malloc(size_t(params->numRows * params->numCols)*sizeof(float));

        memcpy(sourceProj, targetProj, sizeof(float) * params->numRows * params->numCols);

        for (int j = 0; j < params->numRows; j++)
        {
            float* targetLine = &targetProj[j*params->numCols];
            float* sourceLine = &sourceProj[j * params->numCols];

            for (int k = 0; k < params->numCols; k++)
            {
                double tempInterp = hs[k][0] * sourceLine[ind_lowests[k]];
                for (int l = 1; l < Ms[k]; l++)
                {
                    tempInterp += hs[k][l] * sourceLine[l + ind_lowests[k]];
                }
                targetLine[k] = tempInterp;
            }
        }
    }
    //*/

    // Clean up
    free(ind_lowests);
    free(Ms);
    for (int k = 0; k < params->numCols; k++)
    {
        free(hs[k]);
        hs[k] = NULL;
    }
    free(hs);

    return true;
}
