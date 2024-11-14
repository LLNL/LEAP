////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include "analytic_ray_tracing.h"
#include "leap_defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>

using namespace std;

analyticRayTracing::analyticRayTracing()
{
    params = NULL;
}

analyticRayTracing::~analyticRayTracing()
{
}

bool analyticRayTracing::rayTrace(float* g, parameters* params_in, phantom* aPhantom, int oversampling)
{
    params = params_in;
    if (g == NULL || params == NULL || aPhantom == NULL)
        return false;
    
    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    if (params->geometry == parameters::MODULAR)
        params->normalizeConeAndFanCoordinateFunctions = false;
    else
        params->normalizeConeAndFanCoordinateFunctions = true;

    oversampling = max(1, min(oversampling, 11));
    if (oversampling % 2 == 0)
        oversampling += 1;
    oversampling = max(1, min(oversampling, 11));

    /*
    double sourcePos[3];
    double r[3];
    setSourcePosition(0, 0, params->numCols/2, sourcePos);
    setTrajectory(0, 0, params->numCols / 2, r);
    double val = aPhantom->lineIntegral(sourcePos, r);
    printf("ray trace value = %f\n", val);
    return false;
    //*/

    int num_threads = omp_get_num_procs();
    aPhantom->makeTempData(num_threads);

    if (oversampling == 1)
    {
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < params->numAngles; i++)
        {
            double sourcePos[3];
            double r[3];
            float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];

            if (params->geometry == parameters::MODULAR || params->geometry == parameters::CONE)
                setSourcePosition(i, 0, 0, sourcePos);

            for (int j = 0; j < params->numRows; j++)
            {
                float* aLine = &aProj[j * params->numCols];

                if (params->geometry == parameters::FAN)
                    setSourcePosition(i, j, 0, sourcePos);

                for (int k = 0; k < params->numCols; k++)
                {
                    if (params->geometry == parameters::PARALLEL || params->geometry == parameters::CONE_PARALLEL)
                        setSourcePosition(i, j, k, sourcePos);
                    setTrajectory(i, j, k, r);
                    aLine[k] = float(aPhantom->lineIntegral(sourcePos, r));
                }
            }
        }
    }
    else
    {
        double T_v_os = double(params->v(1) - params->v(0)) / double(oversampling + 1);
        double T_u_os = double(params->u(1) - params->u(0)) / double(oversampling + 1);

        int os_radius = (oversampling - 1) / 2;

        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < params->numAngles; i++)
        {
            //double sourcePos_save[3];
            //double r_save[3];

            double sourcePos[3];
            double r[3];
            float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];

            if (params->geometry == parameters::MODULAR || params->geometry == parameters::CONE)
                setSourcePosition(i, 0, 0, sourcePos);

            for (int j = 0; j < params->numRows; j++)
            {
                float* aLine = &aProj[j * params->numCols];

                //if (params->geometry == parameters::FAN)
                //    setSourcePosition(i, j, 0, sourcePos);

                for (int k = 0; k < params->numCols; k++)
                {
                    double accum = 0.0;
                    for (int j_os = -os_radius; j_os <= os_radius; j_os++)
                    {
                        double dv = j_os * T_v_os;

                        if (params->geometry == parameters::FAN)
                            setSourcePosition(i, j, 0, sourcePos, dv);

                        for (int k_os = -os_radius; k_os <= os_radius; k_os++)
                        {
                            double du = k_os * T_u_os;

                            if (params->geometry == parameters::PARALLEL || params->geometry == parameters::CONE_PARALLEL)
                                setSourcePosition(i, j, k, sourcePos, dv, du);
                            setTrajectory(i, j, k, r, dv, du);
                            accum += exp(-aPhantom->lineIntegral(sourcePos, r));
                        }
                    }
                    
                    aLine[k] = -log(accum / double(oversampling * oversampling));
                }
            }
        }
    }

    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
    
    return true;
}

bool analyticRayTracing::setSourcePosition(int iProj, int iRow, int iCol, double* sourcePos, double dv, double du)
{
    if (sourcePos == NULL)
        return false;

    double phi = 0.0;
    if (params->phis != NULL)
        phi = params->phis[iProj];
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);

    float s, sqrt_R2_minus_s2;

    switch (params->geometry)
    {
    case parameters::PARALLEL:
        sourcePos[0] = -(params->u(iCol)+du) * sin_phi;
        sourcePos[1] = (params->u(iCol)+du) * cos_phi;
        sourcePos[2] = params->v(iRow)+dv;
        break;
    case parameters::FAN:
        sourcePos[0] = params->sod * cos_phi + params->tau * sin_phi;
        sourcePos[1] = params->sod * sin_phi - params->tau * cos_phi;
        sourcePos[2] = params->v(iRow)+dv;
        break;
    case parameters::CONE:
        sourcePos[0] = params->sod * cos_phi + params->tau * sin_phi;
        sourcePos[1] = params->sod * sin_phi - params->tau * cos_phi;
        sourcePos[2] = params->z_source(iProj);
        break;
    case parameters::MODULAR:
        sourcePos[0] = params->sourcePositions[iProj * 3 + 0];
        sourcePos[1] = params->sourcePositions[iProj * 3 + 1];
        sourcePos[2] = params->sourcePositions[iProj * 3 + 2];
        break;
    case parameters::CONE_PARALLEL:
        s = params->u(iCol) + du;
        sqrt_R2_minus_s2 = sqrt(params->sod * params->sod - s * s);
        sourcePos[0] = -s * sin_phi + sqrt_R2_minus_s2 * cos_phi;
        sourcePos[1] = s * cos_phi + sqrt_R2_minus_s2 * sin_phi;
        sourcePos[2] = params->z_source(iProj, iCol);
        break;
    default:
        sourcePos[0] = 0.0;
        sourcePos[1] = 0.0;
        sourcePos[2] = 0.0;
    }
    return true;
}

bool analyticRayTracing::setTrajectory(int iProj, int iRow, int iCol, double* r, double dv, double du)
{
    if (r == NULL)
        return false;

    double u, v;

    if (params->geometry != parameters::MODULAR)
    {
        u = params->u(iCol) + du;
        v = params->v(iRow) + dv;
    }
    else
    {
        u = params->col(iCol) + du;
        v = params->row(iRow) + dv;
    }


    double phi = 0.0;
    if (params->phis != NULL)
        phi = params->phis[iProj];
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);

    double cos_tilt = 1.0;
    double sin_tilt = 0.0;
    if (fabs(params->tiltAngle) > 1.0e-6 && params->geometry == parameters::CONE)
    {
        cos_tilt = cos(params->tiltAngle * PI / 180.0);
        sin_tilt = sin(params->tiltAngle * PI / 180.0);
        double u_tilt = u * cos_tilt - v * sin_tilt;
        double v_tilt = u * sin_tilt + v * cos_tilt;
        u = u_tilt;
        v = v_tilt;
    }

    float* s = NULL;
    float* c = NULL;
    float* u_vec = NULL;
    float* v_vec = NULL;

    switch (params->geometry)
    {
    case parameters::PARALLEL:
        r[0] = -cos_phi;
        r[1] = -sin_phi;
        r[2] = 0.0;
        break;
    case parameters::FAN:
        r[0] = -(cos_phi + u * sin_phi);
        r[1] = -(sin_phi - u * cos_phi);
        r[2] = 0.0;
        break;
    case parameters::CONE:
        if (params->detectorType == parameters::CURVED)
        {
            r[0] = -cos(phi - u);
            r[1] = -sin(phi - u);
            r[2] = v;
        }
        else
        {
            r[0] = -(cos_phi + u * sin_phi);
            r[1] = -(sin_phi - u * cos_phi);
            r[2] = v;
        }
        break;
    case parameters::MODULAR:
        s = &(params->sourcePositions[iProj * 3]);
        c = &(params->moduleCenters[iProj * 3]);
        u_vec = &(params->colVectors[iProj * 3]);
        v_vec = &(params->rowVectors[iProj * 3]);
        r[0] = c[0] + u * u_vec[0] + v * v_vec[0] - s[0];
        r[1] = c[1] + u * u_vec[1] + v * v_vec[1] - s[1];
        r[2] = c[2] + u * u_vec[2] + v * v_vec[2] - s[2];
        break;
    case parameters::CONE_PARALLEL:
        r[0] = -cos_phi;
        r[1] = -sin_phi;
        r[2] = v;
        break;
    default:
        r[0] = 0.0;
        r[1] = 0.0;
        r[2] = 0.0;
    }

    double mag = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    r[0] = r[0] / mag;
    r[1] = r[1] / mag;
    r[2] = r[2] / mag;

    return true;
}
