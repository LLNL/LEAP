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

analyticRayTracing::analyticRayTracing(parameters* params_in)
{
    params = params_in;
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

    float* s = NULL;
    float* c = NULL;
    float* u_vec = NULL;
    float* v_vec = NULL;

    if (params->geometry == parameters::PARALLEL)
    {
        r[0] = -cos_phi;
        r[1] = -sin_phi;
        r[2] = 0.0;
    }
    else if (params->geometry == parameters::FAN)
    {
        r[0] = -(cos_phi + u * sin_phi);
        r[1] = -(sin_phi - u * cos_phi);
        r[2] = 0.0;
    }
    else if (params->geometry == parameters::CONE)
    {
        if (params->detectorType == parameters::CURVED)
        {
            r[0] = -cos(phi - u);
            r[1] = -sin(phi - u);
            r[2] = v;
        }
        else
        {
            if (fabs(params->pitchAngle) > 1.0e-6)
            {
                double cos_tilt = cos(params->tiltAngle * PI / 180.0);
                double sin_tilt = sin(params->tiltAngle * PI / 180.0);

                double cos_pitch = cos(params->pitchAngle * PI / 180.0);
                double sin_pitch = sin(params->pitchAngle * PI / 180.0);

                r[0] = -cos_phi*cos_pitch + u*(-sin_phi*cos_tilt + cos_phi*sin_pitch*sin_tilt) + v*(sin_phi*sin_tilt + cos_phi*sin_pitch*cos_tilt);
				r[1] = -sin_phi*cos_pitch + u*(cos_phi*cos_tilt + sin_phi*sin_pitch*sin_tilt) + v*(-cos_phi*sin_tilt + sin_phi*sin_pitch*cos_tilt),
				r[2] = sin_pitch + u*(cos_pitch*sin_tilt) + v*(cos_pitch*cos_tilt);
            }
            else if (fabs(params->tiltAngle) > 1.0e-6)
            {
                double cos_tilt = cos(params->tiltAngle * PI / 180.0);
                double sin_tilt = sin(params->tiltAngle * PI / 180.0);
                double u_tilt = u * cos_tilt - v * sin_tilt;
                double v_tilt = u * sin_tilt + v * cos_tilt;

                r[0] = -(cos_phi + u_tilt * sin_phi);
                r[1] = -(sin_phi - u_tilt * cos_phi);
                r[2] = v_tilt;
            }
            else
            {
                r[0] = -(cos_phi + u * sin_phi);
                r[1] = -(sin_phi - u * cos_phi);
                r[2] = v;
            }
        }
    }
    else if (params->geometry == parameters::MODULAR)
    {
        s = &(params->sourcePositions[iProj * 3]);
        c = &(params->moduleCenters[iProj * 3]);
        u_vec = &(params->colVectors[iProj * 3]);
        v_vec = &(params->rowVectors[iProj * 3]);
        r[0] = c[0] + u * u_vec[0] + v * v_vec[0] - s[0];
        r[1] = c[1] + u * u_vec[1] + v * v_vec[1] - s[1];
        r[2] = c[2] + u * u_vec[2] + v * v_vec[2] - s[2];
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        r[0] = -cos_phi;
        r[1] = -sin_phi;
        r[2] = v;
    }
    else
    {
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

bool analyticRayTracing::setModuleCenter(int iProj, double* v)
{
    if (params->geometry == parameters::CONE)
    {
        double phi = 0.0;
        if (params->phis != NULL)
            phi = params->phis[iProj];
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);

        float cos_tilt = cos(params->tiltAngle * PI / 180.0);
        float sin_tilt = sin(params->tiltAngle * PI / 180.0);
        if (fabs(params->tiltAngle) < 1.0e-6)
        {
            cos_tilt = 1.0;
            sin_tilt = 0.0;
        }

        float cos_pitch = cos(params->pitchAngle * PI / 180.0);
        float sin_pitch = sin(params->pitchAngle * PI / 180.0);
        if (fabs(params->pitchAngle) < 1.0e-6)
        {
            cos_pitch = 1.0;
            sin_pitch = 0.0;
        }

        /*
        s_pos[3 * iphi + 0] = sod * cos_phi + tau * sin_phi;
		s_pos[3 * iphi + 1] = sod * sin_phi - tau * cos_phi;
		s_pos[3 * iphi + 2] = z_source(iphi);

		d_pos[3 * iphi + 0] = s_pos[3 * iphi + 0] - cos_phi*cos_pitch*sdd;
		d_pos[3 * iphi + 1] = s_pos[3 * iphi + 1] - sin_phi*cos_pitch*sdd;
		d_pos[3 * iphi + 2] = s_pos[3 * iphi + 2] + sin_pitch*sdd;
        */

        v[0] = params->sod * cos_phi + params->tau * sin_phi - cos_phi*cos_pitch*params->sdd;
		v[1] = params->sod * sin_phi - params->tau * cos_phi - sin_phi*cos_pitch*params->sdd;
		v[2] = params->z_source(iProj) + sin_pitch*params->sdd;

        return true;
    }
    else if (params->geometry == parameters::MODULAR)
    {
        v[0] = params->moduleCenters[iProj * 3 + 0];
        v[1] = params->moduleCenters[iProj * 3 + 1];
        v[2] = params->moduleCenters[iProj * 3 + 2];
        return true;
    }
    else
    {
        return false;
    }
}

bool analyticRayTracing::setRowVector(int iProj, double* v)
{
    if (params->geometry == parameters::CONE)
    {
        double phi = 0.0;
        if (params->phis != NULL)
            phi = params->phis[iProj];
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);

        float cos_tilt = cos(params->tiltAngle * PI / 180.0);
        float sin_tilt = sin(params->tiltAngle * PI / 180.0);
        if (fabs(params->tiltAngle) < 1.0e-6)
        {
            cos_tilt = 1.0;
            sin_tilt = 0.0;
        }

        float cos_pitch = cos(params->pitchAngle * PI / 180.0);
        float sin_pitch = sin(params->pitchAngle * PI / 180.0);
        if (fabs(params->pitchAngle) < 1.0e-6)
        {
            cos_pitch = 1.0;
            sin_pitch = 0.0;
        }

        v[0] = sin_phi * sin_tilt + cos_phi * sin_pitch * cos_tilt;
        v[1] = -cos_phi * sin_tilt + sin_phi * sin_pitch * cos_tilt;
        v[2] = cos_pitch * cos_tilt;
        return true;
    }
    else if (params->geometry == parameters::MODULAR)
    {
        v[0] = params->rowVectors[iProj * 3 + 0];
        v[1] = params->rowVectors[iProj * 3 + 1];
        v[2] = params->rowVectors[iProj * 3 + 2];
        return true;
    }
    else
        return false;
}

bool analyticRayTracing::setColVector(int iProj, double* v)
{
    if (params->geometry == parameters::CONE)
    {
        double phi = 0.0;
        if (params->phis != NULL)
            phi = params->phis[iProj];
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);

        float cos_tilt = cos(params->tiltAngle * PI / 180.0);
        float sin_tilt = sin(params->tiltAngle * PI / 180.0);
        if (fabs(params->tiltAngle) < 1.0e-6)
        {
            cos_tilt = 1.0;
            sin_tilt = 0.0;
        }

        float cos_pitch = cos(params->pitchAngle * PI / 180.0);
        float sin_pitch = sin(params->pitchAngle * PI / 180.0);
        if (fabs(params->pitchAngle) < 1.0e-6)
        {
            cos_pitch = 1.0;
            sin_pitch = 0.0;
        }

        v[0] = -sin_phi * cos_tilt + cos_phi * sin_pitch * sin_tilt;
		v[1] = cos_phi * cos_tilt + sin_phi * sin_pitch * sin_tilt;
		v[2] = cos_pitch * sin_tilt;
        return true;
    }
    else if (params->geometry == parameters::MODULAR)
    {
        v[0] = params->colVectors[iProj * 3 + 0];
        v[1] = params->colVectors[iProj * 3 + 1];
        v[2] = params->colVectors[iProj * 3 + 2];
        return true;
    }
    else
        return false;
}

bool analyticRayTracing::setDetectorNormal(int iView, double* v)
{
    double colVector[3];
    double rowVector[3];
    if (setRowVector(iView, rowVector) == false)
        return false;
    if (setColVector(iView, colVector) == false)
        return false;
    v[0] = colVector[1] * rowVector[2] - colVector[2] * rowVector[1];
	v[1] = colVector[2] * rowVector[0] - colVector[0] * rowVector[2];
	v[2] = colVector[0] * rowVector[1] - colVector[1] * rowVector[0];
    return true;
}
