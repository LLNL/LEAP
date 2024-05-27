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
