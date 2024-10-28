////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based sinogram replacement (a MAR method) routines
////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "sinogram_replacement.h"

using namespace std;

bool sinogramReplacement(float* g, float* priorSinogram, float* metalTrace, parameters* params, int* windowSize)
{
    if (g == NULL || priorSinogram == NULL || metalTrace == NULL || params == NULL || windowSize == NULL)
        return false;

    bool retVal = true;

    int minSamples = 12;
    double scale_softConstraintWeight = 0.4;
    double shift_softConstraintWeight = 0.4;
    double linearTerm_max = 1.5;
    double linearTerm_min = 1.0 / linearTerm_max;
    double minAttenuationToReplace = 0.0;

    int N_viewWindow = max(2, windowSize[0]);
    int N_rowWindow = max(0, windowSize[1]);
    int N_colWindow = max(2, windowSize[2]);
    //printf("window size: %d x %d x %d\n", 2 * N_viewWindow + 1, 2 * N_rowWindow + 1, 2 * N_colWindow + 1);
    
    int debugSlice = params->numRows / 2;

    float* fitCoefficients = (float*)malloc(size_t(uint64(2*params->numAngles)*uint64(params->numCols))*sizeof(float));
    
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int j = 0; j < params->numRows; j++)
    {
        for (int i = 0; i < params->numAngles; i++)
        {
            float* scaleLine = NULL;
            float* shiftLine = NULL;
            if (fitCoefficients != NULL && j == debugSlice)
            {
                scaleLine = &fitCoefficients[uint64(i)*uint64(params->numCols)];
                shiftLine = &fitCoefficients[uint64(i+params->numAngles) * uint64(params->numCols)];
            }
            
            float* g_line = &g[uint64(i) * uint64(params->numRows*params->numCols) + uint64(j * params->numCols)];
            float* p_line = &priorSinogram[uint64(i) * uint64(params->numRows * params->numCols) + uint64(j * params->numCols)];
            float* m_line = &metalTrace[uint64(i) * uint64(params->numRows * params->numCols) + uint64(j * params->numCols)];
            
            vector<float*> g_lines;
            vector<float*> p_lines;
            vector<float*> m_lines;
            for (int di = -N_viewWindow; di <= N_viewWindow; di++)
            {
                if (0 <= i+di && i+di < params->numAngles)
                {
                    for (int dj = -N_rowWindow; dj <= N_rowWindow; dj++)
                    {
                        if (0 <= j+dj && j+dj < params->numRows)
                        {
                            g_lines.push_back(&g[uint64(i+di) * uint64(params->numRows * params->numCols) + uint64((j+dj) * params->numCols)]);
                            p_lines.push_back(&priorSinogram[uint64(i + di) * uint64(params->numRows * params->numCols) + uint64((j + dj) * params->numCols)]);
                            m_lines.push_back(&metalTrace[uint64(i + di) * uint64(params->numRows * params->numCols) + uint64((j + dj) * params->numCols)]);
                        }
                    }
                }
            }
            
            vector<int> goodData_startIndex;
            vector<int> goodData_finalIndex;
            bool isInsideMetalTrace = false;
            goodData_startIndex.push_back(0); // always assume first sample is not a metal trace
            for (int k = 0; k < params->numCols; k++)
            {
                if (isInsideMetalTrace == false)
                {
                    if (m_line[k] > 0.0) // is inside metal trace (bad data)
                    {
                        isInsideMetalTrace = true;
                        // switched from outside trace to inside trace
                        goodData_finalIndex.push_back(k-1);
                    }
                    //else // continue to lie inside metal trace
                }
                else
                {
                    if (m_line[k] <= 0.0) // is outside metal trace (good data)
                    {
                        isInsideMetalTrace = false;
                        // switched from inside trace to outside trace
                        goodData_startIndex.push_back(k);
                    }
                    //else // continue to lie inside metal trace
                }
            }
            goodData_finalIndex.push_back(params->numCols-1); // always assume last sample is not a metal trace
            
            if (goodData_startIndex.size() != goodData_finalIndex.size())
            {
                // this can only happen if there is some bug in this code, i.e., if this warning message is ever displayed, I need to fix the code!
                printf("interval mismatch (view=%d, row=%d)!!!\n", i, j);
                retVal = false;
                continue;
            }
            
            if (goodData_startIndex.size() > 1 && goodData_finalIndex[0] < params->numCols -1) // make sure this view has a metal trace
            {
                for (int n = 0; n < goodData_startIndex.size()-1; n++)
                {
                    // this iteration will fix nth bad data section k \in [goodData_finalIndex[n]+1, goodData_startIndex[n+1]-1]
                    vector<float> g_data;
                    vector<float> p_data;
                    int numSamples_left = 0;
                    int numSamples_right = 0;
                    
                    int k_left = goodData_finalIndex[n];
                    int k_right = goodData_startIndex[n+1];
                    
                    int metal_ind_left = k_left+1;
                    int metal_ind_right = k_right-1;
                    
                    int k_min = max(0, k_left-2*N_colWindow); // -4 -3 -2 -1 0
                    for (int k = k_left; k >= k_min; k--)
                    {
                        for (int l = 0; l < g_lines.size(); l++)
                        {
                            if (m_lines[l][k] <= 0.0)
                            {
                                g_data.push_back(g_lines[l][k]);
                                p_data.push_back(p_lines[l][k]);
                                numSamples_left += 1;
                            }
                        }
                        if (numSamples_left >= minSamples) // used to be 12
                            break;
                    }
                    
                    int k_max = min(params->numCols -1, k_right+2*N_colWindow);
                    for (int k = k_right; k <= k_max; k++)
                    {
                        for (int l = 0; l < g_lines.size(); l++)
                        {
                            if (m_lines[l][k] <= 0.0)
                            {
                                g_data.push_back(g_lines[l][k]);
                                p_data.push_back(p_lines[l][k]);
                                numSamples_right += 1;
                            }
                        }
                        if (numSamples_right >= minSamples) // used to be 12
                            break;
                    }
                    
                    if (numSamples_left >= 2 && numSamples_right >= 2)
                    {
                        double A = 0.0;
                        double B = 0.0;
                        double C = 0.0;
                        //double D = 0.0;
                        double EE = 0.0;
                        double F = 0.0;
                        for (int n = 0; n < g_data.size(); n++)
                        {
                            A += p_data[n]*p_data[n];
                            B += p_data[n];
                            C += 1.0;
                            
                            EE += g_data[n]*p_data[n];
                            F += g_data[n];
                        }
                        //A += ridgeRegressionWeight;
                        //C += ridgeRegressionWeight;
                        //EE += ridgeRegressionWeight;
                        A += scale_softConstraintWeight;
                        C += shift_softConstraintWeight;
                        EE += scale_softConstraintWeight;
                        
                        double det = A*C-B*B;
                        if (det != 0.0)
                        {
                            double linearTerm = max(linearTerm_min, min((C*EE-B*F) / det, linearTerm_max));
                            double constantTerm = (F-linearTerm*B) / C;
                            for (int k = metal_ind_left; k <= metal_ind_right; k++)
                            {
                                if (g_line[k] > minAttenuationToReplace)
                                {
                                    g_line[k] = linearTerm*p_line[k] + constantTerm;
                                    if (scaleLine != NULL && shiftLine != NULL)
                                    {
                                        scaleLine[k] = linearTerm;
                                        shiftLine[k] = constantTerm;
                                    }
                                }
                            }
                        }
                    }
                    
                }
            }
        }
    }
    return retVal;
}
