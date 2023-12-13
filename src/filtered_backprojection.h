////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Class for performing analytic inversion, i.e., Filtered Backprojection (FBP)
// algorithms.
////////////////////////////////////////////////////////////////////////////////
#ifndef __FILTERED_BACKPROJECTION_H
#define __FILTERED_BACKPROJECTION_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "projectors.h"

/**
 *  filteredBackprojection class
 * This class is responsible for all the logic required for analytic reconstruction, e.g., Filtered Backprojection (FBP),
 * of a particular geometry.
 */

class filteredBackprojection
{
public:
    filteredBackprojection();
    ~filteredBackprojection();

    bool HilbertFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar, float sampleShift = 0.0);
    bool rampFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar);
    bool filterProjections(float* g, parameters* ctParams, bool cpu_to_gpu);

    bool execute(float* g, float* f, parameters* params, bool cpu_to_gpu);

private:
    bool convolve1D(float* g, parameters* params, bool cpu_to_gpu, float scalar, int whichFilter, float sampleShift = 0.0);
    bool filterProjections_Novikov(float* g, parameters* ctParams, bool cpu_to_gpu);

    bool execute_attenuated(float* g, float* f, parameters* params, bool cpu_to_gpu);
    bool execute_Novikov(float* g, float* f, parameters* params, bool cpu_to_gpu);

    projectors proj;
};

#endif
