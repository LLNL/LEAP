////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// projectors class header which selects the appropriate projector to use based on the
// CT geometry and CT volume specification, and some other parameters such as
// whether the calculation should happen on the CPU or GPU
////////////////////////////////////////////////////////////////////////////////
#ifndef __PROJECTORS_H
#define __PROJECTORS_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include "parameters.h"

/**
 *  projectors class
 * This class is responsible for all the logic required for CPU- and GPU-based forward projection and backprojection algorithms.
 * Based on whether one wishes to run the computation and the geometry, this class dispatches the correct algorithm.
 */

class projectors
{
public:
    projectors();
    ~projectors();

    bool project(float* g, float* f, parameters* params, bool cpu_to_gpu);
    bool backproject(float* g, float* f, parameters* params, bool cpu_to_gpu);

    bool weightedBackproject(float* g, float* f, parameters* params, bool cpu_to_gpu);
};

#endif
