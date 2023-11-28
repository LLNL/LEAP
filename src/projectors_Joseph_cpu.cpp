////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for cpu projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors_Joseph_cpu.h"

using namespace std;

bool project_Joseph_cpu(float* g, float* f, parameters* params)
{
    return false;
}

bool backproject_Joseph_cpu(float* g, float* f, parameters* params)
{
    return false;
}

bool project_Joseph_modular_cpu(float* g, float* f, parameters* params)
{
    return false;
}

bool backproject_Joseph_modular_cpu(float* g, float* f, parameters* params)
{
    return false;
}

float projectLine_Joseph(float* f, parameters* params, float* pos, float* traj)
{
    return 0.0;
}
