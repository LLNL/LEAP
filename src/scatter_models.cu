////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module scatter simulation and correction
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.h"

#include "scatter_models.cuh"

bool simulateScatter_firstOrder_singleMaterial(float* g, float* f, parameters* params, float* source, float* energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu)
{
    return false;
}
