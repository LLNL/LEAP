////////////////////////////////////////////////////////////////////////////////
// Copyright 2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#ifndef __ANALYTIC_RAY_TRACING_CUH
#define __ANALYTIC_RAY_TRACING_CUH

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include <vector>
#include "parameters.h"
#include "phantom.h"

/**
 * This class provides GPU-based implementations to perform analytic ray tracing simulation through geometric solids.
 */

struct geometricSolid
{
	int type;
    float3 centers;
    float3 radii;
    float val;
    float A[9];
    float clippingPlanes[6][4];
    
    bool isRotated;
    int numClippingPlanes;
    float2 clipCone;
};

void setConstantMemoryGeometryParameters(parameters* params, int oversampling = 1);
bool rayTrace_gpu(float* g, parameters* params, phantom* aPhantom, bool data_on_cpu, int oversampling = 1);

#endif
