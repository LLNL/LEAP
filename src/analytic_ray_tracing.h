////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////
#ifndef __ANALYTIC_RAY_TRACING_H
#define __ANALYTIC_RAY_TRACING_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "phantom.h"

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to perform analytic ray tracing simulation through geometric solids.
 */

class analyticRayTracing
{
public:

    // Constructor and destructor; these do nothing
    analyticRayTracing();
    ~analyticRayTracing();

    bool rayTrace(float* g, parameters* params_in, phantom* aPhantom, int oversampling = 1);

private:
    bool setSourcePosition(int, int, int, double*, double dv = 0.0, double du = 0.0);
    bool setTrajectory(int, int, int, double*, double dv = 0.0, double du = 0.0);
    
    parameters* params;
};

#endif
