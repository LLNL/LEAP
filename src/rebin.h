////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Rebinning algorithms
////////////////////////////////////////////////////////////////////////////////
#ifndef __REBIN_H
#define __REBIN_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to rebin CT projection data.
 */

class rebin
{
public:

    // Constructor and destructor; these do nothing
    rebin();
    ~rebin();

    //rebin a collection of flat detector modules to a curved detector.
    bool rebin_curved(float* g, parameters* params_in, float* fanAngles_in, int order = 6);

    bool rebin_parallel(float* g, parameters* params_in, int order = 6);
    float* rebin_parallel_singleProjection(float* g, parameters* params_in, int order, float desiredAngle);
    int rebin_parallel_singleSinogram(float* g, parameters* params_in, float* parallel_sinogram, int order = 6, int desiredRow=-1, bool reduce180 = true);

private:
    
    double fanAngles_inv(double);

    float LagrangeInterpolation13(float* g, parameters* params, double x1, double x3, int j, bool doWrapAround = false, int order=6);
    double* LagrangeCoefficients(double theShift, int N = 6, bool doFlip = false);

    float* fanAngles;
    double T_fanAngle;
    parameters* params;

    double phi_0;
    double phi_0_new;
    double T_phi;
    int N_phi;
    int N_phi_new;
};

#endif
