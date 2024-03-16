////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
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

private:
    
    double fanAngles_inv(double);

    float* fanAngles;
    double T_fanAngle;
    parameters* params;
};

#endif
