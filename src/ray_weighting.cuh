////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for ray weighting
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAY_WEIGHTING_CUH
#define __RAY_WEIGHTING_CUH

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"
#include <stdlib.h>

/**
 * This header and associated source file provide CUDA-based implementations of functions to calculate
 * the ray weighting steps of various FBP algorithms.
 */

//bool applyPreRampFilterWeights(float* g, parameters*, bool data_on_cpu);
//bool applyPostRampFilterWeights(float* g, parameters*, bool data_on_cpu);

bool applyPreRampFilterWeights_GPU(float* g, parameters*, bool data_on_cpu);
bool applyPostRampFilterWeights_GPU(float* g, parameters*, bool data_on_cpu);

bool convertARTtoERT(float* g, parameters*, bool data_on_cpu, bool doInverse=false);

bool applyViewDependentPolarWeights_gpu(float* g, parameters* params, float* w, bool data_on_cpu, bool doInverse);

#endif
