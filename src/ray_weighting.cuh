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

bool applyPreRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);

bool applyPreRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);

bool convertARTtoERT(float* g, parameters*, bool cpu_to_gpu, bool doInverse=false);

#endif
