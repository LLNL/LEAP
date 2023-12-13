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

float FBPscalar(parameters*);
float* setViewWeights(parameters*);
float* setParkerWeights(parameters*);
float* setRedundantAndNonEquispacedViewWeights(parameters*, float* w = NULL);
float* setInverseConeWeight(parameters*);
float* setPreRampFilterWeights(parameters*);

float* setOffsetScanWeights(parameters*);

bool applyPreRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);

bool applyPreRampFilterWeights_CPU(float* g, parameters*);
bool applyPostRampFilterWeights_CPU(float* g, parameters*);
bool applyPreRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);

bool convertARTtoERT(float* g, parameters*, bool cpu_to_gpu, bool doInverse=false);
bool convertARTtoERT_CPU(float* g, parameters*, bool doInverse = false);

#endif
