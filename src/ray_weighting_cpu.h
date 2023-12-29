////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for ray weighting
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAY_WEIGHTING_CPU_H
#define __RAY_WEIGHTING_CPU_H

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

bool applyPreRampFilterWeights_CPU(float* g, parameters*);
bool applyPostRampFilterWeights_CPU(float* g, parameters*);

bool convertARTtoERT_CPU(float* g, parameters*, bool doInverse = false);

float* setViewDependentPolarWeights(parameters*);

#endif
