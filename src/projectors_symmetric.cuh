////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SYMMETRIC_H
#define __PROJECTORS_SYMMETRIC_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_symmetric(float*& g, float* f, parameters* params, bool cpu_to_gpu);
bool backproject_symmetric(float* g, float*& f, parameters* params, bool cpu_to_gpu);

bool inverse_symmetric(float* g, float*& f, parameters* params, bool cpu_to_gpu);

/* Utility Functions for anti-symmetric projections
float* splitVolume(float*, parameters* params, bool rightHalf = true);
float* splitProjection(float*, parameters* params, bool rightHalf = true);
bool mergeSplitVolume(float*, float*, parameters* params, bool rightHalf = true);
bool mergeSplitProjection(float*, float*, parameters* params, bool rightHalf = true);
//*/

#endif
