////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAMP_FILTER_H
#define __RAMP_FILTER_H

#ifdef WIN32
#pragma once
#endif

#define INCLUDE_CUFFT

#ifdef INCLUDE_CUFFT
#include <cufft.h>
class parameters;
cufftComplex* HilbertTransformFrequencyResponse(int N, parameters* params, float scalar = 1.0, float sampleShift = 0.0);
#endif

#include "parameters.h"

//bool rampFilter(float* f, int N_z, int N_y, int N_x, int smoothingLevel, int whichGPU = 0);
//bool rampFilter(float* f, int N_z, int N_y, int N_x, bool smoothFilter = false, int whichGPU = 0);

bool conv1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar = 1.0, int which = 0, float sampleShift = 0.0);
bool Hilbert1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar = 1.0, float sampleShift = 0.0);
bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar = 1.0);
bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu);

bool rampFilter1D_symmetric(float*& g, parameters* params, float scalar = 1.0);

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params);

bool parallelRay_derivative(float*& g, parameters* params, bool cpu_to_gpu);

#endif
