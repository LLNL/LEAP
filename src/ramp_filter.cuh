////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for GPU-based ramp and Hilbert filters
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAMP_FILTER_H
#define __RAMP_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"
#include "leap_defines.h"
//class parameters;

/**
 * This header and associated source file provide CUDA-based implementations of functions to apply
 * Hilbert and ramp filters to the projection data and 2D ramp filter applied to the z-slices of a volume.
 */

#ifdef __INCLUDE_CUFFT
#include <cufft.h>
cufftComplex* HilbertTransformFrequencyResponse(int N, parameters* params, float scalar = 1.0, float sampleShift = 0.0);
float* rampFilterFrequencyResponseMagnitude(int N, parameters* params);
#endif

float* rampImpulseResponse_modified(int N, parameters* params);

//bool rampFilter(float* f, int N_z, int N_y, int N_x, int smoothingLevel, int whichGPU = 0);
//bool rampFilter(float* f, int N_z, int N_y, int N_x, bool smoothFilter = false, int whichGPU = 0);

bool conv1D(float*& g, parameters* params, bool data_on_cpu, float scalar = 1.0, int which = 0, float sampleShift = 0.0);
bool Hilbert1D(float*& g, parameters* params, bool data_on_cpu, float scalar = 1.0, float sampleShift = 0.0);
bool rampFilter1D(float*& g, parameters* params, bool data_on_cpu, float scalar = 1.0);

bool rampFilter2D(float*& f, parameters* params, bool data_on_cpu);
bool rampFilter2D_XYZ(float*& f, parameters* params, bool data_on_cpu);

bool transmissionFilter_gpu(float*& g, parameters* params, bool data_on_cpu, float* H, int N_H1, int N_H2, bool isAttenuationData);

bool ray_derivative(float*& g, parameters* params, bool data_on_cpu, float scalar = 1.0, float sampleShift = 0.0);

bool Laplacian_gpu(float*& g, int numDims, bool smooth, parameters* params, bool data_on_cpu, float scalar = 1.0);

bool rampFilter1D_symmetric(float*& g, parameters* params, float scalar = 1.0);

bool parallelRay_derivative(float*& g, parameters* params, bool data_on_cpu);
bool parallelRay_derivative_chunk(float*& g, parameters* params, bool data_on_cpu);

float* zeroPadForOffsetScan_GPU(float* g, parameters* params, float* g_out = NULL, bool data_on_cpu = false);

#endif

