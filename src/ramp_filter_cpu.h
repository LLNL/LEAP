////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for CPU-based ramp and Hilbert filters
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAMP_FILTER_CPU_H
#define __RAMP_FILTER_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"
#include <complex>
typedef std::complex<double> Complex;

/**
 * This header and associated source file provide CPU-based implementations (accelerated by OpenMP) of functions to apply
 * Hilbert and ramp filters to the projection data.
 */

bool conv1D_cpu(float*& g, parameters* params, float scalar = 1.0, int whichFilter = 0);
bool Hilbert1D_cpu(float*& g, parameters* params, float scalar = 1.0);
bool rampFilter1D_cpu(float*& g, parameters* params, float scalar = 1.0);
float* rampFilterFrequencyResponseMagnitude_cpu(int N, parameters* params);

float* rampFrequencyResponse2D(int N, double T, double scalingFactor, int smoothingLevel);
float* rampFrequencyResponse(int N, double T);
double rampFrequencyResponse(double X, double T);
double frequencySamples(int i, int N, double T);
double timeSamples(int i, int N);
double rampImpulseResponse(int N, double T, int n, int rampID);
double* rampImpulseResponse(int N, double T, parameters* params);

double rampImpulseResponse_bandLimited(int N, double T, int i, float mu);

// Hilbert Transform
double* HilbertTransformImpulseResponse(int N, int whichDirection = 1);
Complex* HilbertTransformFrequencyResponse_cpu(int N, parameters* params);

bool splitLeftAndRight(float* g, float* g_left, float* g_right, parameters* params);
bool mergeLeftAndRight(float* g, float* g_left, float* g_right, parameters* params);

bool Laplacian_cpu(float*& g, int numDims, bool smooth, parameters* params, float scalar = 1.0);

bool ray_derivative_cpu(float* g, parameters* params, float sampleShift = 0.0, float scalar = 1.0);

int zeroPadForOffsetScan_numberOfColsToAdd(parameters* params);
int zeroPadForOffsetScan_numberOfColsToAdd(parameters* params, bool& padOnLeft);
float* zeroPadForOffsetScan(float* g, parameters* params, float* g_out = NULL);

void fftshift(float* h, int N);

#endif
