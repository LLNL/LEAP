////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for anisotropic Total Variation (TV)
////////////////////////////////////////////////////////////////////////////////
#ifndef __TOTAL_VARIATION_H
#define __TOTAL_VARIATION_H

#ifdef WIN32
#pragma once
#endif

//#include "device_launch_parameters.h"

/**
 * This header and associated source file provide CUDA-based implementations of functions for Anisotropic Total Variation (TV) denoising.
 */

void setConstantMemoryParameters(const float delta, const float p);

// calculate anisotropic Total Variation cost with Huber loss function
float anisotropicTotalVariation_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, int numNeighbors = 26);

// calculate anisotropic Total Variation gradient with Huber loss function
bool anisotropicTotalVariation_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, int numNeighbors = 26, bool doMean = false);

// calculate anisotropic Total Variation quadratic form with Huber loss function
float anisotropicTotalVariation_quadraticForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, int numNeighbors = 26);

// runs a specified number of gradient descent iterations that minimize the TV cost functional
bool diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu, int whichGPU = 0, int numNeighbors = 26);

bool TVdenoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool data_on_cpu, int whichGPU = 0, int numNeighbors = 26, bool doMean = false);

#endif
