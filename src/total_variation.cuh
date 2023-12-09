////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////
#ifndef __TOTAL_VARIATION_H
#define __TOTAL_VARIATION_H

#ifdef WIN32
#pragma once
#endif

#include "device_launch_parameters.h"

// calculate anisotropic Total Variation cost with Huber loss function
float anisotropicTotalVariation_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1);

// calculate anisotropic Total Variation gradient with Huber loss function
bool anisotropicTotalVariation_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1);

// calculate anisotropic Total Variation quadratic form with Huber loss function
float anisotropicTotalVariation_quadraticForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1);

bool diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu, int whichGPU = 0);

#endif
