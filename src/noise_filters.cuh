////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////
#ifndef __NOISE_FILTERS_H
#define __NOISE_FILTERS_H

#ifdef WIN32
#pragma once
#endif

#include "device_launch_parameters.h"

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, bool cpu_to_gpu, int whichGPU = 0);
bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1);

#endif
