////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CUDA-based thresholded median filter and 3D low pass filter calculations
////////////////////////////////////////////////////////////////////////////////
#ifndef __NOISE_FILTERS_H
#define __NOISE_FILTERS_H

#ifdef WIN32
#pragma once
#endif

#include "device_launch_parameters.h"

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);
bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

#endif
