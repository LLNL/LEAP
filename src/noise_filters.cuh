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

/**
 * This header and associated source file provide implementions of some CUDA-based denoising algorithms, such as low-pass filter (blurFilter)
 * and a thresholded median filter.  It also implements a low-pass filter along the azithmuthal angle coordinate of a reconstruction volume
 * which is useful for some spare- or few-view reconstruction problems.
 */

class parameters;

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);
bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool medianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, bool data_on_cpu, int whichGPU = 0);

bool azimuthalBlur(float* f, parameters* params, float filterWidth, bool data_on_cpu, float* f_out = NULL);

#endif
