////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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

//#include "device_launch_parameters.h"

/**
 * This header and associated source file provide implementions of some CUDA-based denoising algorithms, such as low-pass filter (blurFilter)
 * and a thresholded median filter.  It also implements a low-pass filter along the azithmuthal angle coordinate of a reconstruction volume
 * which is useful for some spare- or few-view reconstruction problems.
 */

class parameters;

void setConstantMemoryParameters(const bool doHighPass);

bool lowOrHighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);
bool lowOrHighPassFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);
bool blurFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool highPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);
bool highPassFilter_txt(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int axis, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool momentFilter(float* f, int N_1, int N_2, int N_3, int r, int order, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1, float* f_out = NULL);

bool medianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu, int whichGPU = 0);

bool badPixelCorrection_gpu(float* g, parameters* params, float* badPixelMap, int w, bool data_on_cpu);

bool azimuthalBlur(float* f, parameters* params, float filterWidth, bool data_on_cpu, float* f_out = NULL);

#endif
