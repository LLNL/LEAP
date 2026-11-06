////////////////////////////////////////////////////////////////////////////////
// Copyright 2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for guided filter algorithms
////////////////////////////////////////////////////////////////////////////////
#ifndef __GUIDED_FILTER_H
#define __GUIDED_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

/**
 * This header and associated source file are provide CUDA-based implementations of the Guided Filter.
 * The guided filter is an edge-preserving filter similar to the bilateral filter,
 * but does not suffer from gradient reversal artifacts.
 * This algorithm is described in the following paper:
 * He, Kaiming, Jian Sun, and Xiaoou Tang.
 * "Guided image filtering."
 * IEEE transactions on pattern analysis and machine intelligence 35, no. 6 (2012): 1397-1409.
 */

/**
 * \fn          guidedFilter
 * \brief       Performs guided filter denoising
 * \param[in]   f: pointer to 3D data to denoise
 * \param[in]   N_1: number of elements in the 1st dimension (numZ)
 * \param[in]   N_2: number of elements in the 2nd dimension (numY)
 * \param[in]   N_3: number of elements in the 3rd dimension (numX)
 * \param[in]   r: window radius
 * \param[in]   epsilon: denoising strength
 * \param[in]   numIter: number of iterations to perform
 * \param[in]   data_on_cpu: specifies whether the data is on the CPU (true) or the GPU (false)
 * \param[in]   whichGPU: which GPU to perform to computation
 * \param[in]   sliceStart: the first slice to perform the computation
 * \param[in]   sliceEnd: the last slice to perform the computation
 * \return      true if operation was sucessful, false otherwise
 */
bool guidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu, int whichGPU = 0, int sliceStart = -1, int sliceEnd = -1);

#endif
