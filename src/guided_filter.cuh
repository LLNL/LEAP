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

bool guidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu, int whichGPU = 0);

#endif
