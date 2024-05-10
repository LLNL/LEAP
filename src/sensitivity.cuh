////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for gpu-based sensitivity calculation (P*1)
////////////////////////////////////////////////////////////////////////////////

#ifndef __SENSITIVITY_H
#define __SENSITIVITY_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide CUDA-based implementations of functions to calculate
 * the volumetric sensitivites of a given CT geometry.  In otherwords, calculates of the backprojection of data where all
 * of the elements are equal to one.  Such a calculation are required for SART, MLEM, and OSEM algorithms.
 * One could simply just use the backproject command, but knowning that all projection data elements are equal to one enables
 * faster calculation.
 */

bool sensitivity_gpu(float*& f, parameters* params, bool data_on_cpu);
bool sensitivity_modular_gpu(float*& f, parameters* params, bool data_on_cpu);

#endif
