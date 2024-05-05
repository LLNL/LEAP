////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based sinogram replacement (a MAR method) routines
////////////////////////////////////////////////////////////////////////////////

#ifndef __SINOGRAM_REPLACEMENT_H
#define __SINOGRAM_REPLACEMENT_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool sinogramReplacement(float* X, float* X_prior, float* metalTrace, parameters* params, int* windowSize);

#endif
