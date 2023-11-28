////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for some CPU-based computations
////////////////////////////////////////////////////////////////////////////////

#ifndef __CPU_UTILS_H
#define __CPU_UTILS_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

float tex3D(float* f, int, int, int, parameters* params);
float* reorder_ZYX_to_XYZ(float* f, parameters* params, int sliceStart, int sliceEnd);
//bool transpose_ZYX_to_XYZ(float* f, parameters* params);

#endif
