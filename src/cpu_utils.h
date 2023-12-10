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

float* getSlice(float*, int, parameters*);
float* getProjection(float*, int, parameters*);

float tex3D(float* f, int, int, int, parameters* params);
float tex3D(float* f, float iz, float iy, float ix, parameters* params);
float tex3D_rev(float* f, float ix, float iy, float iz, parameters* params);
float* reorder_ZYX_to_XYZ(float* f, parameters* params, int sliceStart, int sliceEnd);
//bool transpose_ZYX_to_XYZ(float* f, parameters* params);

float innerProduct_cpu(float*, float*, int N_1, int N_2, int N_3);
bool equal_cpu(float*, float*, int N_1, int N_2, int N_3);
bool scalarAdd_cpu(float*, float, float*, int N_1, int N_2, int N_3);

#endif
