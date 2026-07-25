////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for CPU cylindrically symmetric projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SYMMETRIC_CPU_H
#define __PROJECTORS_SYMMETRIC_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of CPU-based (accelerated by OpenMP) forward and backprojection of
 * parallel- and cone-beam geometries where the reconstruction is assumed to be cylindrically-symmetric.
 */

bool CPUinverse_symmetric(float* g, float* f, parameters* params);

bool CPUproject_symmetric(float* g, float* f, parameters* params);
bool CPUbackproject_symmetric(float* g, float* f, parameters* params);

bool CPUproject_AbelParallel(float*, float*, parameters*);
bool CPUbackproject_AbelParallel(float*, float*, parameters*);

bool CPUproject_AbelCone(float*, float*, parameters*);
bool CPUbackproject_AbelCone(float*, float*, parameters*);

bool reorder_ZY_to_YZ(float* f, parameters* params);
bool reorder_YZ_to_ZY(float* f, parameters* params);

float* reorder_ZY_to_YZ_copy(float* f, parameters* params);

#endif
