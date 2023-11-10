////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for cpu projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SYMMETRIC_CPU_H
#define __PROJECTORS_SYMMETRIC_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool CPUproject_symmetric(float* g, float* f, parameters* params);
bool CPUbackproject_symmetric(float* g, float* f, parameters* params);

bool CPUproject_AbelParallel(float*, float*, parameters*);
bool CPUbackproject_AbelParallel(float*, float*, parameters*);

bool CPUproject_AbelCone(float*, float*, parameters*);
bool CPUbackproject_AbelCone(float*, float*, parameters*);

#endif