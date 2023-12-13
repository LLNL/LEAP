////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// C++ header for CPU Siddon projector (deprecated)
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SIDDON_CPU_H
#define __PROJECTORS_SIDDON_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool CPUproject_cone(float*, float*, parameters*);
bool CPUbackproject_cone(float*, float*, parameters*);

bool CPUproject_parallel(float*, float*, parameters*);
bool CPUbackproject_parallel(float*, float*, parameters*);

bool CPUproject_fan(float*, float*, parameters*);
bool CPUbackproject_fan(float*, float*, parameters*);

bool CPUproject_modular(float*, float*, parameters*);
bool CPUbackproject_modular(float*, float*, parameters*);

float projectLine(float* f, parameters* params, float* pos, float* traj);

#endif
