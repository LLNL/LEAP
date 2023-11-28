////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for Joseph CPU projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_JOSEPH_CPU_H
#define __PROJECTORS_JOSEPH_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_Joseph_cpu(float*, float*, parameters*);
bool backproject_Joseph_cpu(float*, float*, parameters*);

bool project_Joseph_modular_cpu(float*, float*, parameters*);
bool backproject_Joseph_modular_cpu(float*, float*, parameters*);

float projectLine_Joseph(float* f, parameters* params, float* pos, float* traj);

#endif
