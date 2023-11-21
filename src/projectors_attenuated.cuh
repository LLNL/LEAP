////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_ATTENUATED_H
#define __PROJECTORS_ATTENUATED_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_attenuated(float*& g, float* f, parameters* params, bool cpu_to_gpu);
bool backproject_attenuated(float* g, float*& f, parameters* params, bool cpu_to_gpu);

#endif
