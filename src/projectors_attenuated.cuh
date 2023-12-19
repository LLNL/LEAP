////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for projectors of the Attenuated Radon Transform
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_ATTENUATED_H
#define __PROJECTORS_ATTENUATED_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_attenuated(float*& g, float* f, parameters* params, bool data_on_cpu);
bool backproject_attenuated(float* g, float*& f, parameters* params, bool data_on_cpu);

#endif
