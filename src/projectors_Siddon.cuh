////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SIDDON_CUH
#define __PROJECTORS_SIDDON_CUH

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_Siddon(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_Siddon(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_cone(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_cone(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_fan(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_fan(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_parallel(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_parallel(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_modular(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_modular(float*, float*&, parameters*, bool cpu_to_gpu);

#endif
