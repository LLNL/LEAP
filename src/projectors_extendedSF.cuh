////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_EXTENDEDSF_H
#define __PROJECTORS_EXTENDEDSF_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_eSF(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_eSF(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_eSF_fan(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_eSF_fan(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_eSF_cone(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_eSF_cone(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_eSF_parallel(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_eSF_parallel(float*, float*&, parameters*, bool cpu_to_gpu);

#endif
