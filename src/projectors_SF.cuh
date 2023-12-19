////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for the primary projectors models in LEAP
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SF_H
#define __PROJECTORS_SF_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_SF(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_SF(float*, float*&, parameters*, bool data_on_cpu);

bool project_SF_fan(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_SF_fan(float*, float*&, parameters*, bool data_on_cpu);

bool project_SF_cone(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_SF_cone(float*, float*&, parameters*, bool data_on_cpu);

bool project_SF_parallel(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_SF_parallel(float*, float*&, parameters*, bool data_on_cpu);

#endif
