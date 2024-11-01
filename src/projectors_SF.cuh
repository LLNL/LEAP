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

/**
 * This header and associated source file provide implementions of CUDA-based (modified) Separable Footprint forward and backprojection of
 * parallel-, fan-, and cone-beam geometries where the voxel sizes are close to the default values.
 */

bool project_SF(float*&, float*, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);
bool backproject_SF(float*, float*&, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

bool project_SF(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_SF(float*, float*&, parameters*, bool data_on_cpu);


#endif
