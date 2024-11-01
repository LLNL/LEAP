////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module header for projectors with voxel sizes that are
// much smaller or much larger than the nominal sizes
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_EXTENDEDSF_H
#define __PROJECTORS_EXTENDEDSF_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of CUDA-based (modified) Separable Footprint forward and backprojection of
 * parallel-, fan-, and cone-beam geometries where the voxel sizes are significantly larger or smaller than the default values.
 */

bool project_eSF(float*&, float*, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);
bool backproject_eSF(float*, float*&, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

bool project_eSF(float*&, float*, parameters*, bool data_on_cpu);
bool backproject_eSF(float*, float*&, parameters*, bool data_on_cpu);


#endif
