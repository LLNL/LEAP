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

/**
 * This header and associated source file provide implementions of CUDA-based forward and backprojection of the Attenuated Radon Transform (ART).
 * LEAP provides two methods to specify the attenuation map.  The first method assumes the attenuation is constant-valued on a cylinder and the second
 * method allows the users to specify a voxel-based attenuation map that is sampled on the same grid as the reconstruction volume.
 * Note that we only provide an implemented of the ART for parallel-beam geometry.
 */

bool project_attenuated(float*& g, float* f, parameters* params, bool data_on_cpu);
bool backproject_attenuated(float* g, float*& f, parameters* params, bool data_on_cpu);

#endif
