////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for the voxel-driven backprojection
////////////////////////////////////////////////////////////////////////////////

#ifndef __BACKPROJECTORS_VD_H
#define __BACKPROJECTORS_VD_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of CUDA-based voxel-driven backprojection of
 * parallel-, fan-, and cone-beam geometries.
 */

bool backproject_VD(float*, float*&, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);
bool backproject_VD_modular(float*, float*&, parameters*, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

bool backproject_VD(float*, float*&, parameters*, bool data_on_cpu);
bool backproject_VD_modular(float*, float*&, parameters*, bool data_on_cpu);

#endif
