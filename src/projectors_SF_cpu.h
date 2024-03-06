////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for the primary CPU projectors models in LEAP
////////////////////////////////////////////////////////////////////////////////
#ifndef __PROJECTORS_SF_CPU_H
#define __PROJECTORS_SF_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of OpenMP accelerated CPU-based (modified) Separable Footprint forward and backprojection of
 * parallel-, fan-, and cone-beam geometries where the voxel sizes are close to the default values.
 */

bool CPUproject_SF_fan(float*, float*, parameters*, bool setToZero = true);
bool CPUbackproject_SF_fan(float*, float*, parameters*, bool setToZero = true);
bool CPUproject_SF_fan_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);
bool CPUbackproject_SF_fan_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);

bool CPUproject_SF_parallel(float*, float*, parameters*, bool setToZero = true);
bool CPUbackproject_SF_parallel(float*, float*, parameters*, bool setToZero = true);

bool CPUproject_SF_cone(float*, float*, parameters*, bool setToZero = true);
bool CPUbackproject_SF_cone(float*, float*, parameters*, bool setToZero = true);

bool CPUproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);
bool CPUbackproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);

bool applyPolarWeight(float* g, parameters* params);
bool applyInversePolarWeight(float* g, parameters* params);

bool CPUproject_SF_ZYX(float* g, float* f, parameters* params);
bool CPUbackproject_SF_ZYX(float* g, float* f, parameters* params);

#endif
