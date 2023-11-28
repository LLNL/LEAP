// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for SF cpu projector
////////////////////////////////////////////////////////////////////////////////
#ifndef __PROJECTORS_SF_CPU_H
#define __PROJECTORS_SF_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

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

#endif
