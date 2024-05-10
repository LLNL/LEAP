////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#ifndef __RESAMPLE_CPU_H
#define __RESAMPLE_CPU_H

#ifdef WIN32
#pragma once
#endif

bool downSample_cpu(float* I, int* N, float* I_dn, int* N_dn, float* factors);
bool upSample_cpu(float* I, int* N, float* I_up, int* N_up, float* factors);

float* bumpFcn(float W, float delay, int& N_taps);

#endif
