////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// GPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#ifndef __RESAMPLE_H
#define __RESAMPLE_H

#ifdef WIN32
#pragma once
#endif

bool downSample(float* I, int* N, float* I_dn, int* N_dn, float* factors, int whichGPU);
bool upSample(float* I, int* N, float* I_up, int* N_up, float* factors, int whichGPU);

#endif
