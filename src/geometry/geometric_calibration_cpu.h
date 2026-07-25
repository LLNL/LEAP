////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based geometric calibration routines
////////////////////////////////////////////////////////////////////////////////

#ifndef __GEOMETRIC_CALIBRATION_CPU_H
#define __GEOMETRIC_CALIBRATION_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide a CPU-based (accelerated by OpenMP) implemention of an algorithm to
 * find the "centerCol", "tau", and "tiltAngle" parameters of cone-beam data by using an inconsistency reconstruction.
 */

bool backprojection_sweep_cpu(float* g, parameters*, float* shifts, int numShifts, float* tilts, int numTilts, int which_param, float* costValues);

#endif
