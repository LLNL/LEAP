////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// GPU-based geometric calibration routines
////////////////////////////////////////////////////////////////////////////////

#ifndef __GEOMETRIC_CALIBRATION_H
#define __GEOMETRIC_CALIBRATION_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This function implements a GPU-based evaluation of a consistency condition what can be used to perform
 * geometric calibration from the radiographs of any scanner object.  The algorithm comes from the following paper:
 * Lesaint, Jerome, Simon Rit, Rolf Clackdoyle, and Laurent Desbat.
 * Calibration for circular cone-beam CT based on consistency conditions.
 * IEEE Transactions on Radiation and Plasma Medical Sciences 1, no. 6 (2017): 517-526.
 */

float consistencyCost(float*, parameters*, bool data_on_cpu, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt);

#endif
