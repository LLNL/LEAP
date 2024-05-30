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

float consistencyCost(float*, parameters*, bool data_on_cpu, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt);

#endif
