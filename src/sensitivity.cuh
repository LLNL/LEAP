////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __SENSITIVITY_H
#define __SENSITIVITY_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool sensitivity_gpu(float*& f, parameters* params, bool cpu_to_gpu);

#endif
