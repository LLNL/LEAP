////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for cpu-based sensitivity calculation (P*1)
////////////////////////////////////////////////////////////////////////////////
#ifndef __SENSITIVITY_CPU_H
#define __SENSITIVITY_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool sensitivity_CPU(float*&, parameters*);
bool sensitivity_cone_CPU(float*&, parameters*);
bool sensitivity_fan_CPU(float*&, parameters*);
bool sensitivity_parallel_CPU(float*&, parameters*);
bool sensitivity_modular_CPU(float*& s, parameters* params);

#endif
