////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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

/**
 * This header and associated source file provide CPU-based implementations (accelerated by OpenMP) of functions to calculate
 * the volumetric sensitivites of a given CT geometry.  In otherwords, calculates of the backprojection of data where all
 * of the elements are equal to one.  Such a calculation are required for SART, MLEM, and OSEM algorithms.
 * One could simply just use the backproject command, but knowning that all projection data elements are equal to one enables
 * faster calculation.
 */

bool sensitivity_CPU(float*&, parameters*);
bool sensitivity_cone_CPU(float*&, parameters*);
//bool sensitivity_coneparallel_CPU(float*&, parameters*);
bool sensitivity_fan_CPU(float*&, parameters*);
bool sensitivity_parallel_CPU(float*&, parameters*);
bool sensitivity_modular_CPU(float*& s, parameters* params);

#endif
