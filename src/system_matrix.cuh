////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for calculating system matrix values
////////////////////////////////////////////////////////////////////////////////

#ifndef __SYSTEM_MATRIX_H
#define __SYSTEM_MATRIX_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of CUDA-based (modified) Separable Footprint calculation of
 * system matrix elements  for parallel-, fan-, and cone-beam geometries.
 */

bool systemMatrix_parallel(float*&, short*&, int N_max, parameters*, int iAngle, int* iCols, int numCols, bool data_on_cpu);
bool systemMatrix_cone(float*&, short*&, int N_max, parameters*, int iAngle, int* iRows, int numRows, int* iCols, int numCols, bool data_on_cpu);

#endif
