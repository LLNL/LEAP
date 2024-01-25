////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based routines to find the center detector pixel
////////////////////////////////////////////////////////////////////////////////

#ifndef __FIND_CENTER_CPU_H
#define __FIND_CENTER_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool findCenter_cpu(float* g, parameters* params, int iRow = -1);

bool findCenter_parallel_cpu(float* g, parameters* params, int iRow = -1);
bool findCenter_fan_cpu(float* g, parameters* params, int iRow = -1);
bool findCenter_cone_cpu(float* g, parameters* params, int iRow = -1);

bool setDefaultRange_centerCol(int numCols, int& centerCol_low, int& centerCol_high);
float findMinimum(double* costVec, int startInd, int endInd, bool findOnlyLocalMin = true);

#endif