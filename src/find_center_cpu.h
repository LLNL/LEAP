////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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

/**
 * This header and associated source file provide a CPU-based (accelerated by OpenMP) implemention of an algorithm to
 * find the "centerCol" parameter of parallel-, fan-, or cone-beam data.  It does not work with so-called offset scan
 * which is also known as a half-fan or half-cone.
 */

bool findCenter_cpu(float* g, parameters* params, int iRow = -1);

bool findCenter_parallel_cpu(float* g, parameters* params, int iRow = -1);
bool findCenter_fan_cpu(float* g, parameters* params, int iRow = -1);
bool findCenter_cone_cpu(float* g, parameters* params, int iRow = -1);

float estimateTilt(float* g, parameters* params, int iRow = -1);

bool setDefaultRange_centerCol(parameters* params, int& centerCol_low, int& centerCol_high);
float findMinimum(double* costVec, int startInd, int endInd, bool findOnlyLocalMin = true);

#endif
