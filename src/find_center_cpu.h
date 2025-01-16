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

float findCenter_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);

float findCenter_parallel_cpu(float* g, parameters* params, int iRow = -1, float* searchBounds = NULL);
float findCenter_fan_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);
float findCenter_cone_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);

float estimateTilt(float* g, parameters* params);
bool getConjugateDifference(float* g, parameters* params, float alpha, float centerCol, float* diff);
bool getConjugateProjections(float* g, parameters* params, float*& proj_A, float*& proj_B);
float interpolate2D(float*, float ind_1, float ind_2, int N_1, int N_2);

bool setDefaultRange_centerCol(parameters* params, int& centerCol_low, int& centerCol_high);
float findMinimum(double* costVec, int startInd, int endInd, float& minValue);

float* get_rotated_sinogram(float* g, parameters* params, int iRow);

#endif
