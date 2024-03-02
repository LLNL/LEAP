////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for matching pursuit algorithms which are a part of
// dictionary denoising.  This file is an adaptation of code written by myself
// (Kyle) several years ago in a package called "3Ddensoing"
////////////////////////////////////////////////////////////////////////////////
#ifndef __MATCHING_PURSUIT_H
#define __MATCHING_PURSUIT_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

#ifndef PI
#define PI 3.141592653589793
#endif

/*
float Xsamples(int, int);
float* setDictionary(int patchSize, int& numPatches);
float* setDictionaryElement(int patchSize, int featureDim, float3 shifts, float2 rotation);
//*/


double matchingPursuit_memory(int N_1, int N_2, int N_3, int numElements, int num1, int num2, int num3, int sparsityThreshold);

bool matchingPursuit(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int num1, int num2, int num3, float epsilon, int sparsityThreshold, bool data_on_cpu, int whichGPU = 0);
bool matchingPursuit_basis(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int num1, int num2, int num3, float epsilon, int sparsityThreshold, bool data_on_cpu, int whichGPU = 0);

bool calcInnerProductPairs(float* dictionary, int numElements, int num1, int num2, int num3, float* innerProductPairs);

#endif
