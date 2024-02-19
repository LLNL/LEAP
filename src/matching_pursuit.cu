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
#include "matching_pursuit.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.h"

#include <iostream>
#include <vector>



bool matchingPursuit(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int num1, int num2, int num3, float epsilon, int sparsityThreshold, bool data_on_cpu, int whichGPU)
{
    return false;
}
