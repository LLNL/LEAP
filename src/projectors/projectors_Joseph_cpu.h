////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for Joseph CPU projector
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_JOSEPH_CPU_H
#define __PROJECTORS_JOSEPH_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

#ifdef __USE_CPU
struct float3
{
	float x;
	float y;
	float z;
};

float3 make_float3(float, float, float);

struct double3
{
	double x;
	double y;
	double z;
};

double3 make_double3(double, double, double);

#else
	#include "cuda_runtime.h"
#endif

bool project_Joseph_cpu(float*, float*, parameters*);
bool backproject_Joseph_cpu(float*, float*, parameters*);

bool project_Joseph_modular_cpu(float*, float*, parameters*);
bool backproject_Joseph_modular_cpu(float*, float*, parameters*);

float projectLine_Joseph(float* f, parameters* params, float3 pos, float3 traj);
float backproject_Joseph_modular_kernel(float* aProj, parameters* params, float* sourcePosition, float* moduleCenter, float* u_vec, float* v_vec, float x, float y, float z);
float backproject_Joseph_modular_parallel_kernel(float* aProj, parameters* params, float* sourcePosition, float* moduleCenter, float* u_vec, float* v_vec, float x, float y, float z);

bool usingSFprojectorsForModularBeam(parameters* params);

#endif
