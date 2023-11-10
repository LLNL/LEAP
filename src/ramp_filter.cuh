#ifndef __RAMP_FILTER_H
#define __RAMP_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

//bool rampFilter(float* f, int N_z, int N_y, int N_x, int smoothingLevel, int whichGPU = 0);
//bool rampFilter(float* f, int N_z, int N_y, int N_x, bool smoothFilter = false, int whichGPU = 0);

bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar = 1.0);
bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu);

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params);

#endif
