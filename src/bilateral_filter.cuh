#ifndef __BILATERAL_FILTER_H
#define __BILATERAL_FILTER_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

bool bilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, bool data_on_cpu, int whichGPU = 0);
bool scaledBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu, int whichGPU = 0);

#endif
