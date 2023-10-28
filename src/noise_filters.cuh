#pragma once

#include "device_launch_parameters.h"

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, bool cpu_to_gpu, int whichGPU = 0);
bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu, int whichGPU = 0);
