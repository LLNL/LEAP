#pragma once

#include "parameters.h"

//bool rampFilter(float* f, int N_z, int N_y, int N_x, int smoothingLevel, int whichGPU = 0);
//bool rampFilter(float* f, int N_z, int N_y, int N_x, bool smoothFilter = false, int whichGPU = 0);

bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar = 1.0);
bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu);

float* rampFrequencyResponse2D(int N, double T, double scalingFactor, int smoothingLevel);
float* rampFrequencyResponse(int N, double T);
double rampFrequencyResponse(double X, double T);
double frequencySamples(int i, int N, double T);

double timeSamples(int i, int N);
double rampImpulseResponse(int N, double T, int n, int rampID);
double* rampImpulseResponse(int N, double T = 1.0, int rampID = 2);
float* rampFilterFrequencyResponseMagnitude(int N, parameters* params);
