#ifndef __RAMP_FILTER_CPU_H
#define __RAMP_FILTER_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool rampFilter1D_cpu(float*& g, parameters* params, float scalar = 1.0);
float* rampFilterFrequencyResponseMagnitude_cpu(int N, parameters* params);

float* rampFrequencyResponse2D(int N, double T, double scalingFactor, int smoothingLevel);
float* rampFrequencyResponse(int N, double T);
double rampFrequencyResponse(double X, double T);
double frequencySamples(int i, int N, double T);
double timeSamples(int i, int N);
double rampImpulseResponse(int N, double T, int n, int rampID);
double* rampImpulseResponse(int N, double T, int rampID);

#endif
