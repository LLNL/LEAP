////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for some CPU-based computations
////////////////////////////////////////////////////////////////////////////////

#ifndef __CPU_UTILS_H
#define __CPU_UTILS_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file are for generic CPU-based functions that are used in LEAP
 */

int optimalFFTsize(int N);

//double getAvailableGBofMemory();
//size_t getPhysicalMemorySize();

float* getSlice(float*, int, parameters*);
float* getProjection(float*, int, parameters*);

float tex3D(float* f, int, int, int, parameters* params);
float tex3D(float* f, float iz, float iy, float ix, parameters* params);
float tex3D_rev(float* f, float ix, float iy, float iz, parameters* params);
float* reorder_ZYX_to_XYZ(float* f, parameters* params, int sliceStart, int sliceEnd);
//bool transpose_ZYX_to_XYZ(float* f, parameters* params);

float innerProduct_cpu(float*, float*, int N_1, int N_2, int N_3);
bool equal_cpu(float*, float*, int N_1, int N_2, int N_3);
bool scale_cpu(float*, float, int N_1, int N_2, int N_3);
bool sub_cpu(float*, float*, int N_1, int N_2, int N_3);
bool scalarAdd_cpu(float*, float, float*, int N_1, int N_2, int N_3);
bool clip_cpu(float*, int N_1, int N_2, int N_3, float clipVal = 0.0);
bool replaceZeros_cpu(float*, int N_1, int N_2, int N_3, float newVal = 1.0);
float sum_cpu(float*, int N_1, int N_2, int N_3);

bool windowFOV_cpu(float* f, parameters* params);

float* rotateAroundAxis(float* theAxis, float phi, float* aVec);

char swapEndian(char);
short swapEndian(short);
unsigned short swapEndian(unsigned short);
int swapEndian(int);
float swapEndian(float);
double swapEndian(double);
unsigned int swapEndian(unsigned int);

template <typename T>
T bswap(T val);

#include <chrono>
#include <assert.h>

using namespace std::chrono_literals;

template <class DT = std::chrono::milliseconds,
    class ClockT = std::chrono::steady_clock>
    class Timer
{
    using timep_t = decltype(ClockT::now());

    timep_t _start = ClockT::now();
    timep_t _end = {};

public:
    void tick() {
        _end = timep_t{};
        _start = ClockT::now();
    }

    void tock() {
        _end = ClockT::now();
    }

    template <class duration_t = DT>
    auto duration() const {
        // Use gsl_Expects if your project supports it.
        assert(_end != timep_t{} && "Timer must toc before reading the time");
        return std::chrono::duration_cast<duration_t>(_end - _start);
    }
};

#endif
