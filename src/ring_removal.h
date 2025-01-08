////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for ring removal algorithms
////////////////////////////////////////////////////////////////////////////////
#ifndef __RING_REMOVAL_H
#define __RING_REMOVAL_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

class ringRemoval
{
public:
    ringRemoval();
    ~ringRemoval();
    bool execute(float* g, int N_1, int N_2, int N_3, float delta = 0.01, float beta = 1.0e3, int numIter = 30, float maxChange = 0.05);
private:
    float h1(float);
    float h2(float);
    float delta;
    float beta;
};

#endif
