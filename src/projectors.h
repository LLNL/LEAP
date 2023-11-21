#ifndef __PROJECTORS_H
#define __PROJECTORS_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"

class projectors
{
public:
    projectors();
    ~projectors();

    bool project(float* g, float* f, parameters* params, bool cpu_to_gpu);
    bool backproject(float* g, float* f, parameters* params, bool cpu_to_gpu);

    bool weightedBackproject(float* g, float* f, parameters* params, bool cpu_to_gpu);
};

#endif
