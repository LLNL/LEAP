#ifndef __FILTERED_BACKPROJECTION_H
#define __FILTERED_BACKPROJECTION_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "projectors.h"

class filteredBackprojection
{
public:
    filteredBackprojection();
    ~filteredBackprojection();

    bool HilbertFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar);
    bool rampFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar);
    bool filterProjections(float* g, parameters* ctParams, bool cpu_to_gpu);

    bool execute(float* g, float* f, parameters* params, bool cpu_to_gpu);

private:
    bool conv1D(float* g, parameters* params, bool cpu_to_gpu, float scalar, int whichFilter);
    bool filterProjections_Novikov(float* g, parameters* ctParams, bool cpu_to_gpu);

    bool execute_attenuated(float* g, float* f, parameters* params, bool cpu_to_gpu);
    bool execute_Novikov(float* g, float* f, parameters* params, bool cpu_to_gpu);

    projectors proj;
};

#endif
