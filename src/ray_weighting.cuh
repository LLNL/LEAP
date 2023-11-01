#pragma once

#include "parameters.h"
#include <stdlib.h>

float FBPscalar(parameters*);
float* setViewWeights(parameters*);
float* setParkerWeights(parameters*);
float* setRedundantAndNonEquispacedViewWeights(parameters*, float* w = NULL);
float* setInverseConeWeight(parameters*);
float* setPreRampFilterWeights(parameters*);

bool applyPreRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights(float* g, parameters*, bool cpu_to_gpu);

bool applyPreRampFilterWeights_CPU(float* g, parameters*);
bool applyPostRampFilterWeights_CPU(float* g, parameters*);
bool applyPreRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);
bool applyPostRampFilterWeights_GPU(float* g, parameters*, bool cpu_to_gpu);

bool FBP_inplaceFiltering(float* g, float* f, parameters*, bool cpu_to_gpu);
