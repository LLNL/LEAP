////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// projectors class header which selects the appropriate projector to use based on the
// CT geometry and CT volume specification, and some other parameters such as
// whether the calculation should happen on the CPU or GPU
////////////////////////////////////////////////////////////////////////////////
#ifndef __PROJECTORS_H
#define __PROJECTORS_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include "parameters.h"

/**
 *  projectors class
 * This class is responsible for all the logic required for CPU- and GPU-based forward projection and backprojection algorithms.
 * Based on whether one wishes to run the computation and the geometry, this class dispatches the correct algorithm.
 * This class assumes that one has enough memory (CPU or GPU) memory to perform the given operation.  Splitting data into chunks
 * that fit into GPU memory is done by the tomographicModels class.
 * This class also only performs the calculations on a single GPU.  Again, the tomographicModels is responsible for break up jobs
 * from multiple GPUs.
 * If params->whichGPU < 0, then the computations will be carried out on the CPU, otherwise it is assumed that params->whichGPU
 * specifies the GPU index to perform the computations.  And if the data_on_cpu function argument is false, it is assumed that the
 * projection data  is on the GPU specified by the params->whichGPU index.  If data_on_cpu is true and params->whichGPU >= 0,
 * then the data is copied to the specified GPU, the computation is performed, and then the result is copied back to the CPU.  All temporary
 * GPU memory is freed.
 */

class projectors
{
public:
    // Constructor and destructor; these do nothing
    projectors();
    ~projectors();

    /**
     * \fn          project
     * \brief       Performs a forward projection using a modified Separable Footprint method
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool project(float* g, float* f, parameters* params, bool data_on_cpu);

    /**
     * \fn          project
     * \brief       Performs a forward projection using a modified Separable Footprint method
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   volume_on_cpu, true if volume (f) is on the cpu, false if it is on the gpu
     * \param[in]   accumulate, true if this backprojection is supposed to increment on the volume, false otherwise
     * \return      true if operation  was sucessful, false otherwise
     */
    bool project(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

    /**
     * \fn          backproject
     * \brief       Performs a backprojection using a modified Separable Footprint method
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool backproject(float* g, float* f, parameters* params, bool data_on_cpu);

    /**
     * \fn          backproject
     * \brief       Performs a backprojection using a modified Separable Footprint method
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   volume_on_cpu, true if volume (f) is on the cpu, false if it is on the gpu
     * \param[in]   accumulate, true if this backprojection is supposed to increment on the volume, false otherwise
     * \return      true if operation  was sucessful, false otherwise
     */
    bool backproject(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

    /**
     * \fn          weightedBackproject
     * \brief       Performs a weighted backprojection using a modified Separable Footprint method.
                    A weighted backprojection is different than just a plain backprojection because uses extrapolation on the detector
                    and for fan-beam and helical cone-beam data has additional weighting terms.  See the LEAP technical manual for more information.
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool weightedBackproject(float* g, float* f, parameters* params, bool data_on_cpu);

    /**
     * \fn          weightedBackproject
     * \brief       Performs a weighted backprojection using a modified Separable Footprint method.
                    A weighted backprojection is different than just a plain backprojection because uses extrapolation on the detector
                    and for fan-beam and helical cone-beam data has additional weighting terms.  See the LEAP technical manual for more information.
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   volume_on_cpu, true if volume (f) is on the cpu, false if it is on the gpu
     * \param[in]   accumulate, true if this backprojection is supposed to increment on the volume, false otherwise
     * \return      true if operation  was sucessful, false otherwise
     */
    bool weightedBackproject(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);
};

#endif
