////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Class for performing analytic inversion, i.e., Filtered Backprojection (FBP)
// algorithms.
////////////////////////////////////////////////////////////////////////////////
#ifndef __FILTERED_BACKPROJECTION_H
#define __FILTERED_BACKPROJECTION_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "projectors.h"

/**
 *  filteredBackprojection class
 * This class is responsible for all the logic required for analytic reconstruction, e.g., Filtered Backprojection (FBP),
 * of a particular geometry.
 * This class assumes that one has enough memory (CPU or GPU) memory to perform the given operation.  Splitting data into chunks
 * that fit into GPU memory is done by the tomographicModels class.
 * This class also only performs the calculations on a single GPU.  Again, the tomographicModels is responsible for break up jobs
 * from multiple GPUs.
 * If params->whichGPU < 0, then the computations will be carried out on the CPU, otherwise it is assumed that params->whichGPU
 * specifies the GPU index to perform the computations.  And if the data_on_gpu function argument is true, it is assumed that the
 * data (projection and volume) is on the GPU specified by the params->whichGPU index.  If data_on_gpu is false and params->whichGPU >= 0,
 * then the data is copied to the specified GPU, the computation is performed, and then the result is copied back to the CPU.  All temporary
 * GPU memory is freed.
 */

class filteredBackprojection
{
public:
    // Constructor and destructor; these do nothing
    filteredBackprojection();
    ~filteredBackprojection();

    /**
     * \fn          HilbertFilterProjections
     * \brief       Performs a Hilbert filter along each detector row
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \param[in]   sampleShift, the delay on the Hilbert filter which can be -0.5, 0.0, or 0.5 (+/-0.5 delay results in the smoother filter)
     * \return      true if operation  was sucessful, false otherwise
     */
    bool HilbertFilterProjections(float* g, parameters* params, bool data_on_cpu, float scalar, float sampleShift = 0.0);

    /**
     * \fn          rampFilterProjections
     * \brief       Performs a ramp filter along each detector row
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \return      true if operation  was sucessful, false otherwise
     */
    bool rampFilterProjections(float* g, parameters* params, bool data_on_cpu, float scalar);

    /**
     * \fn          filterProjections
     * \brief       Performs all the necessary filtering and ray weighting needed to perform an FBP reconstruction.  FBP algorithms are achieve by this function followed by weighted backprojection
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \return      true if operation  was sucessful, false otherwise
     */
    bool filterProjections(float* g, float* g_out, parameters* ctParams, bool data_on_cpu);

    /**
     * \fn          preRampFiltering
     * \brief       Applies all pre ramp filter weighting
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \return      true if operation  was sucessful, false otherwise
     */
    bool preRampFiltering(float* g, parameters* ctParams, bool data_on_cpu);

    /**
     * \fn          postRampFiltering
     * \brief       Applies all post ramp filter weighting
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \return      true if operation  was sucessful, false otherwise
     */
    bool postRampFiltering(float* g, parameters* ctParams, bool data_on_cpu);

    /**
     * \fn          execute
     * \brief       Performs an FBP reconstruction
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool execute(float* g, float* f, parameters* params, bool data_on_cpu);

    /**
     * \fn          execute
     * \brief       Performs an FBP reconstruction
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   volume_on_cpu, true if volume (f) is on the cpu, false if it is on the gpu
     * \param[in]   accumulate, true if this backprojection is supposed to increment on the volume, false otherwise
     * \return      true if operation  was sucessful, false otherwise
     */
    bool execute(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate = false);

private:

    /**
     * \fn          convolve1D
     * \brief       Applies either a ramp or Hilbert filter along each detector row
     * \param[in]   g, pointer to the projection data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \param[in]   scalar, an additional scalar applied to the projection data
     * \param[in]   whichFilter, if 0 performs ramp filter, if 1 performs Hilbert filter
     * \param[in]   sampleShift, the delay on the Hilbert filter which can be -0.5, 0.0, or 0.5 (+/-0.5 delay results in the smoother filter)
     * \return      true if operation  was sucessful, false otherwise
     */
    bool convolve1D(float* g, parameters* params, bool data_on_cpu, float scalar, int whichFilter, float sampleShift = 0.0);

    /**
     * \fn          filterProjections
     * \brief       Performs all the necessary filtering and ray weighting needed to perform an FBP reconstruction of the Attenuated Radon Transform via Novikov's inversion formula.  FBP algorithms are achieve by this function followed by weighted backprojection
     * \param[in]   g, pointer to the projection data
     * \param[in]   ctParams, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool filterProjections_Novikov(float* g, parameters* ctParams, bool data_on_cpu);

    /**
     * \fn          execute_attenuated
     * \brief       Performs an FBP reconstruction of the Attenuated Radon Transform (ART) with costant attenuation on a cylinder (special case of ART)
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool execute_attenuated(float* g, float* f, parameters* params, bool data_on_cpu);

    /**
     * \fn          execute_Novikov
     * \brief       Performs an FBP reconstruction of the Attenuated Radon Transform (ART) with arbitrary attenuation map via Novikov's inversion formula0
     * \param[in]   g, pointer to the projection data
     * \param[in]   f, pointer to the volume data
     * \param[in]   params, pointer to the parameters object
     * \param[in]   data_on_cpu, true if data (g) is on the cpu, false if it is on the gpu
     * \return      true if operation  was sucessful, false otherwise
     */
    bool execute_Novikov(float* g, float* f, parameters* params, bool data_on_cpu);

    // Instance of the projectors class used for backprojection and weighted backproject
    projectors proj;
};

#endif
