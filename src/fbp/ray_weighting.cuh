////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda header for ray weighting
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAY_WEIGHTING_CUH
#define __RAY_WEIGHTING_CUH

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"
#include <stdlib.h>

/**
 * This header and associated source file provide CUDA-based implementations of functions to calculate
 * the ray weighting steps of various FBP algorithms.
 * These functions just apply to weights using GPUs.  The calculation of the weight is in ray_weighting_cpu.cpp
 */

 /**
 * \fn          applyPreRampFilterWeights_GPU
 * \brief       Applies all weights before application of the ramp filter which includes the weights calculated by setPreRampFilterWeights and setViewWeights
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true, the data is on the CPU, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPreRampFilterWeights_GPU(float* g, parameters*, bool data_on_cpu);

/**
 * \fn          applyPostRampFilterWeights_GPU
 * \brief       Applies all weights after application of the ramp filter which includs the weights calculated by setParkerWeights and setInverseConeWeight
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true, the data is on the CPU, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPostRampFilterWeights_GPU(float* g, parameters*, bool data_on_cpu);

/**
 * \fn          applyDBPviewWeights_GPU
 * \brief       Applies the DBP reconstruction weights (as set by setDBPviewWeights)
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true, the data is on the CPU, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyDBPviewWeights_GPU(float* g, parameters*, bool data_on_cpu);

/**
 * \fn          convertARTtoERT
 * \brief       Applies weights to convert between the Attenuated Radon Transform and the Exponential Radon Transform
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true, the data is on the CPU, if false it is on a gpu
 * \param[in]   doInverse: if true converts ART to ERT, if false converts ERT to ART
 * \return      true if the operation was successfull, false otherwise
 */
bool convertARTtoERT(float* g, parameters*, bool data_on_cpu, bool doInverse=false);

/**
 * \fn          setViewDependentPolarWeights_gpu
 * \brief       Calculates sqrt(1+ v^2) for modular-beam data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true, the data is on the CPU, if false it is on a gpu
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
bool applyViewDependentPolarWeights_gpu(float* g, parameters* params, float* w, bool data_on_cpu, bool doInverse);

#endif
