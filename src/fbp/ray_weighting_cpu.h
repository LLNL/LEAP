////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for ray weighting
////////////////////////////////////////////////////////////////////////////////
#ifndef __RAY_WEIGHTING_CPU_H
#define __RAY_WEIGHTING_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"
#include <stdlib.h>

/**
 * This header and associated source file provide CPU-based implementations (accelerated by OpenMP) of functions to calculate
 * the ray weighting steps of various FBP algorithms.
 */

 /**
 * \fn          applyPreRampFilterWeights
 * \brief       Applies all weights before application of the ramp filter which includes the weights calculated by setPreRampFilterWeights and setViewWeights
 *              this function calls either applyPreRampFilterWeights_CPU or applyPreRampFilterWeights_GPU
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true g is on the cpu, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPreRampFilterWeights(float* g, parameters* params, bool data_on_cpu);

/**
 * \fn          applyPostRampFilterWeights
 * \brief       Applies all weights after application of the ramp filter which includs the weights calculated by setParkerWeights and setInverseConeWeight
 *              this function calls either applyPostRampFilterWeights_CPU or applyPostRampFilterWeights_GPU
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true g is on the cpu, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPostRampFilterWeights(float* g, parameters* params, bool data_on_cpu);

/**
 * \fn          applyDBPviewWeights
 * \brief       Applies all weights after application of the derivative operation in DBP reconstructions
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   data_on_cpu: if true g is on the cpu, if false it is on a gpu
 * \return      true if the operation was successfull, false otherwise
 */
bool applyDBPviewWeights(float* g, parameters* params, bool data_on_cpu);

/**
 * \fn          FBPscalar
 * \brief       Calculates the scalar needed to perform quantitatively accurate FBP reconstruction
 * \param[in]   params: pointers to a parameters class object
 * \return      scalar for FBP
 */
float FBPscalar(parameters* params);

/**
 * \fn          setViewWeights
 * \brief       Calculates the FBP view weights (non-equi-spaced projections, projections measured multiple times)
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means no view weights are needed
 */
float* setViewWeights(parameters* params);

/**
 * \fn          setParkerWeights
 * \brief       Calculates the Parker Weights view weights (for scans less than 360 degrees)
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setParkerWeights(parameters* params);

/**
 * \fn          setViewWeights
 * \brief       Calculates the FBP view weights for non-equi-spaced projections and projections measured multiple times
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means no view weights are needed
 */
float* setRedundantAndNonEquispacedViewWeights(parameters* params, float* w = NULL);

/**
 * \fn          setInverseConeWeight
 * \brief       Calculates the inverse cone weight, 1 / sqrt(1 + u^2 + v^2)
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setInverseConeWeight(parameters* params);

/**
 * \fn          setPreRampFilterWeights
 * \brief       Calculates all weights before application of the ramp filter which includes the weights calculated by setPreRampFilterWeights and setViewWeights
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setPreRampFilterWeights(parameters* params);

/**
 * \fn          setOffsetScanWeights
 * \brief       Calculates the so-called Wang Weights for offset scan reconstruction
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setOffsetScanWeights(parameters* params);

/**
 * \fn          setDBPviewWeights
 * \brief       Calculates the DBP view weights, i.e., sign(sin(projection angle))
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setDBPviewWeights(parameters* params);

/**
 * \fn          applyPreRampFilterWeights_CPU
 * \brief       Applies all weights before application of the ramp filter which includes the weights calculated by setPreRampFilterWeights and setViewWeights
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPreRampFilterWeights_CPU(float* g, parameters* params);

/**
 * \fn          applyPostRampFilterWeights_CPU
 * \brief       Applies all weights after application of the ramp filter which includs the weights calculated by setParkerWeights and setInverseConeWeight
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \return      true if the operation was successfull, false otherwise
 */
bool applyPostRampFilterWeights_CPU(float* g, parameters* params);

/**
 * \fn          applyDBPviewWeights_CPU
 * \brief       Applies the DBP reconstruction weights (as set by setDBPviewWeights)
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \return      true if the operation was successfull, false otherwise
 */
bool applyDBPviewWeights_CPU(float* g, parameters* params);

/**
 * \fn          convertARTtoERT_CPU
 * \brief       Applies weights to convert between the Attenuated Radon Transform and the Exponential Radon Transform
 * \param[in]   g: pointer to the input 3D attenuation projection data
 * \param[in]   params: pointer to a parameters class object
 * \param[in]   doInverse: if true converts ART to ERT, if false converts ERT to ART
 * \return      true if the operation was successfull, false otherwise
 */
bool convertARTtoERT_CPU(float* g, parameters* params, bool doInverse = false);

/**
 * \fn          setViewDependentPolarWeights
 * \brief       Calculates sqrt(1+ v^2) for modular-beam data
 * \param[in]   params: pointer to a parameters class object
 * \return      pointer to the weights (calling function needs to free this); if NULL means these weights are needed
 */
float* setViewDependentPolarWeights(parameters* params);

#endif
