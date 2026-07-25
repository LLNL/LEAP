////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based sinogram replacement (a MAR method) routines
////////////////////////////////////////////////////////////////////////////////

#ifndef __SINOGRAM_REPLACEMENT_H
#define __SINOGRAM_REPLACEMENT_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * \fn          sinogramReplacement
 * \brief       performs linear interpolation across metal traces
 * \param[in]   g: pointer to the projection data, nan values specify those pixels to be interpolated
 * \param[in]   params: pointer to the parameters class which describes the data
 * \param[in]   windowSize: 3-element array specifying the window size in each of the 3 dimensions of the data
 *              for which to perform the sinogram replacement
 * \return      true if operation  was sucessful, false otherwise
 */
bool sinogramReplacement(float* g, parameters* params, int* windowSize, int padSide);

/**
 * \fn          sinogramReplacement
 * \brief       performs linear regression-based sinogram replacement across metal traces
 * \param[in]   g: pointer to the projection data
 * \param[in]   prior: pointer to the prior projection data which is used to patch in the metal traces
 * \param[in]   metalTrace: pointer to the projection mask which identifies which pixels are to be replaced;
 *              a positive number specifies that a pixel is bad and needs to be replaced
 * \param[in]   params: pointer to the parameters class which describes the data
 * \param[in]   windowSize: 3-element array specifying the window size in each of the 3 dimensions of the data
 *              for which to perform the sinogram replacement
 * \return      true if operation  was sucessful, false otherwise
 */
bool sinogramReplacement(float* g, float* prior, float* metalTrace, parameters* params, int* windowSize, int padSide);

#endif
