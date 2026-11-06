////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// GPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#ifndef __RESAMPLE_H
#define __RESAMPLE_H

#ifdef WIN32
#pragma once
#endif


/**
 * This class header contains function to perform up and down-sampling of
 * 3D arrays that reside on a GPU.
 */

/**
 * \fn          downSample
 * \brief       Downsamples 3D array
 * \param[in]   I: pointer to the input 3D data
 * \param[in]   N: 3-element array of the shape of I
 * \param[in]   I_dn: pointer to the down-sampled 3D data
 * \param[in]   N_dn: 3-element array of the shape of I_dn
 * \param[in]   factors: 3-element array down-sampling factors
 * \param[in]   whichGPU: index of the GPU where the data resides
 * \return      true if the operation was successfull, false otherwise
 */
bool downSample(float* I, int* N, float* I_dn, int* N_dn, float* factors, int whichGPU);

/**
 * \fn          upSample
 * \brief       Upsamples 3D array
 * \param[in]   I: pointer to the input 3D data
 * \param[in]   N: 3-element array of the shape of I
 * \param[in]   I_up: pointer to the up-sampled 3D data
 * \param[in]   N_up: 3-element array of the shape of I_up
 * \param[in]   factors: 3-element array up-sampling factors
 * \param[in]   set_type: if 0 replaces the data in I_up, if 1 adds to the data in I_up
                if 2 multiplies the data in I_up
 * \param[in]   whichGPU: index of the GPU where the data resides
 * \return      true if the operation was successfull, false otherwise
 */
bool upSample(float* I, int* N, float* I_up, int* N_up, float* factors, int set_type, int whichGPU);

#endif
