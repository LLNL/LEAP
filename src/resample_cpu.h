////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#ifndef __RESAMPLE_CPU_H
#define __RESAMPLE_CPU_H

#ifdef WIN32
#pragma once
#endif

/**
 * This class header contains function to perform up and down-sampling of
 * 3D arrays that reside on the CPU.
 */

class parameters;

/**
 * \fn          downSample
 * \brief       Downsamples 3D array
 * \param[in]   I: pointer to the input 3D data
 * \param[in]   N: 3-element array of the shape of I
 * \param[in]   I_dn: pointer to the down-sampled 3D data
 * \param[in]   N_dn: 3-element array of the shape of I_dn
 * \param[in]   factors: 3-element array down-sampling factors
 * \return      true if the operation was successfull, false otherwise
 */
bool downSample_cpu(float* I, int* N, float* I_dn, int* N_dn, float* factors, int order = 0, float maxWidth = -1.0, float* offset = NULL);

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
 * \param[in]   order: the order of the interpolation kernel
 * \return      true if the operation was successfull, false otherwise
 */
bool upSample_cpu(float* I, int* N, float* I_up, int* N_up, float* factors, int set_type, int order = 0);

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
 * \param[in]   order: the order of the interpolation kernel
 * \return      true if the operation was successfull, false otherwise
 */
bool upSample_high_order_cpu(float* I, int* N, float* I_up, int* N_up, float* factors, int set_type, int order=4);

/**
 * \fn          resampleProjectionAngles_cpu
 * \brief       resamples projection angles
 * \param[in]   g: pointer to the input 3D data of the projections
 * \param[in]   params: pointer to the parameters class object (which is modified by this function)
 * \param[in]   g_new: pointer to the re-sampled 3D projection data
 * \param[in]   phis_new: pointer to the new projection angles
 * \param[in]   N_phis_new: number of samples in phis_new
 * \param[in]   filter_width: the width of the interpolation filter
 * \return      true if the operation was successfull, false otherwise
 */
bool resampleProjectionAngles_cpu(float* g, parameters* params, float* g_new, float* phis_new, int N_phis_new, float filter_width = 0.0, int row_min = -1, int row_max = -1);

/**
 * \fn          bumpFcn
 * \brief       Returns a 1D array which smooths data (and shifts to align samples in new array) before it is downloaded
 * \param[in]   W: width the the smoothing array
 * \param[in]   delay: the group delay of the filter
 * \param[in]   N_taps: the number of elements in the filter
 * \return      a pointer to the smoothing filter
 */
float* bumpFcn(float W, float delay, int& N_taps, int order = 0);

/**
 * \fn          antialias_filter
 * \brief       applies a 3D anti-aliasing filter to a volume
 * \param[in]   volume: pointer to the input 3D data
 * \param[in]   N: 3-element array of the shape of the volume
 * \param[in]   L: filter radius (pixels)
 * \param[in]   order: the order of the accuracy of the low-pass filter
 * \param[in]   h_in: optional user-specified filter
 * \param[in]   N_h: length of optional user-specified filter
 * \param[in]   axis: optional argument lets you specify which dimension to filter
 * * \return      true if the operation was successful, false otherwise
 */
bool antialias_filter(float* volume, int* N, float L, int order = 0, float* h_in = NULL, int N_h = 0, bool* axis = NULL);

/**
 * \fn          finite_difference
 * \brief       applies a finite difference filter to 3D data
 * \param[in]   volume: pointer to the input 3D data
 * \param[in]   N: 3-element array of the shape of the volume
 * \param[in]   order: the order of the accuracy of the finite difference filter
 * \param[in]   shift: -1 for backward difference, 0 for central difference, 1 for forward difference
 * \param[in]   axis: optional argument lets you specify which dimension to filter
 * \param[in]   scalar: optional scalar applied to the filter
 * \return      true if the operation was successful, false otherwise
 */
bool finite_difference(float* volume, int* N, int order, int shift, bool* axis = NULL, float scalar = 1.0);

#endif
