////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based routines to find the center detector pixel
////////////////////////////////////////////////////////////////////////////////

#ifndef __FIND_CENTER_CPU_H
#define __FIND_CENTER_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide a CPU-based (accelerated by OpenMP) implemention of an algorithm to
 * find the "centerCol" parameter of parallel-, fan-, or cone-beam data.  It does not work with so-called offset scan
 * which is also known as a half-fan or half-cone.
 */

/**
 * \fn          findCenter_cpu
 * \brief       estimates the centerCol or tau parameter by minimizing the RMSE of conjugate rays
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   iRow: index of the detector row for which to minimize the cost function
 * \param[in]   find_tau: if true, finds tau, otherwise find centerCol
 * \param[in]   searchBounds: 2-element array specifying the search bounds for unknown parameter
 * \return      returns the estimated optimal value (centerCol or tau)
 */
float findCenter_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);

/**
 * \fn          findCenter_parallel_cpu
 * \brief       estimates the centerCol parameter by minimizing the RMSE of conjugate rays for parallel-beam data
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   iRow: index of the detector row for which to minimize the cost function
 * \param[in]   searchBounds: 2-element array specifying the search bounds for unknown parameter
 * \return      returns the estimated optimal value (centerCol or tau)
 */
float findCenter_parallel_cpu(float* g, parameters* params, int iRow = -1, float* searchBounds = NULL);

/**
 * \fn          findCenter_fan_cpu
 * \brief       estimates the centerCol or tau parameter by minimizing the RMSE of conjugate rays for fan-beam data
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   iRow: index of the detector row for which to minimize the cost function
 * \param[in]   find_tau: if true, finds tau, otherwise find centerCol
 * \param[in]   searchBounds: 2-element array specifying the search bounds for unknown parameter
 * \return      returns the estimated optimal value (centerCol or tau)
 */
float findCenter_fan_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);

/**
 * \fn          findCenter_cone_cpu
 * \brief       estimates the centerCol or tau parameter by minimizing the RMSE of conjugate rays for axial cone-beam data
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   iRow: index of the detector row for which to minimize the cost function
 * \param[in]   find_tau: if true, finds tau, otherwise find centerCol
 * \param[in]   searchBounds: 2-element array specifying the search bounds for unknown parameter
 * \return      returns the estimated optimal value (centerCol or tau)
 */
float findCenter_cone_cpu(float* g, parameters* params, int iRow = -1, bool find_tau = false, float* searchBounds = NULL);

/**
 * \fn          estimateTilt
 * \brief       estimates the detector tilt (i.e., roll) angle parameter by minimizing the RMSE of conjugate rays
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \return      returns the estimated optimal value (detector tilt, i.e., roll)
 */
float estimateTilt(float* g, parameters* params);

/**
 * \fn          getConjugateDifference
 * \brief       calculates a projection image of conjugate differences
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   alpha: the detector tilt (i.e., roll) to use (ignores the value specified in params)
 * \param[in]   centerCol: the centerCol parameter value to use (ignores the value specified in params)
 * \param[in]   diff: pointer to the 2D projection data to store the result
 * \return      returns true if sucessfull, false otherwise
 */
bool getConjugateDifference(float* g, parameters* params, float alpha, float centerCol, float* diff);

/**
 * \fn          getConjugateProjections
 * \brief       calculates a pair of conjugate projections in cone-parallel coordinates
 * \param[in]   g: pointer to the input 3D attenuation data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   proj_A: pointer to the 2D projection data to store one of the conjugate projections
 * \param[in]   proj_B: pointer to the 2D projection data to store one of the conjugate projections
 * \return      returns true if sucessfull, false otherwise
 */
bool getConjugateProjections(float* g, parameters* params, float*& proj_A, float*& proj_B);

/**
 * \fn          interpolate2D
 * \brief       helper function that performs bilinear interpolation of a 2D array
 * \param[in]   I: pointer to the input 2D data
 * \param[in]   ind_1: index of the first coordinate at which to perform the interpolation
 * \param[in]   ind_2: index of the second coordinate at which to perform the interpolation
 * \param[in]   N_1: number of samples in the first coordinate
 * \param[in]   N_2: number of samples in the second coordinate
 * \return      returns the interpolated value
 */
float interpolate2D(float* I, float ind_1, float ind_2, int N_1, int N_2);

/**
 * \fn          setDefaultRange_centerCol
 * \brief       sets the default search bounds for the centerCol parameter
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   centerCol_low: variable to store the lower search bound
 * \param[in]   centerCol_high: variable to store the higher search bound
 * \return      returns true if sucessfull, false otherwise
 */
bool setDefaultRange_centerCol(parameters* params, int& centerCol_low, int& centerCol_high);

/**
 * \fn          findMinimum
 * \brief       finds the minimum of a 1D array; uses quadratic interpolation to find sub-sample minimum
 * \param[in]   costVec: pointer to the array of cost values
 * \param[in]   startInd: first index to consider for the search
 * \param[in]   endInd: last index to consdier for the search
 * \param[in]   minValue: parameter to store the minimum value
 * \return      returns the value at which the minimum occurs, i.e., argmin
 */
float findMinimum(double* costVec, int startInd, int endInd, float& minValue);

/**
 * \fn          get_rotated_sinogram
 * \brief       interpolates a sinogram from a series of rotated projections
 * \param[in]   g: pointer to 3D projection data
 * \param[in]   params: pointer to a Parameters class object
 * \param[in]   iRow: index of the detector row to perform the calculation
 * \return      pointer to the rotated sinogram (calling function is responsible for freeing this memory)
 */
float* get_rotated_sinogram(float* g, parameters* params, int iRow);

#endif
