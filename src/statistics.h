////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for statistics operations
////////////////////////////////////////////////////////////////////////////////
#ifndef __STATISTICS_H
#define __STATISTICS_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include <vector>
#include "leap_defines.h"

/*
 * This is a collection of multi-core statistics routines.  These are very basic
 * calculations that could have been implemented in Python, but these implementations
 * are much faster.
*/

/**
 * \fn          sum
 * \brief       sum the elements of a given array
 * \param[in]   x: pointer to the data
 * \param[in]   N: number of elements in the array
 * \return      the sum of all elements in the given array
 */
float sum(float* I, uint64 N);
float sum(float* I, int N_1, int N_2, int N_3);

/**
 * \fn          minimum
 * \brief       minimum of the elements in a given array
 * \param[in]   x: pointer to the data
 * \param[in]   N: number of elements in the array
 * \return      minimum of the elements in a given array
 */
float minimum(float* I, uint64 N);
float minimum(float* I, int N_1, int N_2, int N_3);

/**
 * \fn          maximum
 * \brief       maximum of the elements in a given array
 * \param[in]   x: pointer to the data
 * \param[in]   N: number of elements in the array
 * \return      maximum of the elements in a given array
 */
float maximum(float* I, uint64 N);
float maximum(float* I, int N_1, int N_2, int N_3);

/**
 * \fn          range
 * \brief       minimum and maximum of the elements in a given array
 * \param[in]   x: pointer to the data
 * \param[in]   N: number of elements in the array
 * \param[in]   minVal: place to store the minimum value
 * \param[in]   maxVal: place to store the maximum value
 * \return      true if the operation was successful, false otherwise
 */
bool range(float* I, uint64 N, float& minVal, float& maxVal);
bool range(float* I, int N_1, int N_2, int N_3, float& minVal, float& maxVal);

/**
 * \fn          basicStats
 * \brief       minimum, maximum, mean and standard deviation of the elements in a given array
 * \param[in]   x: pointer to the data
 * \param[in]   N: number of elements in the array
 * \param[in]   stats: pointer to array to store the four outputs
 * \return      true if the operation was successful, false otherwise
 */
bool basicStats(float* I, uint64 N, float* stats);
bool basicStats(float* I, int N_1, int N_2, int N_3, float* stats);

/**
 * \fn          percentile_1D
 * \brief       mimics numpy.percentile for a 1D array
 * \param[in]   data: pointer to the data
 * \param[in]   N: number of elements in the array
 * \param[in]   q: desired percentile
 * \return      percentile of the data
 */
float percentile_1D(float* data, uint64 N, float q);

/**
 * \fn          percentile_2D
 * \brief       mimics numpy.percentile for a 2D array
 * \param[in]   data: pointer to the data
 * \param[in]   N: number of elements in the array
 * \param[in]   q: desired percentile
 * \param[in]   percentiles: array to store the calculated percentiles
 * \return      true if the operation was successful, false otherwise
 */
bool percentile_2D(float* I, int N_images, int N, float q, float* percentiles);

/**
 * \fn          histogram
 * \brief       calculates the histogram of the given data
 * \param[in]   I: pointer to the 3D data
 * \param[in]   N_1: number of elements in the first dimension of the array
 * \param[in]   N_2: number of elements in the second dimension of the array
 * \param[in]   N_3: number of elements in the third dimension of the array
 * \param[in]   numBins: desired number of bins in the array
 * \param[in]   binSize: distance between bins of the histogram
 * \param[in]   h: array to store the histogram
 * \param[in]   rangeMin: lower bound of the histogram range; if rangeMax > rangeMin the bins span [rangeMin, rangeMax] and values outside are ignored, otherwise the data min/max are used
 * \param[in]   rangeMax: upper bound of the histogram range (see rangeMin)
 * \return      pointer to the histogram array (calling function is responsible for freeing data)
 */
template <typename T>
float* histogram(const T* I, int N_1, int N_2, int N_3, int& numBins, float& binSize, float* h = NULL, float* bins = NULL, bool include_zeros = true, float rangeMin = 0.0f, float rangeMax = 0.0f);

/**
 * \fn          histogram
 * \brief       calculates the histogram of the given data
 * \param[in]   I: pointer to the data
 * \param[in]   N: number of elements in the array
 * \param[in]   numBins: desired number of bins in the array
 * \param[in]   binSize: distance between bins of the histogram
 * \param[in]   h: array to store the histogram
 * \param[in]   rangeMin: lower bound of the histogram range; if rangeMax > rangeMin the bins span [rangeMin, rangeMax] and values outside are ignored, otherwise the data min/max are used
 * \param[in]   rangeMax: upper bound of the histogram range (see rangeMin)
 * \return      pointer to the histogram array (calling function is responsible for freeing data)
 */
template <typename T>
float* histogram(const T* I, uint64 N, int& numBins, float& binSize, float* h = NULL, float* bins = NULL, bool include_zeros = true, float rangeMin = 0.0f, float rangeMax = 0.0f);

/**
 * \fn          kmeans
 * \brief       calculates the 1D K-means of the given data, uses a histogram-based
 *              method which is much faster and nearly as accurate
 * \param[in]   I: pointer to the 3D data
 * \param[in]   N_1: number of elements in the first dimension of the array
 * \param[in]   N_2: number of elements in the second dimension of the array
 * \param[in]   N_3: number of elements in the third dimension of the array
 * \param[in]   means: pointer to an array to store K means values
 * \param[in]   K: the number of means
 * \return      true is the operation was successful, false otherwise
 */
bool kmeans(float* I, int N_1, int N_2, int N_3, float* means, int K);

std::vector<float> initKMeansPlusPlus1DFromHistogram(const std::vector<float>& bin_centers, const std::vector<float>& counts, int k);

/**
 * \fn          Otsu
 * \brief       calculates the Otsu threshold(s)
 * \param[in]   I: pointer to the 3D data
 * \param[in]   N_1: number of elements in the first dimension of the array
 * \param[in]   N_2: number of elements in the second dimension of the array
 * \param[in]   N_3: number of elements in the third dimension of the array
 * \param[in]   thresholds: pointer to an array to store K thresholds
 * \param[in]   K: the number of thresholds
 * \return      true is the operation was successful, false otherwise
 */
bool Otsu(float* I, int N_1, int N_2, int N_3, float* thresholds, int K);

/**
 * \fn          calculate_centroid
 * \brief       calculates the centroid of a 3D volume
 * \param[in]   I: pointer to the 3D data
 * \param[in]   N_1: number of elements in the first dimension of the array
 * \param[in]   N_2: number of elements in the second dimension of the array
 * \param[in]   N_3: number of elements in the third dimension of the array
 * \param[in]   centroid: 3-element array to store the centroid
 * \return      true is the operation was successful, false otherwise
 */
bool calculate_centroid(float* I, int N_1, int N_2, int N_3, float threshold, float* centroid);

/**
 * \fn          calculate_covariance
 * \brief       calculates the covariance of a 3D volume
 * \param[in]   I: pointer to the 3D data
 * \param[in]   N_1: number of elements in the first dimension of the array
 * \param[in]   N_2: number of elements in the second dimension of the array
 * \param[in]   N_3: number of elements in the third dimension of the array
 * \param[in]   cov: 9-element array to store the covariance matrix
 * \param[in]   centroid: 3-element array to store the centroid
 * \return      true is the operation was successful, false otherwise
 */
bool calculate_covariance(float* I, int N_1, int N_2, int N_3, float threshold, float* cov, float* centroid);

bool sum_first_dimension(float* I, int N_1, int N_2, int N_3, float* sums);

bool sum_dimension(float* I, int N_1, int N_2, int N_3, float* sums, int axis);

#endif
