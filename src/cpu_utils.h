////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for some CPU-based computations
////////////////////////////////////////////////////////////////////////////////

#ifndef __CPU_UTILS_H
#define __CPU_UTILS_H

#ifdef WIN32
#pragma once
#endif

#include <cstddef>
#include <cstdint>
#include "parameters.h"

/**
 * This header and associated source file are for generic CPU-based functions that are used in LEAP.
 * Whenever possible, these operations are accelerated by OpenMP (multi-thread CPU).
 */


/**
 * \fn          optimalFFTsize
 * \brief       returns the smallest number of the form N <= 2^(m+1)*3^n
 * \param[in]   N: number of elements
 * \return      returns the smallest number of the form N <= 2^(m+1)*3^n
 */ 
int optimalFFTsize(int N);

/**
 * \fn          getSlice
 * \brief       returns a pointer to a particular volume slice
 * \param[in]   volume: pointer to the volume data
 * \param[in]   iz: index to a particular slice
 * \param[in]   params: pointer to the parameters class
 * \return      returns a pointer to a particular volume slice
 */
float* getSlice(float* volume, int iz, parameters* params);

/**
 * \fn          getProjection
 * \brief       returns a pointer to a particular projection
 * \param[in]   projections: pointer to the projection data
 * \param[in]   iView: index to a particular view
 * \param[in]   params: pointer to the parameters class
 * \return      returns a pointer to a particular projection
 */
float* getProjection(float* projections, int iView, parameters* params);


/**
 * \fn          tex3D
 * \brief       returns the value of the volume at the specified indices
 * \param[in]   f: pointer to the volume data
 * \param[in]   iz: index to a particular z-slice
 * \param[in]   iy: index to a particular y-slice
 * \param[in]   ix: index to a particular x-slice
 * \param[in]   params: pointer to the parameters class
 * \return      returns the value of the volume at the specified indices
 */
float tex3D(float* f, int iz, int iy, int ix, parameters* params);

/**
 * \fn          tex3D
 * \brief       returns the value of the volume at the specified real-valued indices using trilinear interpolation
 * \param[in]   f: pointer to the volume data
 * \param[in]   iz: z-coordinate index
 * \param[in]   iy: y-coordinate index
 * \param[in]   ix: x-coordinate index
 * \param[in]   params: pointer to the parameters class
 * \return      returns the value of the volume at the specified indices
 */
float tex3D(float* f, float iz, float iy, float ix, parameters* params);

/**
 * \fn          tex3D_rev
 * \brief       returns the value of the volume at the specified real-valued indices using trilinear interpolation
 * \param[in]   f: pointer to the volume data
 * \param[in]   ix: x-coordinate index
 * \param[in]   iy: y-coordinate index
 * \param[in]   iz: z-coordinate index
 * \param[in]   params: pointer to the parameters class
 * \return      returns the value of the volume at the specified indices
 */
float tex3D_rev(float* f, float ix, float iy, float iz, parameters* params);

/**
 * \fn          reorder_ZYX_to_XYZ
 * \brief       changes the order of the input from ZYX to XYZ
 * \param[in]   f: pointer to the volume data
 * \param[in]   params: pointer to the parameters class
 * \param[in]   sliceStart: the first z slice to include in the reordering
 * \param[in]   sliceEnd: the last z slice to include in the reordering
 * \return      returns a pointer to the reorder data (calling function is responsible to freeing this memory)
 */
float* reorder_ZYX_to_XYZ(float* f, parameters* params, int sliceStart, int sliceEnd);


/**
 * \fn          innerProduct_cpu
 * \brief       returns the inner product of two 3D arrays
 * \param[in]   x: pointer to a 3D array
 * \param[in]   y: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      inner product of two 3D arrays
 */
float innerProduct_cpu(float* x, float* y, int N_1, int N_2, int N_3);

/**
 * \fn          equal_cpu
 * \brief       sets the values of an array to those of another array, i.e., x[:]=y[:]
 * \param[in]   x: pointer to a 3D array
 * \param[in]   y: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      true if the operation was successful, false otherwise
 */
bool equal_cpu(float* x, float* y, int N_1, int N_2, int N_3);

/**
 * \fn          equal_cpu
 * \brief       sets the values of an array to a given constant value, i.e., x[:]=c
 * \param[in]   x: pointer to a 3D array
 * \param[in]   c: scalar
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      true if the operation was successful, false otherwise
 */
bool equal_cpu(float* x, float c, int N_1, int N_2, int N_3);

/**
 * \fn          scale_cpu
 * \brief       scake the values of an array by a given constant value, i.e., x[:]*=c
 * \param[in]   x: pointer to a 3D array
 * \param[in]   c: scalar
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      true if the operation was successful, false otherwise
 */
bool scale_cpu(float* x, float c, int N_1, int N_2, int N_3);

/**
 * \fn          sub_cpu
 * \brief       subtracts the values of an array by those from another array, i.e., x[:]-=y[:]
 * \param[in]   x: pointer to a 3D array
 * \param[in]   y: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      true if the operation was successful, false otherwise
 */
bool sub_cpu(float* x, float* y, int N_1, int N_2, int N_3);

/**
 * \fn          sub_cpu
 * \brief       adds the values of an array by those from a scaled version of another array, i.e., x[:]+=c*y[:]
 * \param[in]   x: pointer to a 3D array
 * \param[in]   c: scalar
 * \param[in]   y: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      true if the operation was successful, false otherwise
 */
bool scalarAdd_cpu(float* x, float c, float* y, int N_1, int N_2, int N_3);

/**
 * \fn          clip_cpu
 * \brief       clips the given input array by a specified lower bound, i.e., x=max(x,c)
 * \param[in]   x: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \param[in]   clipVal: the lower value for the clipping
 * \return      true if the operation was successful, false otherwise
 */
bool clip_cpu(float* x, int N_1, int N_2, int N_3, float clipVal = 0.0);

/**
 * \fn          replaceZeros_cpu
 * \brief       replaces the zero values of the input array by a specified value, i.e., x[x=0.0]=newVal
 * \param[in]   x: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \param[in]   newVal: the value to replace with
 * \return      true if the operation was successful, false otherwise
 */
bool replaceZeros_cpu(float* x, int N_1, int N_2, int N_3, float newVal = 1.0);

/**
 * \fn          sum_cpu
 * \brief       returns the sum of a 3D array
 * \param[in]   x: pointer to a 3D array
 * \param[in]   N_1: number of elements in the 1st dimension
 * \param[in]   N_2: number of elements in the 2nd dimension
 * \param[in]   N_3: number of elements in the 3rd dimension
 * \return      sum of all elements of the input array
 */
float sum_cpu(float* x, int N_1, int N_2, int N_3);


/**
 * \fn          windowFOV_cpu
 * \brief       masks out those voxels in the volume that are outside the field of view
 * \param[in]   x: pointer to a 3D array
 * \param[in]   params: pointer to an object from the parameters class
 * \return      true if the operation was successful, false otherwise
 */
bool windowFOV_cpu(float* f, parameters* params);

/**
 * \fn          rotateAroundAxis
 * \brief       rotates aVec phi radians around theAxis
 * \param[in]   theAxis: pointer 3-element array
 * \param[in]   phi: rotation angle (radians)
 * \param[in]   aVec: pointer 3-element array
 * \return      aVec
 */
float* rotateAroundAxis(float* theAxis, float phi, float* aVec);

// Returns true if the array contains any NaN or ±Inf values
bool has_nan_or_inf_omp_fastmath(const float* a, std::size_t n);

/**
 * \fn          has_nan
 * \brief       returns whether or not the given array has any nan values
 * \param[in]   x: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \return      true if any of the value are nan, false otherwise
 */
bool has_nan(float* x, int N_1, int N_2 = 1, int N_3 = 1);

/**
 * \fn          replace_nan
 * \brief       Replaces NAN values in numpy array with alternate value
 * \param[in]   x: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \param[in]   newValue: the value to replace the NAN values with
 * \return      true if the operation was successful, false otherwise
 */
bool replace_nan(float* x, int N_1, int N_2, int N_3, float newValue);

/**
 * \fn          bounding_box
 * \brief       Sets the axis-aligned bounding box
 * \param[in]   x: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \param[in]   boundary_type: 0 for positive values, 1 for negative values, 2 for NAN values
 * \param[in]   AABB: the axes indices for the bounding box
 * \return      true if the operation was successful, false otherwise
 */
bool bounding_box(float* x, int N_1, int N_2, int N_3, int boundary_type, int* AABB);

/**
 * \fn          step_function
 * \brief       Calculates the step function of the given input
 * \param[in]   x: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \param[in]   scale: optional scalar value applied to the input
 * \param[in]   shift: optional shift value applied to the input
 * \return      true if the operation was successful, false otherwise
 */
bool step_function(float* x, int N_1, int N_2, int N_3, float scale, float shift);

/**
 * \fn          dirac_function
 * \brief       Calculates the dirac delta function of the given input
 * \param[in]   x: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \param[in]   scale: optional scalar value applied to the input
 * \param[in]   shift: optional shift value applied to the input
 * \return      true if the operation was successful, false otherwise
 */
bool dirac_function(float* x, int N_1, int N_2, int N_3, float scale, float shift);

/**
 * \fn          badPixelCorrection_cpu
 * \brief       Performs 2D pad pixel correction
 * \param[in]   g: pointer to array
 * \param[in]   N_1: number of elements in the first dimension
 * \param[in]   N_2: number of elements in the second dimension
 * \param[in]   N_3: number of elements in the third dimension
 * \param[in]   badPixelMap: positive values mark location of bad pixels
 * \param[in]   w: median filter radius
 * \return      true if the operation was successful, false otherwise
 */
bool badPixelCorrection_cpu(float* g, int N_1, int N_2, int N_3, float* badPixelMap, int w);

/**
 * \fn          malloc_aligned
 * \brief       Allocates aligned memory for a floating point array
 * \param[in]   num_bytes: number of bytes to allocate
 * \param[in]   alignment: byte alignment (can be 16, 32, or 64)
 * \return      pointer to the allocated array
 */
float* malloc_aligned(size_t num_bytes, int alignment = 32);

/**
 * \fn          calloc_aligned
 * \brief       Allocates aligned memory for a floating point array and initializes it to zero
 * \param[in]   num_bytes: number of bytes to allocate
 * \param[in]   alignment: byte alignment (can be 16, 32, or 64)
 * \return      pointer to the allocated array
 */
float* calloc_aligned(size_t num_bytes, int alignment = 32);

/**
 * \fn          free_aligned
 * \brief       frees aligned memory of a floating point array
 * \param[in]   data: pointer to the data
 * \return      true if successful, false otherwise
 */
bool free_aligned(float* data);

/**
 * \fn          getAvailableSystemMemory
 * \brief       returns the available system memory in GB
 * \return      available system memory in GB, or 0.0 on unsupported platforms
 */
float getAvailableSystemMemory();

extern int max_threads;

/**
 * \fn          num_cpu_threads
 * \return      the number of CPU threads to use
 */
int num_cpu_threads();

// These functions swap the endian of the given input
char swapEndian(char);
short swapEndian(short);
unsigned short swapEndian(unsigned short);
int swapEndian(int);
float swapEndian(float);
double swapEndian(double);
unsigned int swapEndian(unsigned int);

template <typename T>
T bswap(T val);

void unpack01_from_float(float packed, float& a, float& b);

// Unpack two IEEE half-precision (fp16) values packed into one 32-bit float
// (high 16 bits -> a, low 16 bits -> b). Matches pack_half2_as_float() on the GPU.
void unpack_half2_from_float(float packed, float& a, float& b);

// The following enables a stop watch to perform speed tests
/* Usage:
Timer stopWatch;
stopWatch.tick();
...
stopWatch.tock();
printf("Elapsed time: %f\n", stopWatch.duration().count());
*/
#include <chrono>
#include <assert.h>

using namespace std::chrono_literals;

template <
    class DT = std::chrono::duration<double>,
    class ClockT = std::chrono::steady_clock>
class Timer
{
    using timep_t = decltype(ClockT::now());

    timep_t _start = ClockT::now();
    timep_t _end = {};

public:
    void tick() {
        _end = timep_t{};
        _start = ClockT::now();
    }

    void tock() {
        _end = ClockT::now();
    }

    template <class duration_t = DT>
    auto duration() const {
        // Use gsl_Expects if your project supports it.
        assert(_end != timep_t{} && "Timer must toc before reading the time");
        return std::chrono::duration_cast<duration_t>(_end - _start);
    }
};

#endif
