////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for 2D segmentation
////////////////////////////////////////////////////////////////////////////////
#ifndef __SEGMENTATION_H
#define __SEGMENTATION_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include <vector>
#include "leap_defines.h"

/*
 * This class implements some very basic 2D segmentation routines based off of thresholding. 
 * For 3D data, the routines are applied individually to each "slice" and these slices
 * are processed in parallel using OpenMP.
 * All of this could be done easily in Python, but the purpose of this function is to provide
 * routines which happen in-place (so no extra memory is required) and are parallelized for faster processing.
*/

class segmentation
{
public:
    segmentation();
    ~segmentation();
    
    /**
	 * \fn          init
	 * \brief       sets the class private variables for processing a given 2D image
	 * \param[in]   x: pointer to the 2D image data
     * \param[in]   numRows: number of rows
     * \param[in]   numCols: number of columns
	 * \return      true if x != NULL && numRows > 0 && numCols > 0, false otherwise
	 */
    bool init(float* x, int numRows, int numCols, float* x_lo = NULL, float* x_hi = NULL);

    // Segmented pixels labeling scheme
    // FILL_NAN: labeled pixels are given a NAN value, all other pixels are unchanged
    // BINERIZE: labeled pixels are given a value of 1.0, all other pixels given a value of 0.0
    enum fill_type_list {FILL_NAN=0, BINERIZE, FLIP_SIGN};
    
    /**
	 * \fn          threshold
	 * \brief       labels pixels in 2D image based on thresholding
	 * \param[in]   value: the threshold value
	 * \param[in]   greater_than: if true labels those pixels that are great than or equal to the given threshold,
                    if false labels those pixels that are less than or equal to the given threshold
     * \param[in]   fill_type: if FILL_NAN, then gives the labeled pixels the value of NAN, all other pixels are unchanged
                    if BINERIZE then give the labeled pixels the value of 1.0 and all other pixels the value of 0.0
     * \param[in]   num_pixel_dilate: pixel radius for which to perform a dilation morphological operation
	 * \return      true if operation  was sucessful, false otherwise
	 */
    bool threshold(float value, bool greater_than = true, int fill_type = FILL_NAN, int num_pixel_dilate = 0);
    
    /**
	 * \fn          region_growing
	 * \brief       labels pixels in 2D image based on dual thresholding method
	 * \param[in]   startThreshold: the threshold value for which to start the region growing
	 * \param[in]   endThreshold: the threshold value for which to stop the region growing
     * \param[in]   fill_type: if FILL_NAN, then gives the labeled pixels the value of NAN, all other pixels are unchanged
                    if BINERIZE then give the labeled pixels the value of 1.0 and all other pixels the value of 0.0
     * \param[in]   num_pixel_dilate: pixel radius for which to perform a dilation morphological operation
	 * \return      true if operation was sucessful, false otherwise
	 */
    bool region_growing(float startThreshold, float endThreshold, int fill_type = FILL_NAN, int num_pixel_dilate = 0);

    /**
	 * \fn          dilate
	 * \brief       performs a morphological dilate operation
     * \param[in]   pixelRadius: pixel radius for which to perform a dilation morphological operation
     * \param[in]   fill_type: if FILL_NAN, then gives the labeled pixels the value of NAN, all other pixels are unchanged
                    if BINERIZE then give the labeled pixels the value of 1.0 and all other pixels the value of 0.0
	 * \return      true if operation was sucessful, false otherwise
	 */
    bool dilate(int pixelRadius, int fill_type = FILL_NAN);
    
private:

    /**
	 * \fn          initialized
	 * \return      true if data != NULL, numRows > 0, and numCols > 0, false otherwise
	 */
    bool initialized();

    /**
	 * \fn          flood_fill
	 * \brief       labels pixels that neighbor the given pixel based on two-value thresholding
     * \param[in]   mask: pointer to the data for which to store the labels
     * \param[in]   i: image row index to start the region growing
     * \param[in]   j: image column index to start the region growing
     * \param[in]   blobnumber: integer label given to the current region
	 * \param[in]   startThreshold: the threshold value for which to start the region growing
     * \param[in]   endThreshold: the threshold value for which to stop the region growing
	 * \return      true if operation was sucessful, false otherwise
	 */
    int flood_fill(float* mask, int i, int j, int blobnumber, float startThreshold, float endThreshold);

    /**
	 * \fn          remove_small_segments
	 * \brief       removes labels of objects that contain less than a specified number of members
     * \param[in]   mask: pointer to the data where each value specifies how many pixels are in its segment
     * \param[in]   segment_count: the lower threshold of segment size for which to remove a segment
	 * \return      true if operation was sucessful, false otherwise
	 */
    bool remove_small_segments(float* mask, int segment_count = 1);

    // The following member variables are just for convenience and simplify the
    // function arguments.  These store the image data and size
    float* data_lo;
    float* data_hi;
    float* data;
    int numRows;
    int numCols;
    uint64 numElements; // equal to numRows * numCols
};

#endif
