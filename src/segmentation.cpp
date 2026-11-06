////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for 2D sementation
////////////////////////////////////////////////////////////////////////////////

#include "segmentation.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace std;

segmentation::segmentation()
{
    data = NULL;
    data_lo = NULL;
    data_hi = NULL;
    numRows = 0;
    numCols = 0;
}

segmentation::~segmentation()
{
}

bool segmentation::init(float* x, int numRows_in, int numCols_in, float* x_lo, float* x_hi)
{
    data = x;
    data_lo = x_lo;
    data_hi = x_hi;
    numRows = numRows_in;
    numCols = numCols_in;
    numElements = uint64(numRows) * uint64(numCols);
    return initialized();
}    

bool segmentation::initialized()
{
    if (data != NULL && numRows > 0 && numCols > 0)
        return true;
    else
        return false;
}

bool segmentation::threshold(float value, bool greater_than, int fill_type, int num_pixel_dilate)
{
    if (!initialized())
        return false;
    else
    {
        for (uint64 i = 0; i < numElements; i++)
        {
            if ((data[i] >= value && greater_than) || (data[i] <= value && !greater_than))
            {
                if (fill_type == FILL_NAN)
                    data[i] = NAN;
                else if (fill_type == BINERIZE)
                    data[i] = 1.0;
                else //if (fill_type == FLIP_SIGN)
                    data[i] *= -1.0;
            }
            else if (fill_type == BINERIZE)
                data[i] = 0.0;
        }
        return dilate(num_pixel_dilate, fill_type);
    }
}

bool segmentation::region_growing(float startThreshold, float endThreshold, int fill_type, int num_pixel_dilate)
{
    if (!initialized())
        return false;
    
    //if (data_lo != NULL && data_hi != NULL)
    //    printf("have boundary slices\n");

    float* mask = new float[numElements];
    memset(mask, 0, sizeof(float)*numElements);
    
    bool fill_inclusions = true;
    if (startThreshold < endThreshold)
        fill_inclusions = false;

    int blobsofar = 0;
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            uint64 ind = i*numCols+j;
            float curVal = data[ind];
            float extremeVal = curVal;
            if (data_lo != NULL)
            {
                if (fill_inclusions)
                    extremeVal = max(extremeVal, data_lo[ind]);
                else
                    extremeVal = min(extremeVal, data_lo[ind]);
            }
            if (data_hi != NULL)
            {
                if (fill_inclusions)
                    extremeVal = max(extremeVal, data_hi[ind]);
                else
                    extremeVal = min(extremeVal, data_hi[ind]);
            }
            if ((fill_inclusions && extremeVal > startThreshold) || (!fill_inclusions && extremeVal < startThreshold))
            {
                blobsofar++;
                //printf("[%d, %d] = %d, %f\n", i, j, blobsofar, curVal);
                mask[ind] = blobsofar;
                int count = flood_fill(mask, i, j, blobsofar, startThreshold, endThreshold);
                bool keep = false;
                if ((fill_inclusions && curVal > startThreshold) || (!fill_inclusions && curVal < startThreshold))
                    keep = true;
                if (!keep)
                    mask[ind] = 0.0;
            }
        }
    }

    //remove_small_segments(float* mask, 1);

    for (uint64 i = 0; i < numElements; i++)
    {
        if (mask[i] > 0.0)
        {
            if (fill_type == FILL_NAN)
                data[i] = NAN;
            else if (fill_type == BINERIZE)
                data[i] = 1.0;
            else //if (fill_type == FLIP_SIGN)
                data[i] *= -1.0;
        }
        else if (fill_type == BINERIZE)
            data[i] = 0.0;
    }
    delete [] mask;
    
    return dilate(num_pixel_dilate, fill_type);
}

int segmentation::flood_fill(float* mask, int i, int j, int blobnumber, float startThreshold, float endThreshold)
{
    vector<int> stack_i;
    vector<int> stack_j;
    int voxel[2];
    int count = 1;
    
    stack_i.push_back(i);
    stack_j.push_back(j);
    
    bool fill_inclusions = true;
    if (startThreshold < endThreshold)
        fill_inclusions = false;
    
    while (!stack_i.empty())
    {
        voxel[0] = stack_i.back();
        voxel[1] = stack_j.back();
        stack_i.pop_back();
        stack_j.pop_back();
        
        for (int x = voxel[0]-1; x <= voxel[0]+1; x++)
        {
            if (x < 0 || x >= numRows)
                continue;
            for (int y = voxel[1]-1; y <= voxel[1]+1; y++)
            {
                if (y < 0 || y >= numCols)
                    continue;
                uint64 ind = x*numCols+y;
                if (mask[ind] > 0.0)
                    continue;
                float curVal = data[ind];
                if ((fill_inclusions && curVal > endThreshold) || (!fill_inclusions && curVal < endThreshold))
                {
                    //printf("flood_fill: [%d, %d] = %d, %f\n", x, y, blobnumber, curVal);
                    mask[ind] = blobnumber;
                    count++;
                    stack_i.push_back(x);
                    stack_j.push_back(y);
                }
            }
        }
    }
    
    return count;
}

bool segmentation::remove_small_segments(float* mask, int segment_count)
{
    if (mask == NULL || !initialized())
        return false;
    else
    {
        for (uint64 i = 0; i < numElements; i++)
        {
            if (mask[i] <= segment_count)
                mask[i] = 0.0;
        }
        return true;
    }
}

bool segmentation::dilate(int pixelRadius, int fill_type)
{
    if (pixelRadius <= 0)
        return true;
    if (!initialized())
        return false;

    float* orig = new float[numElements];
    memcpy(orig, data, sizeof(float) * numElements);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            float curVal = orig[i*numCols + j];
            if ((fill_type == FILL_NAN && isnan(curVal)) || (fill_type == BINERIZE && curVal > 0.0) || (fill_type == FLIP_SIGN && curVal < 0.0))
            {
                for (int di = -pixelRadius; di <= pixelRadius; di++)
                {
                    int ii = i + di;
                    if (0 <= ii && ii < numRows)
                    {
                        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
                        {
                            int jj = j + dj;
                            if (0 <= jj && jj < numCols)
                                data[ii*numCols + jj] = curVal;
                        }
                    }
                }
            }
        } 
    }
    delete [] orig;
    return true;
}
