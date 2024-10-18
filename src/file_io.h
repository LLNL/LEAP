////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for some file I/O routines
////////////////////////////////////////////////////////////////////////////////

#ifndef __FILE_IO_H
#define __FILE_IO_H

#ifdef WIN32
#pragma once
#endif

//#define USE_OLD_PARAM_NAMES
#include "parameters.h"
#include "leap_defines.h"

/**
 * This file specifies routines to save CT geometry files and read/write TIFF images.
 */

/**
 * \fn          saveParamsToFile
 * \brief       save the CT geometry and CT volume parameters to an ascii text file
 * \param[in]	param_fn, the name of the file to save the parameter values to
 * \return      returns true if successful, false otherwise
 */
bool saveParametersToFile(const char* param_fn, parameters* params);

#define MAXFILENAME 2048

struct ImageHeader
{
    uint32 numCols;      // width of the image
    uint32 numRows;     // height of the image
    uint32 numFrames;  // number of frames
    double pixelWidth;     // pixel width
    double pixelHeight;     // pixel height
    int datatype;   // TIFF SampleFormat: 1:uint, 2:int, 3:float, 4:undef; bitdepth unspecified
                           // Here:
                           // 0: 8-bit unsigned char   (can be slope/offset)
                           // 1: 16-bit unsigned short (can be slope/offset)
                           // 2: 32-bit signed int     (can be slope/offset)
                           // 3: 32-bit float
                           // 4: 64-bit double
    float slope;      // these scale integers back to floating point; the conversion is
    float offset;     // f = i * slope + offset, which means i = (f-offset) / slope
    char fileName[MAXFILENAME];
    size_t bytes_of_data; //just a check to not overrun the file
    size_t offset_to_data; //the offset from the beginning of the file to first data; used in small reads.
    int compressionType;

    bool bigEndian;
};

FILE* read_header_leave_open(ImageHeader* h);
bool read_header(char* fileName, int* shape, float* size, float* slope_and_offset);
float* load_tif(char* fileName, float* data = NULL);
float* load_tif_rows(char* fileName, int firstRow, int lastRow, float* data = NULL);
float* load_tif_rows(ImageHeader* h, int firstRow, int lastRow, float* data = NULL);
float* load_tif_cols(char* fileName, int firstCol, int lastCol, float* data = NULL);
float* load_tif_roi(char* fileName, int firstRow, int lastRow, int firstCol, int lastCol, float* data = NULL);
bool write_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight = 1.0, float pixelWidth = 1.0, int dtype = 3, float wmin = 0.0, float wmax = 1.0);


#endif
