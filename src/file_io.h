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

/**
 * \fn          saveParamsToFile
 * \brief       save the CT geometry and CT volume parameters to an ascii text file
 * \param[in]	param_fn, the name of the file to save the parameter values to
 * \return      returns true if successful, false otherwise
 */
bool saveParametersToFile(const char* param_fn, parameters* params);

// The following are not implemented yet, but the purpose of these is to develop
// routines that can read a subset of the image rows from a tif file rapidly
bool tif_image_size(const char* fileName, int& numRows, int& numCols);
bool read_tif(const char* fileName, float* data, int firstRow = -1, int lastRow = -1);
bool save_tif(const char* fileName, float* data, int numRows, int numCols, bool quantize = false);

#endif
