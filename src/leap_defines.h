////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// global defines, constants, etc
////////////////////////////////////////////////////////////////////////////////

#ifndef __LEAP_DEFINES_H
#define __LEAP_DEFINES_H

#ifdef WIN32
#pragma once
#endif

/**
 * This header file contains commonly used constants in LEAP.
 */

//#define __INCLUDE_CUFFT

#ifndef PI
	#define PI 3.1415926535897932385
#endif

#ifndef RAD_TO_DEG
    #define RAD_TO_DEG 57.29577951308232 // 180.0 / PI
#endif

#ifndef PIINV
    #define PIINV 0.3183098861837907 // 1.0 / PI
#endif

//#ifndef NAN
//	#define NAN sqrt(-1)
//#endif

#ifndef E
	#define E 2.7182818284590452354
#endif

#ifdef WIN32
	typedef unsigned long long uint64;
#else
	typedef unsigned long int uint64;
#endif

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

/**
 * LEAP Error Codes
 */
enum class ErrorCode : int
{
    /**
     * The API call returned with no errors.
     */
    SUCCESS = 0,

    /*----------- VALUE [1, 199] -----------*/
    /**
     * The parameter is not within acceptable range of values.
     */
    OUT_OF_RANGE = 1,

    /**
     * The parameter value is invalid physically such as a negative value for length.
     */
    INVALID_VALUE = 2,

    /**
     * The parameter array size is inconsistent with member data to process together.
     */
    INCONSISTENT_ARRAY_SIZE = 3,

    /*----------- POINTER [200, 299]-----------*/
    /**
     * Null pointer passed to the API which is not allowed.
     */
    NULL_POINTER_NOT_ALLOWED = 200,

    /*----------- FILE SYSTEM [300, 399] -----------*/
    /**
     * File does not exist.
     */
    FILE_NOT_FOUND = 300,

    /**
     * File format is not valid or its content is incomplete.
     */
    FILE_INVALID_FORMAT = 301,

    /**
     * File exists, but not granted to read.
     */
    FILE_NOT_ACCESSIBLE = 302,

    /**
     * Directory does not exist.
     */
    DIRECTORY_NOT_FOUND = 303,

    /*----------- SEQUENCE [400, 499] -----------*/
    /**
     * Prior conditions are not met to run the requested operation.
     */
    PREREQUISITES_NOT_MET = 400,

    /*----------- DEVICE [900, 998] -----------*/
    /**
     * No CUDA device found.
     */
    CUDA_DEVICE_NOT_FOUND = 900,

    /**
     * CUDA API returns error code.
     */
    CUDA_API_ERROR = 901,

    /**
     * There remains an error from a previous CUDA runtime call, but it was not captured and resolved.
     */
    CUDA_ERROR_UNHANDLED = 902,

    /**
     * The installed NVIDIA driver is older than the CUDA runtime library.
     * User should update NVIDIA driver.
     */
    CUDA_INSUFFICIENT_DRIVER = 998,

    /*----------- UNKNOWN [999] -----------*/
    /**
     * Error not classified.
     */
    UNKNOWN_ERROR = 999,
};

#endif
