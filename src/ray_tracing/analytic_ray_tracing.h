////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////
#ifndef __ANALYTIC_RAY_TRACING_H
#define __ANALYTIC_RAY_TRACING_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"
#include "phantom.h"

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to perform analytic ray tracing simulation through geometric solids.
 */

class analyticRayTracing
{
public:

    // Constructor and destructor; these do nothing
    analyticRayTracing();
    ~analyticRayTracing();
    analyticRayTracing(parameters*);

    /**
     * \fn          rayTrace
     * \brief       performs multi-core CPU-based analytic ray tracing simulation through geometric solids
     * \param[in]   g: pointer to the projection data
     * \param[in]   params: pointer to the parameters class
     * \param[in]   aPhantom: pointer to an instance of the phantom class which stores the phantom parameters
     * \param[in]   oversampling: over sampling factor for each ray
     */
    bool rayTrace(float* g, parameters* params, phantom* aPhantom, int oversampling = 1);

    /**
     * \fn          setSourcePosition
     * \brief       calculate the 3D source position for any geometry
     * \param[in]   iView: view index
     * \param[in]   iRow: detector row index
     * \param[in]   iCol: detector column index
     * \param[in]   sourcePosition: pointer to a 3-element array to save the source position to
     * \param[in]   dv: the deviation for the detector row position (for oversampling)
     * \param[in]   du: the deviation for the detector column position (for oversampling)
     * \param[in]   oversampling: over sampling factor for each ray
     */
    bool setSourcePosition(int iView, int iRow, int iCol, double* sourcePosition, double dv = 0.0, double du = 0.0);

    /**
     * \fn          setTrajectory
     * \brief       calculate the normalized 3D vector that points from a source position to a detector pixel
     * \param[in]   iView: view index
     * \param[in]   iRow: detector row index
     * \param[in]   iCol: detector column index
     * \param[in]   traj: pointer to a 3-element array to save the source position to
     * \param[in]   dv: the deviation for the detector row position (for oversampling)
     * \param[in]   du: the deviation for the detector column position (for oversampling)
     * \param[in]   oversampling: over sampling factor for each ray
     */
    bool setTrajectory(int iView, int iRow, int iCol, double* traj, double dv = 0.0, double du = 0.0);

    /**
     * \fn          setModuleCenter
     * \brief       returns the module center vector
     * \param[in]   iView: view index
     * \param[in]   v: vector to store the result
     * \return      true if successful, false otherwise
     */
    bool setModuleCenter(int iView, double* v);

    /**
     * \fn          setRowVector
     * \brief       returns the module row vector
     * \param[in]   iView: view index
     * \param[in]   v: vector to store the result
     * \return      true if successful, false otherwise
     */
    bool setRowVector(int iView, double* v);

    /**
     * \fn          setColVector
     * \brief       returns the module col vector
     * \param[in]   iView: view index
     * \param[in]   v: vector to store the result
     * \return      true if successful, false otherwise
     */
    bool setColVector(int iView, double* v);

    /**
     * \fn          setDetectorNormal
     * \brief       returns the module normal vector
     * \param[in]   iView: view index
     * \param[in]   v: vector to store the result
     * \return      true if successful, false otherwise
     */
    bool setDetectorNormal(int iView, double* v);

private:
    // pointer to the parameters class which stores the CT geometry
    parameters* params;
};

#endif
