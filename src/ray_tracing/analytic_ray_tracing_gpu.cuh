////////////////////////////////////////////////////////////////////////////////
// Copyright 2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#ifndef __ANALYTIC_RAY_TRACING_CUH
#define __ANALYTIC_RAY_TRACING_CUH

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
#include <vector>
#include "parameters.h"
#include "phantom.h"
#include "vector_types.h"

/**
 * This class provides GPU-based implementations to perform analytic ray tracing simulation through geometric solids.
 */

struct geometricSolid
{
	int type; // ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7
    float3 centers; // center of the object
    float3 radii; // half axes sizes
    float val; // LAC value inside the object
    float A[9]; // rotation matrix
    float clippingPlanes[6][4]; // clipping planes
    
    bool isRotated;
    int numClippingPlanes;
    float2 clipCone;

    int materialType;
};

/**
 * \fn          setConstantMemoryGeometryParameters
 * \brief       copies the CT geometry parameters to constant GPU memory
 * \param[in]   params: pointer to the parameters class
 * \param[in]   oversampling: over sampling factor for each ray
 * \param[in]   doNormalize: whether to normalize the cone-beam parameters by sdd
 */
void setConstantMemoryGeometryParameters(parameters* params, int oversampling = 1, bool doNormalize = true);

/**
 * \fn          rayTrace_gpu
 * \brief       performs GPU-based analytic ray tracing simulation through geometric solids
 * \param[in]   g: pointer to the projection data
 * \param[in]   params: pointer to the parameters class
 * \param[in]   aPhantom: pointer to an instance of the phantom class which stores the phantom parameters
 * \param[in]	spectralResponse: optional pointer to total system spectral response
 * \param[in]	energies: optional pointer to energy samples of total system spectral response
 * \param[in]   data_on_cpu: true if the projection data is on the CPU, false if it is on the GPU
 * \param[in]   oversampling: over sampling factor for each ray
 */
bool rayTrace_gpu(float* g, parameters* params, phantom* aPhantom, float* spectralResponse, float* energies, int N_energies, bool data_on_cpu, int oversampling = 1);

/**
 * \fn          rayTraceMesh_gpu
 * \brief       performs GPU-based ray tracing through a meshed surface
 * \param[in]   g: pointer to the projection data
 * \param[in]   params: pointer to the parameters class
 * \param[in]   aPhantom: pointer to an instance of the phantom class which stores the phantom parameters
 * \param[in]	spectralResponse: optional pointer to total system spectral response
 * \param[in]	energies: optional pointer to energy samples of total system spectral response
 * \param[in]   data_on_cpu: true if the projection data is on the CPU, false if it is on the GPU
 * \param[in]   oversampling: over sampling factor for each ray
 */
bool rayTraceMesh_gpu(float* g, parameters* params, phantom* aPhantom, float* spectralResponse, float* energies, int N_energies, bool data_on_cpu, int oversampling = 1, int which = -1);

/**
 * \fn          rayTraceMesh_gpu
 * \brief       performs GPU-based mesh voxelization
 * \param[in]   f: pointer to the volume data
 * \param[in]   params: pointer to the parameters class
 * \param[in]   aPhantom: pointer to an instance of the phantom class which stores the phantom parameters
 * \param[in]	val: value to fill the interior of the mesh with
 * \param[in]   data_on_cpu: true if the volume data is on the CPU, false if it is on the GPU
 * \param[in]   oversampling: over sampling factor for each voxel
 */
bool voxelizeMesh_gpu(float* f, parameters* params, phantom* aPhantom, float val, bool data_on_cpu, int oversampling = 1);

#endif
