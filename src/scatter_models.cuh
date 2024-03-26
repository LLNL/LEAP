////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module scatter simulation and correction
////////////////////////////////////////////////////////////////////////////////

#ifndef __SCATTER_MODELS_H
#define __SCATTER_MODELS_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

/**
 * This header and associated source file provide implementions of CUDA-based scatter simulation and correction algorithms.
 */


// f, the voxelized object model in units of mass density (g/mm^3)
// source, the source spectra
// energies, the energies of the source spectra
// detector, the detector response sampled in 1 keV bins
// sigma, the PE, CS, and RS cross sections sample in 1 keV bins
// scatterDist, the CS and RS distributions sampled in 1 keV bins and 0.1 degree angular bins
bool simulateScatter_firstOrder_singleMaterial(float* g, float* f, parameters* params, float* source, float* energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu);

#endif
