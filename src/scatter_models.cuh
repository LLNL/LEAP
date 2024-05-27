////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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
#include "cuda_runtime.h"

/**
 * This header and associated source file provide implementions of CUDA-based scatter simulation and correction algorithms.
 */

/*
struct PhysicsTables
{
	cudaTextureObject_t source_txt;
	cudaTextureObject_t energies_txt;
	cudaTextureObject_t sigma_PE_txt;
	cudaTextureObject_t sigma_CS_txt;
	cudaTextureObject_t sigma_RS_txt;
	cudaTextureObject_t scatterDist_txt;

	int N_energies;
	int maxEnergy;
};
//*/

struct hypercube
{
	int4 N;
	float4 T;
	float4 startVal;
};

// f, the voxelized object model in units of mass density (g/mm^3)
// source: the source spectra
// energies: the energies of the source spectra
// detector: the detector response sampled in 1 keV bins
// sigma: the PE, CS, and RS cross sections sampled in 1 keV bins
// scatterDist: the CS and RS distributions sampled in 1 keV bins and 1.0 degree angular bins
bool simulateScatter_firstOrder_singleMaterial(float* g, float* f, parameters* params, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType);

bool firstLeg(cudaTextureObject_t f_data_txt, parameters* params, float* dev_Df, float3 sourcePosition);

#endif
