////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Class for performing analytic inversion, i.e., Filtered Backprojection (FBP)
// algorithms.
////////////////////////////////////////////////////////////////////////////////

#include "filtered_backprojection.h"
#include "projectors.h"
#include "ramp_filter.cuh"
#include "ray_weighting_cpu.h"
#include "ray_weighting.cuh"
#include "cuda_utils.h"
#include "projectors_symmetric.cuh"
#include "projectors_symmetric_cpu.h"
#include "ramp_filter_cpu.h"
#include "projectors_attenuated.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>

filteredBackprojection::filteredBackprojection()
{

}

filteredBackprojection::~filteredBackprojection()
{

}

bool filteredBackprojection::HilbertFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar, float sampleShift)
{
	return convolve1D(g, params, cpu_to_gpu, scalar, 1, sampleShift);
}

bool filteredBackprojection::rampFilterProjections(float* g, parameters* params, bool cpu_to_gpu, float scalar)
{
	return convolve1D(g, params, cpu_to_gpu, scalar, 0);
}

bool filteredBackprojection::convolve1D(float* g, parameters* params, bool cpu_to_gpu, float scalar, int whichFilter, float sampleShift)
{
	if (params->whichGPU < 0)
	{
		if (params->isSymmetric())
		{
			float* g_left = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
			float* g_right = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
			splitLeftAndRight(g, g_left, g_right, params);
			bool retVal = true;
			retVal = rampFilter1D_cpu(g_left, params, scalar);
			rampFilter1D_cpu(g_right, params, scalar);
			mergeLeftAndRight(g, g_left, g_right, params);
			free(g_left);
			free(g_right);
			return retVal;
		}
		else
		{
			if (whichFilter == 0)
				return rampFilter1D_cpu(g, params, scalar);
			else
				return Hilbert1D_cpu(g, params, scalar);
		}
	}
	else
	{
		if (params->isSymmetric())
		{
			if (cpu_to_gpu)
			{
				float* g_left = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
				float* g_right = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
				splitLeftAndRight(g, g_left, g_right, params);
				bool retVal = rampFilter1D(g_left, params, cpu_to_gpu, scalar);
				rampFilter1D(g_right, params, cpu_to_gpu, scalar);
				mergeLeftAndRight(g, g_left, g_right, params);
				free(g_left);
				free(g_right);
				return retVal;
			}
			else
				return rampFilter1D_symmetric(g, params, scalar);
		}
		else
		{
			if (whichFilter == 0)
			{
				/*
				ray_derivative(g, params, cpu_to_gpu, scalar, 1.0);
				return Hilbert1D(g, params, cpu_to_gpu, 1.0, -1.0);
				//*/
				return rampFilter1D(g, params, cpu_to_gpu, scalar);
			}
			else
				return Hilbert1D(g, params, cpu_to_gpu, scalar, sampleShift);
		}
	}
}

bool filteredBackprojection::filterProjections(float* g, parameters* params, bool cpu_to_gpu)
{
	if (params->geometry == parameters::MODULAR)
	{
		printf("Error: not implemented for modular geometries\n");
		return false;
	}

	if (params->muSpecified())
		return filterProjections_Novikov(g, params, cpu_to_gpu);

	if (params->whichGPU < 0 || cpu_to_gpu == false)
	{
		// no transfers to/from GPU are necessary; just run the code
		if (params->geometry == parameters::CONE && params->helicalPitch != 0.0)
		{
			parallelRay_derivative(g, params, false);
			applyPostRampFilterWeights(g, params, false);
			return HilbertFilterProjections(g, params, false, FBPscalar(params), -1.0);
		}
		else
		{
			applyPreRampFilterWeights(g, params, cpu_to_gpu);
			if (params->muCoeff != 0.0)
				convertARTtoERT(g, params, false, false);
			rampFilterProjections(g, params, cpu_to_gpu, FBPscalar(params));
			if (params->muCoeff != 0.0)
				convertARTtoERT(g, params, false, true);
			return applyPostRampFilterWeights(g, params, cpu_to_gpu);
		}
	}
	else
	{
		if (getAvailableGPUmemory(params->whichGPU) < params->projectionDataSize())
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		bool retVal = true;

		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		float* dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		if (dev_g == 0)
			return false;
		retVal = filterProjections(dev_g, params, false);

		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

		if (dev_g != 0)
			cudaFree(dev_g);
		return retVal;
	}
}

bool filteredBackprojection::execute(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (params->geometry == parameters::MODULAR)
	{
		printf("Error: FBP not implemented for modular geometries\n");
		return false;
	}
	if (params->geometry == parameters::CONE && params->helicalPitch != 0.0 && params->whichGPU < 0)
	{
		printf("Error: CPU-based FBP not yet implemented for helical cone-beam geometry\n");
		return false;
	}

	if (params->muSpecified())
		return execute_attenuated(g, f, params, cpu_to_gpu);

	if (params->whichGPU < 0 || cpu_to_gpu == false)
	{
		float* g_pad = NULL;
		float* g_save = g;
		if (params->offsetScan)
		{
			/*
			if (params->whichGPU >= 0 && cpu_to_gpu == false)
			{
				printf("Error: currently offsetScan reconstruction only works on CPU or on GPU and the data resided on the CPU\n");
				printf("In other words, you did something we did not expect.  Please submit a new feature request\n");
				return false;
			}
			//*/
			g_pad = zeroPadForOffsetScan(g, params);
			params->offsetScan = false;
			if (g_pad != NULL)
				g = g_pad;
		}

		// no transfers to/from GPU are necessary; just run the code
		filterProjections(g, params, false);

		bool retVal = true;
		bool doWeightedBackprojection_save = params->doWeightedBackprojection;
		params->doWeightedBackprojection = true;
		bool doExtrapolation_save = params->doExtrapolation;
		if (params->geometry != parameters::CONE || params->helicalPitch == 0.0)
			params->doExtrapolation = true;
		if (params->isSymmetric())
		{
			if (params->whichGPU < 0)
				retVal = CPUinverse_symmetric(g, f, params);
			else
				retVal = inverse_symmetric(g, f, params, cpu_to_gpu);
		}
		else
			retVal = proj.backproject(g, f, params, cpu_to_gpu);
		params->doWeightedBackprojection = doWeightedBackprojection_save;
		params->doExtrapolation = doExtrapolation_save;
		params->colShiftFromFilter = 0.0;
		params->rowShiftFromFilter = 0.0;

		if (g_pad != NULL)
			free(g_pad);

		return retVal;
	}
	else
	{
		if (params->hasSufficientGPUmemory() == false)
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		bool retVal = true;

		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;
		float* dev_g = 0;

		float* g_pad = zeroPadForOffsetScan(g, params);
		if (g_pad != NULL)
		{
			dev_g = copyProjectionDataToGPU(g_pad, params, params->whichGPU);
			params->offsetScan = false;
			free(g_pad);
		}
		else
		{
			dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		}
		if (dev_g == 0)
			return false;

		float* dev_f = 0;
		if ((cudaStatus = cudaMalloc((void**)&dev_f, params->numX * params->numY * params->numZ * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			if (dev_g != 0)
				cudaFree(dev_g);
			return false;
		}

		retVal = execute(dev_g, dev_f, params, false);

		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
		if (dev_f != 0)
			cudaFree(dev_f);
		if (dev_g != 0)
			cudaFree(dev_g);
		return retVal;
	}
}

bool filteredBackprojection::execute_attenuated(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (params->mu != NULL)
		return execute_Novikov(g, f, params, cpu_to_gpu);
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * params->T_phi())
	{
		printf("Error: FBP of attenuated x-ray transform requires at least 360 degree angular range!\n");
		return false;
	}
	if (params->whichGPU < 0 || cpu_to_gpu == false)
	{
		// data transfers to/from GPU are not necessary
		// don't need memory checks
		applyPreRampFilterWeights(g, params, cpu_to_gpu);
		convertARTtoERT(g, params, false, false);
		rampFilterProjections(g, params, false, FBPscalar(params));
		convertARTtoERT(g, params, false, true);

		params->muCoeff *= -1.0;
		bool retVal = proj.backproject(g, f, params, false);
		params->muCoeff *= -1.0;
		return retVal;
	}
	else
	{
		if (params->hasSufficientGPUmemory() == false)
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		bool retVal = true;

		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		float* dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		if (dev_g == 0)
			return false;

		float* dev_f = 0;
		if ((cudaStatus = cudaMalloc((void**)&dev_f, params->numX * params->numY * params->numZ * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			if (dev_g != 0)
				cudaFree(dev_g);
			return false;
		}

		retVal = execute_attenuated(dev_g, dev_f, params, false);

		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
		if (dev_f != 0)
			cudaFree(dev_f);
		if (dev_g != 0)
			cudaFree(dev_g);
		return retVal;
	}
}

bool filteredBackprojection::filterProjections_Novikov(float* g, parameters* params, bool cpu_to_gpu)
{
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * params->T_phi())
	{
		printf("Error: FBP of attenuated x-ray transform requires at least 360 degree angular range!\n");
		return false;
	}
	if (params->whichGPU < 0)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for GPU!\n");
		return false;
	}

	float* params_mu_save = params->mu;
	float* dev_g = 0;
	float* dev_mu = 0;
	if (cpu_to_gpu)
	{
		if (params->hasSufficientGPUmemory() == false)
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		if (dev_g == 0)
			return false;

		dev_mu = copyVolumeDataToGPU(params->mu, params, params->whichGPU);
		if (dev_mu == 0)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			if (dev_g != 0)
				cudaFree(dev_g);
			return false;
		}

	}
	else
	{
		dev_g = g;
		dev_mu = params->mu;
	}

	params->mu = NULL;
	int3 N_g = make_int3(params->numAngles, params->numRows, params->numCols);

	// *** NOVIKOV ALGORITHM START ***
	// Calculate h_1 and h_2
	float* h_1 = 0;
	float* h_2 = 0;
	cudaMalloc((void**)&h_1, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&h_2, params->numAngles * params->numRows * params->numCols * sizeof(float));
	proj.project(h_1, dev_mu, params, false);
	scale(h_1, 0.5, N_g, params->whichGPU);
	equal(h_2, h_1, N_g, params->whichGPU);
	Hilbert1D(h_2, params, false);

	// Calculate g_a
	float* g_a_1 = 0;
	float* g_a_2 = 0;
	float* temp = 0;
	cudaMalloc((void**)&g_a_1, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&g_a_2, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&temp, params->numAngles * params->numRows * params->numCols * sizeof(float));

	// Calculate g_a_1
	equal(g_a_1, dev_g, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	cosFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	expFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	Hilbert1D(g_a_1, params, false);

	// Calculate g_a_2
	equal(g_a_2, dev_g, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	sinFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	expFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	Hilbert1D(g_a_2, params, false);

	// Combine g_a_1 and g_a_2
	float* g_a = dev_g;
	equal(temp, h_2, N_g, params->whichGPU);
	sinFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	cosFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	equal(g_a, g_a_2, N_g, params->whichGPU);
	add(g_a, g_a_1, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	negExpFcn(temp, N_g, params->whichGPU);
	multiply(g_a, temp, N_g, params->whichGPU);

	cudaFree(temp);
	cudaFree(h_1);
	cudaFree(h_2);
	cudaFree(g_a_1);
	cudaFree(g_a_2);
	params->mu = dev_mu;

	// Clean up
	params->mu = params_mu_save;
	if (cpu_to_gpu)
	{
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_mu != 0)
			cudaFree(dev_mu);
	}
	return true;
}

bool filteredBackprojection::execute_Novikov(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * params->T_phi())
	{
		printf("Error: FBP of attenuated x-ray transform requires at least 360 degree angular range!\n");
		return false;
	}
	if (params->whichGPU < 0)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for GPU!\n");
		return false;
	}

	//printf("Error: FBP of attenuated x-ray transform implementation still in progress!\n");
	//return false;

	float* params_mu_save = params->mu;
	float* dev_g = 0;
	float* dev_f = 0;
	float* dev_mu = 0;
	if (cpu_to_gpu)
	{
		if (params->hasSufficientGPUmemory() == false)
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		if (dev_g == 0)
			return false;

		dev_mu = copyVolumeDataToGPU(params->mu, params, params->whichGPU);
		if (dev_mu == 0)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			if (dev_g != 0)
				cudaFree(dev_g);
			return false;
		}

		dev_f = 0;
		if ((cudaStatus = cudaMalloc((void**)&dev_f, params->numX * params->numY * params->numZ * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			if (dev_g != 0)
				cudaFree(dev_g);
			if (dev_mu != 0)
				cudaFree(dev_mu);
			return false;
		}
	}
	else
	{
		dev_g = g;
		dev_f = f;
		dev_mu = params->mu;
	}
	params->mu = dev_mu;

	int3 N_g = make_int3(params->numAngles, params->numRows, params->numCols);

	// *** NOVIKOV ALGORITHM START ***
	// Calculate h_1 and h_2
	params->mu = NULL;
	float* h_1 = 0;
	float* h_2 = 0;
	cudaMalloc((void**)&h_1, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&h_2, params->numAngles * params->numRows * params->numCols * sizeof(float));
	proj.project(h_1, dev_mu, params, false);
	scale(h_1, 0.5, N_g, params->whichGPU);
	equal(h_2, h_1, N_g, params->whichGPU);
	Hilbert1D(h_2, params, false);

	// Calculate g_a
	float* g_a_1 = 0;
	float* g_a_2 = 0;
	float* temp = 0;
	cudaMalloc((void**)&g_a_1, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&g_a_2, params->numAngles * params->numRows * params->numCols * sizeof(float));
	cudaMalloc((void**)&temp, params->numAngles * params->numRows * params->numCols * sizeof(float));

	// Calculate g_a_1
	equal(g_a_1, dev_g, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	cosFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	expFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	Hilbert1D(g_a_1, params, false);

	// Calculate g_a_2
	equal(g_a_2, dev_g, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	sinFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	expFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	Hilbert1D(g_a_2, params, false);

	// Combine g_a_1 and g_a_2
	float* g_a = dev_g;
	equal(temp, h_2, N_g, params->whichGPU);
	sinFcn(temp, N_g, params->whichGPU);
	multiply(g_a_2, temp, N_g, params->whichGPU);
	equal(temp, h_2, N_g, params->whichGPU);
	cosFcn(temp, N_g, params->whichGPU);
	multiply(g_a_1, temp, N_g, params->whichGPU);
	equal(g_a, g_a_2, N_g, params->whichGPU);
	add(g_a, g_a_1, N_g, params->whichGPU);
	equal(temp, h_1, N_g, params->whichGPU);
	negExpFcn(temp, N_g, params->whichGPU);
	multiply(g_a, temp, N_g, params->whichGPU);
	scale(g_a, FBPscalar(params), N_g, params->whichGPU);

	cudaFree(temp);
	cudaFree(h_1);
	cudaFree(h_2);
	cudaFree(g_a_1);
	cudaFree(g_a_2);
	params->mu = dev_mu;

	// Weighted Backprojection
	proj.weightedBackproject(dev_g, dev_f, params, false);
	params->mu = params_mu_save;

	if (cpu_to_gpu)
	{
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
		if (dev_f != 0)
			cudaFree(dev_f);
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_mu != 0)
			cudaFree(dev_mu);
	}
	return true;
}
