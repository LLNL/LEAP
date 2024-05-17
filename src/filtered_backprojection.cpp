////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Class for performing analytic inversion, i.e., Filtered Backprojection (FBP)
// algorithms.
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "filtered_backprojection.h"
#include "projectors.h"
#include "ray_weighting_cpu.h"
#include "projectors_symmetric_cpu.h"
#include "ramp_filter_cpu.h"
#ifndef __USE_CPU
#include "cuda_utils.h"
#include "ramp_filter.cuh"
#include "ray_weighting.cuh"
#include "projectors_symmetric.cuh"
#include "projectors_attenuated.cuh"
#endif

filteredBackprojection::filteredBackprojection()
{

}

filteredBackprojection::~filteredBackprojection()
{

}

bool filteredBackprojection::HilbertFilterProjections(float* g, parameters* params, bool data_on_cpu, float scalar, float sampleShift)
{
	return convolve1D(g, params, data_on_cpu, scalar, 1, sampleShift);
}

bool filteredBackprojection::rampFilterProjections(float* g, parameters* params, bool data_on_cpu, float scalar)
{
	return convolve1D(g, params, data_on_cpu, scalar, 0);
}

bool filteredBackprojection::convolve1D(float* g, parameters* params, bool data_on_cpu, float scalar, int whichFilter, float sampleShift)
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
#ifndef __USE_CPU
		if (params->isSymmetric())
		{
			if (data_on_cpu)
			{
				float* g_left = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
				float* g_right = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
				splitLeftAndRight(g, g_left, g_right, params);
				bool retVal = rampFilter1D(g_left, params, data_on_cpu, scalar);
				rampFilter1D(g_right, params, data_on_cpu, scalar);
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
				ray_derivative(g, params, data_on_cpu, scalar, 1.0);
				return Hilbert1D(g, params, data_on_cpu, 1.0, -1.0);
				//*/
				return rampFilter1D(g, params, data_on_cpu, scalar);
			}
			else
				return Hilbert1D(g, params, data_on_cpu, scalar, sampleShift);
		}
#else
		return false;
#endif
	}
}

bool filteredBackprojection::filterProjections(float* g, parameters* params, bool data_on_cpu)
{
	if (params->geometry == parameters::MODULAR && params->modularbeamIsAxiallyAligned() == false)
	{
		printf("Error: projection filtering only implemented for modular geometries whose rowVectors are aligned with the z-axis\n");
		return false;
	}

	if (params->muSpecified())
		return filterProjections_Novikov(g, params, data_on_cpu);

	if (params->whichGPU < 0 || data_on_cpu == false)
	{
		// no transfers to/from GPU are necessary; just run the code
		if (params->geometry == parameters::CONE && params->helicalPitch != 0.0)
		{
#ifndef __USE_CPU
			parallelRay_derivative(g, params, false);
			applyPostRampFilterWeights(g, params, false);
			return HilbertFilterProjections(g, params, false, FBPscalar(params), -1.0);
#else
			printf("Error: helical FBP filtering is only implemented on the GPU at this time!\n");
			return false;
#endif
		}
		else
		{
#ifndef __USE_CPU
			applyPreRampFilterWeights(g, params, data_on_cpu);
			if (params->muCoeff != 0.0)
				convertARTtoERT(g, params, false, false);
			if (params->inconsistencyReconstruction == true && params->angularRange >= 358.0)
			{
				if (params->whichGPU < 0)
					ray_derivative_cpu(g, params);
				else
					ray_derivative(g, params, data_on_cpu);
			}
			else
				rampFilterProjections(g, params, data_on_cpu, FBPscalar(params));
			if (params->muCoeff != 0.0)
				convertARTtoERT(g, params, false, true);
			return applyPostRampFilterWeights(g, params, data_on_cpu);
#else
			applyPreRampFilterWeights_CPU(g, params);
			if (params->muCoeff != 0.0)
				convertARTtoERT_CPU(g, params, false);
			if (params->inconsistencyReconstruction == true && params->angularRange >= 358.0)
			{
				ray_derivative_cpu(g, params);
			}
			else
				rampFilterProjections(g, params, data_on_cpu, FBPscalar(params));
			if (params->muCoeff != 0.0)
				convertARTtoERT_CPU(g, params, true);
			return applyPostRampFilterWeights_CPU(g, params);
#endif
		}
	}
	else
	{
#ifndef __USE_CPU
		if (getAvailableGPUmemory(params->whichGPU) < params->projectionDataSize())
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		bool retVal = true;

		cudaSetDevice(params->whichGPU);
		//cudaError_t cudaStatus;

		float* dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		if (dev_g == 0)
			return false;
		retVal = filterProjections(dev_g, params, false);

		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

		if (dev_g != 0)
			cudaFree(dev_g);
		return retVal;
#else
		return false;
#endif
	}
}

bool filteredBackprojection::execute(float* g, float* f, parameters* params, bool data_on_cpu)
{
	if (params->geometry == parameters::MODULAR)
	{
		if (params->modularbeamIsAxiallyAligned() == false || params->min_T_phi() < 1.0e-16)
		{
			printf("Error: FBP only implemented for modular geometries whose rowVectors are aligned with the z-axis\n");
			return false;
		}
	}
	if (params->geometry == parameters::CONE && params->helicalPitch != 0.0 && params->whichGPU < 0)
	{
		printf("Error: CPU-based FBP not yet implemented for helical cone-beam geometry\n");
		return false;
	}

	if (params->muSpecified())
		return execute_attenuated(g, f, params, data_on_cpu);

	if (params->whichGPU < 0 || data_on_cpu == false)
	{
		float* g_pad = NULL;
		float* g_save = g;
		if (params->offsetScan)
		{
			/*
			if (params->whichGPU >= 0 && data_on_cpu == false)
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
		//printf("WARNING: disabling filtering in FBP for debugging purposes!!!!\n");
		filterProjections(g, params, false);

		bool retVal = true;
		bool doWeightedBackprojection_save = params->doWeightedBackprojection;
		params->doWeightedBackprojection = true;
		bool doExtrapolation_save = params->doExtrapolation;
		if (params->geometry != parameters::CONE || params->helicalPitch == 0.0)
			params->doExtrapolation = true;
		if (params->isSymmetric())
		{
#ifndef __USE_CPU
			if (params->whichGPU < 0)
				retVal = CPUinverse_symmetric(g, f, params);
			else
				retVal = inverse_symmetric(g, f, params, data_on_cpu);
#else
			retVal = CPUinverse_symmetric(g, f, params);
#endif
		}
		else
			retVal = proj.backproject(g, f, params, data_on_cpu);
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
#ifndef __USE_CPU
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
		if ((cudaStatus = cudaMalloc((void**)&dev_f, uint64(params->numX) * uint64(params->numY) * uint64(params->numZ) * sizeof(float))) != cudaSuccess)
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
#else
		return false;
#endif
	}
}

bool filteredBackprojection::execute_attenuated(float* g, float* f, parameters* params, bool data_on_cpu)
{
	if (params->mu != NULL)
		return execute_Novikov(g, f, params, data_on_cpu);
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * fabs(params->T_phi())*180.0/PI)
	{
		printf("Error: FBP of attenuated x-ray transform requires at least 360 degree angular range!\n");
		return false;
	}
	if (params->whichGPU < 0 || data_on_cpu == false)
	{
		// data transfers to/from GPU are not necessary
		// don't need memory checks

#ifndef __USE_CPU
		applyPreRampFilterWeights(g, params, data_on_cpu);
		convertARTtoERT(g, params, false, false);
		rampFilterProjections(g, params, false, FBPscalar(params));
		convertARTtoERT(g, params, false, true);
#else
		applyPreRampFilterWeights_CPU(g, params);
		convertARTtoERT_CPU(g, params, false);
		rampFilterProjections(g, params, false, FBPscalar(params));
		convertARTtoERT_CPU(g, params, true);
#endif

		params->muCoeff *= -1.0;
		bool retVal = proj.backproject(g, f, params, false);
		params->muCoeff *= -1.0;
		return retVal;
	}
	else
	{
#ifndef __USE_CPU
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
#else
		return false;
#endif
	}
}

bool filteredBackprojection::filterProjections_Novikov(float* g, parameters* params, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * fabs(params->T_phi())*180.0/PI)
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
	if (data_on_cpu)
	{
		if (params->hasSufficientGPUmemory() == false)
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		cudaSetDevice(params->whichGPU);
		//cudaError_t cudaStatus;

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
	if (data_on_cpu)
	{
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_mu != 0)
			cudaFree(dev_mu);
	}
	return true;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool filteredBackprojection::execute_Novikov(float* g, float* f, parameters* params, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params->geometry != parameters::PARALLEL)
	{
		printf("Error: FBP of attenuated x-ray transform only implemented for parallel-beam data!\n");
		return false;
	}
	if (params->angularRange < 360.0 - 0.5 * fabs(params->T_phi()) * 180.0 / PI)
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
	if (data_on_cpu)
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

	if (data_on_cpu)
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
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}
