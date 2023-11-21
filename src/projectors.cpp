////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "filtered_backprojection.h"
#include "projectors.h"
#include "projectors_SF.h"
#include "projectors_cpu.h"
#include "ramp_filter.cuh"
#include "ray_weighting.cuh"
#include "noise_filters.cuh"
#include "total_variation.cuh"
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

projectors::projectors()
{

}

projectors::~projectors()
{

}

bool projectors::project(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (params == NULL)
		return false;
	if (params->allDefined() == false || g == NULL || f == NULL)
	{
		printf("ERROR: project: invalid parameters or invalid input arrays!\n");
		return false;
	}
	else if (params->whichGPU >= 0)
	{
		if (cpu_to_gpu)
		{
			if (params->hasSufficientGPUmemory() == false)
			{
				printf("Error: insufficient GPU memory\n");
				return false;
			}
		}

		if (params->isSymmetric())
			return project_symmetric(g, f, params, cpu_to_gpu);
		else if (params->muSpecified())
			return project_attenuated(g, f, params, cpu_to_gpu);
		else
			return project_SF(g, f, params, cpu_to_gpu);
	}
	else
	{
		if (params->isSymmetric())
			return CPUproject_symmetric(g, f, params);

		if (params->geometry == parameters::CONE)
		{
			if (params->useSF())
				return CPUproject_SF_cone(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based cone-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_cone(g, f, params);
			}
		}
		else if (params->geometry == parameters::FAN)
		{
			if (params->useSF())
				return CPUproject_SF_fan(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based fan-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_fan(g, f, params);
			}
		}
		else if (params->geometry == parameters::PARALLEL)
		{
			if (params->useSF())
				return CPUproject_SF_parallel(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based parallel-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_parallel(g, f, params);
			}
		}
		else
			return CPUproject_modular(g, f, params);
	}
}

bool projectors::backproject(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (params->allDefined() == false || g == NULL || f == NULL)
		return false;
	else if (params->whichGPU >= 0)
	{
		if (cpu_to_gpu)
		{
			if (params->hasSufficientGPUmemory() == false)
			{
				printf("Error: insufficient GPU memory\n");
				return false;
			}
		}

		if (params->isSymmetric())
			return backproject_symmetric(g, f, params, cpu_to_gpu);
		else if (params->muSpecified())
			return backproject_attenuated(g, f, params, cpu_to_gpu);
		else
			return backproject_SF(g, f, params, cpu_to_gpu);
	}
	else
	{
		if (params->mu != NULL)
		{
			printf("Error: attenuated radon transform for voxelize attenuation map only works for GPU projectors!\n");
			return false;
		}

		if (params->isSymmetric())
			return CPUbackproject_symmetric(g, f, params);

		if (params->geometry == parameters::CONE)
		{
			if (params->useSF())
				return CPUbackproject_SF_cone(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based cone-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_cone(g, f, params);
			}
		}
		else if (params->geometry == parameters::FAN)
		{
			if (params->useSF())
				return CPUbackproject_SF_fan(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based fan-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_fan(g, f, params);
			}
		}
		else if (params->geometry == parameters::PARALLEL)
		{
			if (params->useSF())
				return CPUbackproject_SF_parallel(g, f, params);
			else
			{
				printf("Error: The voxel size for CPU-based parallel-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_parallel(g, f, params);
			}
		}
		else
			return CPUbackproject_modular(g, f, params);
	}
}

bool projectors::weightedBackproject(float* g, float* f, parameters* params, bool cpu_to_gpu)
{
	bool doWeightedBackprojection_save = params->doWeightedBackprojection;
	params->doWeightedBackprojection = true;
	bool retVal = backproject(g, f, params, cpu_to_gpu);
	params->doWeightedBackprojection = doWeightedBackprojection_save;
	return retVal;
}
