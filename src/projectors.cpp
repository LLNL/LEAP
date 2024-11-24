////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// projectors class which selects the appropriate projector to use based on the
// CT geometry and CT volume specification, and some other parameters such as
// whether the calculation should happen on the CPU or GPU
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors.h"
#include "projectors_SF_cpu.h"
#include "projectors_Joseph_cpu.h"
#include "projectors_Siddon_cpu.h"
#include "cuda_utils.h"
#include "projectors_symmetric_cpu.h"
#include "projectors_SF.cuh"
#ifndef __USE_CPU
#include "projectors_symmetric.cuh"
#include "projectors_attenuated.cuh"
#include "projectors_Joseph.cuh"
#include "backprojectors_VD.cuh"
#endif

projectors::projectors()
{

}

projectors::~projectors()
{

}

bool projectors::project(float* g, float* f, parameters* params, bool data_on_cpu)
{
	return project(g, f, params, data_on_cpu, data_on_cpu);
}

bool projectors::project(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate)
{
	if (params == NULL)
		return false;
	if (params->allDefined() == false || g == NULL || f == NULL)
	{
		printf("ERROR: project: invalid parameters or invalid input arrays!\n");
		return false;
	}
#ifndef __USE_CPU
	else if (params->whichGPU >= 0)
	{
		int numVolumeData = 1;
		if (volume_on_cpu == false)
			numVolumeData = 0;
		int numProjectionData = 1;
		if (data_on_cpu == false)
			numProjectionData = 0;

		if (data_on_cpu)
		{
			if (params->hasSufficientGPUmemory(false, 0, numProjectionData, numVolumeData) == false)
			{
				printf("Error: insufficient GPU memory\n");
				return false;
			}
		}

		if (params->isSymmetric())
			return project_symmetric(g, f, params, data_on_cpu);
		else if (params->muSpecified())
			return project_attenuated(g, f, params, data_on_cpu);
		else if (params->geometry == parameters::MODULAR)
			return project_Joseph_modular(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
		else
			return project_SF(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
	}
#endif
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
		else if (params->geometry == parameters::MODULAR)
			return project_Joseph_modular_cpu(g, f, params);
		else
		{
			printf("Error: CPU-based projector not yet implemented for this geometry.\n");
			return false;
		}
	}
}

bool projectors::backproject(float* g, float* f, parameters* params, bool data_on_cpu)
{
	return backproject(g, f, params, data_on_cpu, data_on_cpu);
}

bool projectors::backproject(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate)
{
	if (params->allDefined() == false || g == NULL || f == NULL)
		return false;
#ifndef __USE_CPU
	else if (params->whichGPU >= 0)
	{
		int numVolumeData = 1;
		if (volume_on_cpu == false)
			numVolumeData = 0;
		int numProjectionData = 1;
		if (data_on_cpu == false)
			numProjectionData = 0;

		if (data_on_cpu)
		{
			if (params->hasSufficientGPUmemory(false, 0, numProjectionData, numVolumeData) == false)
			{
				printf("Error: insufficient GPU memory\n");
				printf("available memory: %f\n", getAvailableGPUmemory(params->whichGPU));
				printf("required memory: %f\n", params->requiredGPUmemory(0, numProjectionData, numVolumeData));

				return false;
			}
		}

		if (params->isSymmetric())
			return backproject_symmetric(g, f, params, data_on_cpu);
		else if (params->muSpecified())
			return backproject_attenuated(g, f, params, data_on_cpu);
		else if (params->whichProjector == parameters::VOXEL_DRIVEN /* && (params->helicalPitch == 0.0 || params->doWeightedBackprojection == false)*/)
			return backproject_VD(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
		else if (params->geometry == parameters::MODULAR)
			return backproject_Joseph_modular(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
		else
			return backproject_SF(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
	}
#endif
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
		else if (params->geometry == parameters::MODULAR)
			return backproject_Joseph_modular_cpu(g, f, params);
		else
		{
			printf("Error: CPU-based backprojector not yet implemented for this geometry.\n");
			return false;
		}
	}
}

bool projectors::weightedBackproject(float* g, float* f, parameters* params, bool data_on_cpu)
{
	return weightedBackproject(g, f, params, data_on_cpu, data_on_cpu);
}

bool projectors::weightedBackproject(float* g, float* f, parameters* params, bool data_on_cpu, bool volume_on_cpu, bool accumulate)
{
	bool doWeightedBackprojection_save = params->doWeightedBackprojection;
	params->doWeightedBackprojection = true;
	bool doExtrapolation_save = params->doExtrapolation;
	params->doExtrapolation = true;
	bool retVal = backproject(g, f, params, data_on_cpu, volume_on_cpu, accumulate);
	params->doWeightedBackprojection = doWeightedBackprojection_save;
	params->doExtrapolation = doExtrapolation_save;
	return retVal;
}
