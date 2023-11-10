////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "tomographic_models.h"
#include "parameters.h"
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

#ifndef PI
#define PI 3.141592653589793
#endif

tomographicModels::tomographicModels()
{
	params.setDefaults(1);
}

tomographicModels::~tomographicModels()
{
	reset();
}

bool tomographicModels::printParameters()
{
	params.printAll();
	return true;
}

bool tomographicModels::reset()
{
	params.clearAll();
	params.setDefaults(1);
	return true;
}

bool tomographicModels::project_gpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = project(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::backproject_gpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = backproject(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::project_cpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = project(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::backproject_cpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = backproject(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::project(float* g, float* f, parameters* ctParams, bool cpu_to_gpu)
{
	if (ctParams == NULL)
		return false;
	if (ctParams->allDefined() == false || g == NULL || f == NULL)
	{
		printf("ERROR: project: invalid parameters or invalid input arrays!\n");
		return false;
	}
	else if (ctParams->whichGPU >= 0)
	{
		if (cpu_to_gpu)
		{
			if (getAvailableGPUmemory(ctParams->whichGPU) < ctParams->projectionDataSize() + ctParams->volumeDataSize())
			{
				printf("Error: insufficient GPU memory\n");
				return false;
			}
		}

		if (ctParams->isSymmetric())
			return project_symmetric(g, f, ctParams, cpu_to_gpu);
		else
			return project_SF(g, f, ctParams, cpu_to_gpu);
	}
	else
	{
		if (ctParams->isSymmetric())
			return CPUproject_symmetric(g, f, ctParams);

		if (ctParams->geometry == parameters::CONE)
		{
			if (ctParams->useSF())
				return CPUproject_SF_cone(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based cone-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_cone(g, f, ctParams);
			}
		}
		else if (ctParams->geometry == parameters::FAN)
		{
			if (ctParams->useSF())
				return CPUproject_SF_fan(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based fan-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_fan(g, f, ctParams);
			}
		}
		else if (ctParams->geometry == parameters::PARALLEL)
		{
			if (ctParams->useSF())
				return CPUproject_SF_parallel(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based parallel-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUproject_parallel(g, f, ctParams);
			}
		}
		else
			return CPUproject_modular(g, f, ctParams);
	}
}

bool tomographicModels::backproject(float* g, float* f, parameters* ctParams, bool cpu_to_gpu)
{
	if (ctParams->allDefined() == false || g == NULL || f == NULL)
		return false;
	else if (ctParams->whichGPU >= 0)
	{
		if (cpu_to_gpu)
		{
			if (getAvailableGPUmemory(ctParams->whichGPU) < ctParams->projectionDataSize() + ctParams->volumeDataSize())
			{
				printf("Error: insufficient GPU memory\n");
				return false;
			}
		}

		if (ctParams->isSymmetric())
			return backproject_symmetric(g, f, ctParams, cpu_to_gpu);
		else
			return backproject_SF(g, f, ctParams, cpu_to_gpu);
	}
	else
	{
		if (ctParams->isSymmetric())
			return CPUbackproject_symmetric(g, f, ctParams);

		if (ctParams->geometry == parameters::CONE)
		{
			if (ctParams->useSF())
				return CPUbackproject_SF_cone(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based cone-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_cone(g, f, ctParams);
			}
		}
		else if (ctParams->geometry == parameters::FAN)
		{
			if (ctParams->useSF())
				return CPUbackproject_SF_fan(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based fan-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_fan(g, f, ctParams);
			}
		}
		else if (ctParams->geometry == parameters::PARALLEL)
		{
			if (ctParams->useSF())
				return CPUbackproject_SF_parallel(g, f, ctParams);
			else
			{
				printf("Error: The voxel size for CPU-based parallel-beam projectors must be closer to the nominal size.\n");
				printf("Please either change the voxel size or use the GPU-based projectors.");
				return false;
				//return CPUbackproject_parallel(g, f, ctParams);
			}
		}
		else
			return CPUbackproject_modular(g, f, ctParams);
	}
}

bool tomographicModels::project(float* g, float* f, bool cpu_to_gpu)
{
	return project(g, f, &params, cpu_to_gpu);
}

bool tomographicModels::backproject(float* g, float* f, bool cpu_to_gpu)
{
	return backproject(g, f, &params, cpu_to_gpu);
}

bool tomographicModels::rampFilterProjections(float* g, bool cpu_to_gpu, float scalar)
{
	if (params.whichGPU < 0)
		return rampFilter1D_cpu(g, &params, scalar);
	else
		return rampFilter1D(g, &params, cpu_to_gpu, scalar);
}

bool tomographicModels::rampFilterVolume(float* f, bool cpu_to_gpu)
{
	if (params.whichGPU < 0)
	{
		printf("Error: 2D ramp filter only implemented for GPU\n");
		return false;
	}
	return rampFilter2D(f, &params, cpu_to_gpu);
}

bool tomographicModels::FBP(float* g, float* f, bool cpu_to_gpu)
{
	if (params.geometry == parameters::MODULAR)
	{
		printf("Error: FBP not implemented for modular geometries\n");
		return false;
	}

	if (params.whichGPU < 0 || cpu_to_gpu == false)
	{
		// no transfers to/from GPU are necessary; just run the code
		applyPreRampFilterWeights(g, &params, cpu_to_gpu);
		rampFilterProjections(g, cpu_to_gpu, get_FBPscalar());
		applyPostRampFilterWeights(g, &params, cpu_to_gpu);
		return backproject(g, f, cpu_to_gpu);
	}
	else
	{
		if (getAvailableGPUmemory(params.whichGPU) < params.projectionDataSize() + params.volumeDataSize())
		{
			printf("Error: insufficient GPU memory\n");
			return false;
		}

		bool retVal = true;
		cudaSetDevice(params.whichGPU);
		cudaError_t cudaStatus;

		float* dev_g = copyProjectionDataToGPU(g, &params, params.whichGPU);
		if (dev_g == 0)
			return false;
		//printf("applyPreRampFilterWeights...\n");
		applyPreRampFilterWeights(dev_g, &params, false);
		//printf("rampFilterProjections...\n");
		rampFilterProjections(dev_g, false, get_FBPscalar());
		//printf("applyPostRampFilterWeights...\n");
		applyPostRampFilterWeights(dev_g, &params, false);

		float* dev_f = 0;
		if ((cudaStatus = cudaMalloc((void**)&dev_f, params.numX * params.numY * params.numZ * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
			retVal = false;
		}
		else
		{
			//printf("backproject...\n");
			retVal = backproject(dev_g, dev_f, false);
			pullVolumeDataFromGPU(f, &params, dev_f, params.whichGPU);
			cudaFree(dev_f);
		}

		if (dev_g != 0)
			cudaFree(dev_g);
		return retVal;
	}
}

float tomographicModels::get_FBPscalar()
{
	return FBPscalar(&params);
}

bool tomographicModels::setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd)
{
	params.geometry = parameters::CONE;
	params.detectorType = parameters::FLAT;
	params.sod = sod;
	params.sdd = sdd;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.setAngles(phis, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::setFanBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd)
{
	params.geometry = parameters::FAN;
	params.detectorType = parameters::FLAT;
	params.sod = sod;
	params.sdd = sdd;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.setAngles(phis, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	params.geometry = parameters::PARALLEL;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.setAngles(phis, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	params.geometry = parameters::MODULAR;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.setSourcesAndModules(sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	if (voxelWidth <= 0.0)
	{
		if (params.geometry == parameters::PARALLEL)
			voxelWidth = params.pixelWidth;
		else
			voxelWidth = params.pixelWidth * params.sod / params.sdd;
	}
	if (voxelHeight <= 0.0)
	{
		if (params.geometry == parameters::PARALLEL || params.geometry == parameters::FAN)
			voxelHeight = params.pixelHeight;
		else
			voxelHeight = params.pixelHeight * params.sod / params.sdd;
	}

	params.numX = numX;
	params.numY = numY;
	params.numZ = numZ;
	params.voxelWidth = voxelWidth;
	params.voxelHeight = voxelHeight;
	params.offsetX = offsetX;
	params.offsetY = offsetY;
	params.offsetZ = offsetZ;
	return params.volumeDefined();
}

bool tomographicModels::setDefaultVolumeParameters(float scale)
{
	return params.setDefaultVolumeParameters(scale);
}

bool tomographicModels::setVolumeDimensionOrder(int which)
{
	if (parameters::XYZ <= which && which <= parameters::ZYX)
	{
		if (which == parameters::ZYX && params.isSymmetric())
		{
			printf("Error: Symmetric objects can only be specified in XYZ order\n");
			return false;
		}
		else
		{
			params.volumeDimensionOrder = which;
			return true;
		}
	}
	else
	{
		printf("Error: volume dimension order must be 0 for XYZ or 1 for ZYX\n");
		return false;
	}
}

int tomographicModels::getVolumeDimensionOrder()
{
	return params.volumeDimensionOrder;
}

bool tomographicModels::setGPU(int whichGPU)
{
	if (numberOfGPUs() <= 0)
		params.whichGPU = -1;
	else
		params.whichGPU = whichGPU;
	return true;
}

int tomographicModels::getGPU()
{
	return params.whichGPU;
}

bool tomographicModels::setProjector(int which)
{
	if (which == parameters::SEPARABLE_FOOTPRINT)
		params.whichProjector = parameters::SEPARABLE_FOOTPRINT;
	else
	{
		printf("Error: currently only SF projectors are implemented!\n");
		return false;
		//params.whichProjector = 0;
	}
	return true;
}

bool tomographicModels::set_axisOfSymmetry(float axisOfSymmetry)
{
	if (params.volumeDimensionOrder == parameters::ZYX)
	{
		printf("Error: Symmetric objects can only be specified in XYZ order\n");
		return false;
	}
	else
	{
		params.axisOfSymmetry = axisOfSymmetry;
		return true;
	}
}

bool tomographicModels::clear_axisOfSymmetry()
{
	params.axisOfSymmetry = 90.0;
	return true;
}

bool tomographicModels::set_rFOV(float rFOV_in)
{
	params.rFOVspecified = rFOV_in;
	return true;
}

bool tomographicModels::set_rampID(int whichRampFilter)
{
	if (whichRampFilter < 0 || whichRampFilter > 10)
		return false;
	else
	{
		params.rampID = whichRampFilter;
		return true;
	}
}

bool tomographicModels::projectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::FAN;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return project(g, f, &tempParams, cpu_to_gpu);
}

bool tomographicModels::backprojectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::FAN;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return backproject(g, f, &tempParams, cpu_to_gpu);
}

bool tomographicModels::projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::CONE;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return project(g, f, &tempParams, cpu_to_gpu);
}

bool tomographicModels::backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::CONE;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return backproject(g, f, &tempParams, cpu_to_gpu);
}

bool tomographicModels::projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::PARALLEL;
	tempParams.detectorType = parameters::FLAT;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return project(g, f, &tempParams, cpu_to_gpu);
}

bool tomographicModels::backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::PARALLEL;
	tempParams.detectorType = parameters::FLAT;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return backproject(g, f, &tempParams, cpu_to_gpu);
}

int tomographicModels::get_numAngles()
{
	return params.numAngles;
}

int tomographicModels::get_numRows()
{
	return params.numRows;
}

int tomographicModels::get_numCols()
{
	return params.numCols;
}

int tomographicModels::get_numX()
{
	return params.numX;
}

int tomographicModels::get_numY()
{
	return params.numY;
}

int tomographicModels::get_numZ()
{
	return params.numZ;
}

bool tomographicModels::BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool cpu_to_gpu)
{
	return blurFilter(f, N_1, N_2, N_3, FWHM, 3, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu)
{
	return medianFilter(f, N_1, N_2, N_3, threshold, cpu_to_gpu, params.whichGPU);
}

float tomographicModels::TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_cost(f, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_gradient(f, Df, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

float tomographicModels::TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_quadraticForm(f, d, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu)
{
	return diffuse(f, N_1, N_2, N_3, delta, numIter, cpu_to_gpu, params.whichGPU);
}

/*
// Scanner Parameters
int geometry;
int detectorType;
float sod, sdd;
float pixelWidth, pixelHeight, angularRange;
int numCols, numRows, numAngles;
float centerCol, centerRow;
float* phis;

// Volume Parameters
int volumeDimensionOrder;
int numX, numY, numZ;
float voxelWidth, voxelHeight;
float offsetX, offsetY, offsetZ;
//*/
