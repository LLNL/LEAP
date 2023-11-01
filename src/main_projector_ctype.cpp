////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "main_projector_ctype.h"
#include "parameters.h"
#include "projectors.h"
#include "projectors_SF.h"
#include "projectors_cpu.h"
#include "rampFilter.cuh"
#include "ray_weighting.cuh"
#include "noise_filters.cuh"
#include "total_variation.cuh"
#include "cuda_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef PI
#define PI 3.141592653589793
#endif

//#ifdef DEFINE_STATIC_UI
parameters params;

bool printParameters()
{
	params.printAll();
	return true;
}

bool reset()
{
	params.clearAll();
	params.setDefaults(1);
	return true;
}

bool project(float* g, float* f, bool cpu_to_gpu)
{
	if (params.allDefined() == false || g == NULL || f == NULL)
	{
		printf("ERROR: project: invalid parameters or invalid input arrays!\n");
		return false;
	}
	else if (params.whichGPU >= 0)
	{
		if (params.geometry == parameters::CONE)
        {
            if (params.useSF())
                return project_SF_cone(g, f, &params, cpu_to_gpu);
            else
                return project_cone(g, f, &params, cpu_to_gpu);
        }
		else if (params.geometry == parameters::PARALLEL)
        {
            if (params.useSF())
                return project_SF_parallel(g, f, &params, cpu_to_gpu);
            else
                return project_parallel(g, f, &params, cpu_to_gpu);
        }
		else
			return project_modular(g, f, &params, cpu_to_gpu);
	}
	else
	{
		if (params.geometry == parameters::CONE)
		{
			if (params.useSF())
				return CPUproject_SF_cone(g, f, &params);
			else
				return CPUproject_cone(g, f, &params);
		}
		else if (params.geometry == parameters::PARALLEL)
		{
			if (params.useSF())
				return CPUproject_SF_parallel(g, f, &params);
			else
				return CPUproject_parallel(g, f, &params);
		}
		else
			return CPUproject_modular(g, f, &params);
	}
}

bool backproject(float* g, float* f, bool cpu_to_gpu)
{
	if (params.allDefined() == false || g == NULL || f == NULL)
		return false;
	else if (params.whichGPU >= 0)
	{
		if (params.geometry == parameters::CONE)
        {
            if (params.useSF())
                return backproject_SF_cone(g, f, &params, cpu_to_gpu);
            else
                return backproject_cone(g, f, &params, cpu_to_gpu);
        }
		else if (params.geometry == parameters::PARALLEL)
        {
            if (params.useSF())
                return backproject_SF_parallel(g, f, &params, cpu_to_gpu);
            else
                return backproject_parallel(g, f, &params, cpu_to_gpu);
        }
		else
			return backproject_modular(g, f, &params, cpu_to_gpu);
	}
	else
	{
		if (params.geometry == parameters::CONE)
		{
			if (params.useSF())
				return CPUbackproject_SF_cone(g, f, &params);
			else
				return CPUbackproject_cone(g, f, &params);
		}
		else if (params.geometry == parameters::PARALLEL)
		{
			if (params.useSF())
				return CPUbackproject_SF_parallel(g, f, &params);
			else
				return CPUbackproject_parallel(g, f, &params);
		}
		else
			return CPUbackproject_modular(g, f, &params);
	}
}

bool rampFilterProjections(float* g, bool cpu_to_gpu, float scalar)
{
	if (params.whichGPU < 0)
	{
		printf("Error: ramp filter only implemented for GPU\n");
		return false;
	}
	return rampFilter1D(g, &params, cpu_to_gpu, scalar);
}

bool rampFilterVolume(float* f, bool cpu_to_gpu)
{
	if (params.whichGPU < 0)
	{
		printf("Error: ramp filter only implemented for GPU\n");
		return false;
	}
	return rampFilter2D(f, &params, cpu_to_gpu);
}

bool FBP(float* g, float* f, bool cpu_to_gpu)
{
	/*
	if (params.whichGPU < 0)
	{
		printf("Error: ramp filter only implemented for GPU\n");
		return false;
	}
	//*/
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

float get_FBPscalar()
{
	return FBPscalar(&params);
}

bool setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd)
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

bool setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
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

bool setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
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

bool setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
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

bool setDefaultVolumeParameters()
{
	return params.setDefaultVolumeParameters();
}

bool setVolumeDimensionOrder(int which)
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

int getVolumeDimensionOrder()
{
	return params.volumeDimensionOrder;
}

bool setGPU(int whichGPU)
{
	params.whichGPU = whichGPU;
	return true;
}

bool setProjector(int which)
{
    if (which == parameters::SEPARABLE_FOOTPRINT)
        params.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        params.whichProjector = 0;
    return true;
}

bool set_axisOfSymmetry(float axisOfSymmetry)
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

bool clear_axisOfSymmetry()
{
	params.axisOfSymmetry = 90.0;
	return true;
}

bool set_rFOV(float rFOV_in)
{
	params.rFOVspecified = rFOV_in;
	return true;
}

bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return project_SF_cone(g, f, &tempParams, cpu_to_gpu);
        else
            return project_cone(g, f, &tempParams, cpu_to_gpu);
    }
}

bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return backproject_SF_cone(g, f, &tempParams, cpu_to_gpu);
        else
            return backproject_cone(g, f, &tempParams, cpu_to_gpu);
    }
}

bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return project_SF_parallel(g, f, &tempParams, cpu_to_gpu);
        else
            return project_parallel(g, f, &tempParams, cpu_to_gpu);
    }
}

bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return backproject_SF_parallel(g, f, &tempParams, cpu_to_gpu);
        else
            return backproject_parallel(g, f, &tempParams, cpu_to_gpu);
    }
}

int get_numAngles()
{
	return params.numAngles;
}

int get_numRows()
{
	return params.numRows;
}

int get_numCols()
{
	return params.numCols;
}

int get_numX()
{
	return params.numX;
}

int get_numY()
{
	return params.numY;
}

int get_numZ()
{
	return params.numZ;
}

bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool cpu_to_gpu)
{
	return blurFilter(f, N_1, N_2, N_3, FWHM, 3, cpu_to_gpu, params.whichGPU);
}

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu)
{
	return medianFilter(f, N_1, N_2, N_3, threshold, cpu_to_gpu, params.whichGPU);
}

float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_cost(f, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_gradient(f, Df, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return anisotropicTotalVariation_quadraticForm(f, d, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu)
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

