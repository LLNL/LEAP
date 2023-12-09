////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "tomographic_models.h"
#include "ray_weighting.cuh"
#include "ramp_filter_cpu.h"
#include "ramp_filter.cuh"
#include "noise_filters.cuh"
#include "total_variation.cuh"
#include "cuda_utils.h"
#include "sensitivity_cpu.h"
#include "sensitivity.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>

#ifndef PI
#define PI 3.141592653589793
#endif

tomographicModels::tomographicModels()
{
	params.initialize();
	maxSlicesForChunking = 128;
}

tomographicModels::~tomographicModels()
{
	reset();
}

bool tomographicModels::print_parameters()
{
	params.printAll();
	return true;
}

bool tomographicModels::reset()
{
	params.clearAll();
	params.initialize();
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

bool tomographicModels::project(float* g, float* f, bool cpu_to_gpu)
{
	if (cpu_to_gpu == true && project_multiGPU(g, f) == true)
		return true;
	else
		return proj.project(g, f, &params, cpu_to_gpu);
}

bool tomographicModels::backproject(float* g, float* f, bool cpu_to_gpu)
{
	if (cpu_to_gpu == true && backproject_multiGPU(g, f) == true)
		return true;
	else
		return proj.backproject(g, f, &params, cpu_to_gpu);
}

bool tomographicModels::weightedBackproject(float* g, float* f, bool cpu_to_gpu)
{
	bool doWeight_save = params.doWeightedBackprojection;
	params.doWeightedBackprojection = true;
	bool retVal = backproject(g, f, cpu_to_gpu);
	params.doWeightedBackprojection = doWeight_save;
	return retVal;
}

bool tomographicModels::HilbertFilterProjections(float* g, bool cpu_to_gpu, float scalar)
{
	return FBP.HilbertFilterProjections(g, &params, cpu_to_gpu, scalar);
}

bool tomographicModels::rampFilterProjections(float* g, bool cpu_to_gpu, float scalar)
{
	return FBP.rampFilterProjections(g, &params, cpu_to_gpu, scalar);
}

bool tomographicModels::filterProjections(float* g, bool cpu_to_gpu)
{
	return FBP.filterProjections(g, &params, cpu_to_gpu);
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

float* tomographicModels::copyRows(float* g, int firstSlice, int lastSlice)
{
	int numSlices = lastSlice - firstSlice + 1;
	float* g_chunk = (float*)malloc(sizeof(float) * uint64(params.numAngles) * uint64(params.numCols) * uint64(numSlices));

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = 0; iphi < params.numAngles; iphi++)
	{
		float* g_proj = &g[uint64(iphi) * uint64(params.numRows * params.numCols)];
		float* g_chunk_proj = &g_chunk[uint64(iphi) * uint64(numSlices * params.numCols)];
		for (int iRow = firstSlice; iRow <= lastSlice; iRow++)
		{
			float* g_line = &g_proj[iRow * params.numCols];
			float* g_chunk_line = &g_chunk_proj[(iRow - firstSlice) * params.numCols];
			for (int iCol = 0; iCol < params.numCols; iCol++)
				g_chunk_line[iCol] = g_line[iCol];
		}
	}
	return g_chunk;
}

bool tomographicModels::combineRows(float* g, float* g_chunk, int firstRow, int lastRow)
{
	int numRows = lastRow - firstRow + 1;
	
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = 0; iphi < params.numAngles; iphi++)
	{
		float* g_proj = &g[uint64(iphi) * uint64(params.numRows * params.numCols)];
		float* g_chunk_proj = &g_chunk[uint64(iphi) * uint64(numRows * params.numCols)];
		for (int iRow = firstRow; iRow <= lastRow; iRow++)
		{
			float* g_line = &g_proj[iRow * params.numCols];
			float* g_chunk_line = &g_chunk_proj[(iRow - firstRow) * params.numCols];
			for (int iCol = 0; iCol < params.numCols; iCol++)
				g_line[iCol] = g_chunk_line[iCol];
		}
	}
	return true;
}

bool tomographicModels::backproject_multiGPU(float* g, float* f)
{
	return backproject_FBP_multiGPU(g, f, false);
}

bool tomographicModels::FBP_multiGPU(float* g, float* f)
{
	return backproject_FBP_multiGPU(g, f, true);
}

bool tomographicModels::project_multiGPU(float* g, float* f)
{
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;
	if ((params.geometry == parameters::CONE && params.helicalPitch != 0.0) || params.geometry == parameters::MODULAR)
		return project_multiGPU_splitViews(g, f);

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numRowsPerChunk = std::min(64, params.numRows);
	int numRowsPerChunk = std::max(1, int(ceil(float(params.numRows) / std::max(2.0, double(params.whichGPUs.size())) )));
	numRowsPerChunk = std::min(numRowsPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	if (params.hasSufficientGPUmemory(true) == false)
	{
		float memAvailable = getAvailableGPUmemory(params.whichGPUs);
		float memNeeded = project_memoryRequired(numRowsPerChunk);

		while (memAvailable < memNeeded)
		{
			numRowsPerChunk = numRowsPerChunk / 2;
			if (numRowsPerChunk <= 1)
				return false;
			memNeeded = project_memoryRequired(numRowsPerChunk);
		}
		numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory() <= params.chunkingMemorySizeThreshold)
		return false;
	else
	{
		numRowsPerChunk = int(ceil(float(params.numRows) / float(params.whichGPUs.size())));
		numRowsPerChunk = std::min(numRowsPerChunk, maxSlicesForChunking);
		numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	}

	//printf("numRowsPerChunk = %d\n", numRowsPerChunk);

	if (numChunks <= 1)
		return false;

	if (params.geometry != parameters::FAN && params.geometry != parameters::PARALLEL && params.geometry != parameters::CONE)
		return false;

	omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstRow = ichunk * numRowsPerChunk;
		int lastRow = std::min(firstRow + numRowsPerChunk - 1, params.numRows - 1);
		int numRows = lastRow - firstRow + 1;

		int sliceRange[2];
		params.sliceRangeNeededForProjection(firstRow, lastRow, sliceRange);
		int numSlices = sliceRange[1] - sliceRange[0] + 1;

		//printf("row range: %d to %d\n", firstRow, lastRow);
		//printf("slices range: %d to %d\n", sliceRange[0], sliceRange[1]);

		float* f_chunk = &f[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

		// make a copy of the relavent rows
		float* g_chunk = (float*)malloc(sizeof(float) * params.numAngles * params.numCols * numRows);

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.numRows = numRows;
		chunk_params.numZ = numSlices;
		//chunk_params.offsetZ = params.offsetZ + (sliceRange[0] - firstRow) * params.voxelHeight;
		chunk_params.centerRow = params.centerRow - firstRow;

		// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
		chunk_params.offsetZ += sliceRange[0] * params.voxelHeight + params.z_0() - chunk_params.z_0();

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();
		if (params.mu != NULL)
			chunk_params.mu = &params.mu[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

		// Do Computation
		proj.project(g_chunk, f_chunk, &chunk_params, true);
		//printf("about to combine...\n");
		combineRows(g, g_chunk, firstRow, lastRow);
		//printf("combine done\n");

		// clean up
		free(g_chunk);

	}
	return true;
}

float tomographicModels::project_memoryRequired(int numRowsPerChunk)
{
	float maxMemory = 0.0;

	int numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstRow = ichunk * numRowsPerChunk;
		int lastRow = std::min(firstRow + numRowsPerChunk - 1, params.numRows - 1);
		int numRows = lastRow - firstRow + 1;

		int sliceRange[2];
		params.sliceRangeNeededForProjection(firstRow, lastRow, sliceRange);
		int numSlices = sliceRange[1] - sliceRange[0] + 1;

		float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numRows) / float(params.numRows) * params.projectionDataSize();
		maxMemory = std::max(maxMemory, memoryNeeded);
	}

	return maxMemory + params.get_extraMemoryReserved();
}

float tomographicModels::project_memoryRequired_splitViews(int numViewsPerChunk)
{
	float maxMemory = 0.0;

	int numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstView = ichunk * numViewsPerChunk;
		int lastView = std::min(firstView + numViewsPerChunk - 1, params.numAngles - 1);
		int numViews = lastView - firstView + 1;

		int sliceRange[2];
		params.sliceRangeNeededForProjectionRange(firstView, lastView, sliceRange);
		int numSlices = sliceRange[1] - sliceRange[0] + 1;

		float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
		maxMemory = std::max(maxMemory, memoryNeeded);
	}
	return maxMemory + params.get_extraMemoryReserved();
}

bool tomographicModels::project_multiGPU_splitViews(float* g, float* f)
{
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numRowsPerChunk = std::min(64, params.numRows);
	int numViewsPerChunk = std::max(1, int(ceil(float(params.numAngles) / std::max(2.0, double(params.whichGPUs.size())))));
	//numViewsPerChunk = std::min(numViewsPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
	if (params.hasSufficientGPUmemory(true) == false)
	{
		// FIXME: this does not properly calculate the amount of memory necessary
		float memAvailable = getAvailableGPUmemory(params.whichGPUs);
		float memNeeded = project_memoryRequired_splitViews(numViewsPerChunk);

		while (memAvailable < memNeeded)
		{
			numViewsPerChunk = numViewsPerChunk / 2;
			if (numViewsPerChunk <= 1)
				return false;
			memNeeded = project_memoryRequired_splitViews(numViewsPerChunk);
		}
		numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory() <= params.chunkingMemorySizeThreshold)
		return false;
	else
	{
		numViewsPerChunk = int(ceil(float(params.numAngles) / float(params.whichGPUs.size())));
		//numViewsPerChunk = std::min(numViewsPerChunk, maxSlicesForChunking);
		numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
	}

	if (numChunks <= 1)
		return false;

	omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstView = ichunk * numViewsPerChunk;
		int lastView = std::min(firstView + numViewsPerChunk - 1, params.numAngles - 1);
		int numViews = lastView - firstView + 1;

		int sliceRange[2];
		params.sliceRangeNeededForProjectionRange(firstView, lastView, sliceRange);
		int numSlices = sliceRange[1] - sliceRange[0] + 1;

		float* f_chunk = &f[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

		// make a copy of the relavent rows
		float* g_chunk = &g[uint64(firstView)* uint64(params.numRows*params.numCols)];

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.removeProjections(firstView, lastView);
		chunk_params.numZ = numSlices;

		// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
		chunk_params.offsetZ += sliceRange[0] * params.voxelHeight + params.z_0() - chunk_params.z_0();

		/*
		chunk_params.offsetZ = params.offsetZ + (sliceRange[0] - 0) * params.voxelHeight; // FIXME?
		if (params.helicalPitch != 0.0)
			chunk_params.offsetZ += float(chunk_params.numZ - params.numZ) * 0.5 * params.voxelHeight;
		//*/

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();
		if (params.mu != NULL)
			chunk_params.mu = &params.mu[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

		//printf("full numAngles = %d, chunk numAngles = %d\n", params.numAngles, chunk_params.numAngles);
		//printf("GPU %d: view range: (%d, %d)    slice range: (%d, %d)\n", chunk_params.whichGPU, firstView, lastView, sliceRange[0], sliceRange[1]);

		// Do Computation
		proj.project(g_chunk, f_chunk, &chunk_params, true);
	}
	return true;
}

bool tomographicModels::backproject_FBP_multiGPU(float* g, float* f, bool doFBP)
{
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;
	if ((params.geometry == parameters::CONE && params.helicalPitch != 0.0) || params.geometry == parameters::MODULAR)
		return backproject_FBP_multiGPU_splitViews(g, f, doFBP);

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numSlicesPerChunk = std::min(64, params.numZ);
	int numSlicesPerChunk = std::max(1, int(ceil(float(params.numZ) / std::max(2.0, double(params.whichGPUs.size())))));
	numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	if (params.hasSufficientGPUmemory(true) == false)
	{
		float memAvailable = getAvailableGPUmemory(params.whichGPUs);
		float memNeeded = backproject_memoryRequired(numSlicesPerChunk);

		while (memAvailable < memNeeded)
		{
			numSlicesPerChunk = numSlicesPerChunk / 2;
			if (numSlicesPerChunk <= 1)
				return false;
			memNeeded = backproject_memoryRequired(numSlicesPerChunk);
		}
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory() <= params.chunkingMemorySizeThreshold)
		return false;
	else
	{
		numSlicesPerChunk = int(ceil(float(params.numZ) / float(params.whichGPUs.size())));
		numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}

	if (numChunks <= 1)
		return false;

	//if (params.geometry != parameters::FAN && params.geometry != parameters::PARALLEL && params.geometry != parameters::CONE)
	//	return false;

	omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ-1);
		int numSlices = lastSlice - firstSlice + 1;

		float* f_chunk = &f[uint64(firstSlice) * uint64(params.numX * params.numY)];

		// make a copy of the relavent rows
		int rowRange[2];
		params.rowRangeNeededForBackprojection(firstSlice, lastSlice, rowRange);
		float* g_chunk = NULL;
		if (rowRange[0] == 0 && rowRange[1] == params.numRows - 1)
			g_chunk = g;
		else
			g_chunk = copyRows(g, rowRange[0], rowRange[1]);

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.numRows = rowRange[1] - rowRange[0] + 1;
		chunk_params.numZ = numSlices;
		//chunk_params.offsetZ = params.offsetZ + (firstSlice - rowRange[0]) * params.voxelHeight;
		//if (params.geometry == parameters::MODULAR)
		//	chunk_params.offsetZ += 0.5*(chunk_params.numZ - params.numZ)* params.voxelHeight;
		chunk_params.centerRow = params.centerRow - rowRange[0];
		if (params.mu != NULL)
			chunk_params.mu = &params.mu[uint64(firstSlice) * uint64(params.numX * params.numY)];

		// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
		chunk_params.offsetZ += firstSlice * params.voxelHeight + params.z_0() - chunk_params.z_0();

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();

		//printf("z_0: %f to %f\n", params.z_0(), chunk_params.z_0());
		//printf("slices: (%d, %d); rows: (%d, %d); GPU = %d...\n", firstSlice, lastSlice, rowRange[0], rowRange[1], chunk_params.whichGPU);
		//float magFactor = 1.0;
		//if (chunk_params.geometry == parameters::CONE)
		//	magFactor = chunk_params.sod / chunk_params.sdd;
		//printf("slices: (%f, %f); rows: (%f, %f); GPU = %d...\n", chunk_params.z_samples(0), chunk_params.z_samples(chunk_params.numZ-1), magFactor*chunk_params.v(0), magFactor*chunk_params.v(chunk_params.numRows-1), chunk_params.whichGPU);

		//* Do Computation
		if (doFBP)
			FBP.execute(g_chunk, f_chunk, &chunk_params, true);
		else
			proj.backproject(g_chunk, f_chunk, &chunk_params, true);
		//*/

		// clean up
		if (g_chunk != g)
			free(g_chunk);
		
	}
	//printf("done\n");
	return true;
}

float tomographicModels::backproject_memoryRequired(int numSlicesPerChunk)
{
	float maxMemory = 0.0;

	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
		int numSlices = lastSlice - firstSlice + 1;

		int rowRange[2];
		params.rowRangeNeededForBackprojection(firstSlice, lastSlice, rowRange);
		int numRows = rowRange[1] - rowRange[0] + 1;

		float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numRows) / float(params.numRows) * params.projectionDataSize();
		maxMemory = std::max(maxMemory, memoryNeeded);
	}
	return maxMemory + params.get_extraMemoryReserved();
}

float tomographicModels::backproject_memoryRequired_splitViews(int numSlicesPerChunk)
{
	float maxMemory = 0.0;

	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
		int numSlices = lastSlice - firstSlice + 1;

		int viewRange[2];
		params.viewRangeNeededForBackprojection(firstSlice, lastSlice, viewRange);
		int numViews = viewRange[1] - viewRange[0] + 1;

		float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
		maxMemory = std::max(maxMemory, memoryNeeded);
	}
	return maxMemory + params.get_extraMemoryReserved();
}

bool tomographicModels::backproject_FBP_multiGPU_splitViews(float* g, float* f, bool doFBP)
{
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numSlicesPerChunk = std::min(64, params.numZ);
	int numSlicesPerChunk = std::max(1, int(ceil(float(params.numZ) / std::max(2.0, double(params.whichGPUs.size())))));
	numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	if (params.hasSufficientGPUmemory(true) == false)
	{
		float memAvailable = getAvailableGPUmemory(params.whichGPUs);
		float memNeeded = backproject_memoryRequired_splitViews(numSlicesPerChunk);

		while (memAvailable < memNeeded)
		{
			numSlicesPerChunk = numSlicesPerChunk / 2;
			if (numSlicesPerChunk <= 1)
				return false;
			memNeeded = backproject_memoryRequired_splitViews(numSlicesPerChunk);
		}
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory() <= params.chunkingMemorySizeThreshold)
		return false;
	else
	{
		numSlicesPerChunk = int(ceil(float(params.numZ) / float(params.whichGPUs.size())));
		numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}

	if (numChunks <= 1)
		return false;

	//if (params.geometry != parameters::FAN && params.geometry != parameters::PARALLEL && params.geometry != parameters::CONE)
	//	return false;

	omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
		int numSlices = lastSlice - firstSlice + 1;

		float* f_chunk = &f[uint64(firstSlice) * uint64(params.numX * params.numY)];

		// make a copy of the relavent rows
		int viewRange[2];
		params.viewRangeNeededForBackprojection(firstSlice, lastSlice, viewRange);
		float* g_chunk = &g[uint64(viewRange[0])* uint64(params.numRows*params.numCols)];

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.numZ = numSlices;
		chunk_params.removeProjections(viewRange[0], viewRange[1]);
		//chunk_params.offsetZ = params.offsetZ + (firstSlice - rowRange[0]) * params.voxelHeight;
		//if (params.geometry == parameters::MODULAR)
		//	chunk_params.offsetZ += 0.5*(chunk_params.numZ - params.numZ)* params.voxelHeight;
		if (params.mu != NULL)
			chunk_params.mu = &params.mu[uint64(firstSlice) * uint64(params.numX * params.numY)];

		// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
		chunk_params.offsetZ += firstSlice * params.voxelHeight + params.z_0() - chunk_params.z_0();

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();

		//printf("z_0: %f to %f\n", params.z_0(), chunk_params.z_0());
		//printf("slices: (%d, %d); views: (%d, %d); GPU = %d...\n", firstSlice, lastSlice, viewRange[0], viewRange[1], chunk_params.whichGPU);
		
		// Do Computation
		if (doFBP)
			FBP.execute(g_chunk, f_chunk, &chunk_params, true);
		else
			proj.backproject(g_chunk, f_chunk, &chunk_params, true);
	}
	return true;
}

bool tomographicModels::doFBP(float* g, float* f, bool cpu_to_gpu)
{
	if (cpu_to_gpu == true && FBP_multiGPU(g, f) == true)
		return true;
	else
		return FBP.execute(g, f, &params, cpu_to_gpu);
}

bool tomographicModels::sensitivity(float* f, bool cpu_to_gpu)
{
	if (params.muSpecified() == true || params.isSymmetric() == true || params.geometry == parameters::MODULAR)
	{
		if (params.whichGPU < 0 || cpu_to_gpu == true)
		{
			float* g = params.setToConstant(NULL, uint64(params.numAngles) * uint64(params.numRows) * uint64(params.numCols), 1.0);
			bool retVal = backproject(g, f, cpu_to_gpu);
			free(g);
			return retVal;
		}
		else
		{
			float* dev_g = 0;
			if (cudaMalloc((void**)&dev_g, params.numAngles * params.numRows * params.numCols * sizeof(float)) != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc(projection) failed!\n");
				return false;
			}
			setToConstant(dev_g, 1.0, make_int3(params.numAngles, params.numRows, params.numCols), params.whichGPU);
			bool retVal = backproject(dev_g, f, false);
			cudaFree(dev_g);
			return retVal;
		}
	}

	if (params.whichGPU < 0)
		return sensitivity_CPU(f, &params);
	else
	{
		if (cpu_to_gpu)
		{
			if (getAvailableGPUmemory(params.whichGPU) < params.volumeDataSize())
			{
				if (params.volumeDimensionOrder == parameters::XYZ)
				{
					printf("Error: insufficient GPU memory\n");
					return false;
				}
				else
				{
					// do chunking
					int numSlicesPerChunk = std::max(1,params.numZ / 2);
					while (getAvailableGPUmemory(params.whichGPU) < params.volumeDataSize() * float(numSlicesPerChunk) / float(params.numZ))
					{
						numSlicesPerChunk = numSlicesPerChunk / 2;
						if (numSlicesPerChunk <= 1)
						{
							numSlicesPerChunk = 1;
							break;
						}
					}
					int numChunks = int(ceil(float(params.numZ) / float(numSlicesPerChunk)));
					for (int ichunk = 0; ichunk < numChunks; ichunk++)
					{
						int firstSlice = ichunk * numSlicesPerChunk;
						int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ-1);
						if (firstSlice < params.numZ)
						{
							int numZ_save = params.numZ;
							float offsetZ_save = params.offsetZ;

							params.numZ = lastSlice - firstSlice + 1;
							params.offsetZ = params.offsetZ + (firstSlice - 0) * params.voxelHeight;

							float* f_chunk = &f[uint64(firstSlice) * uint64(params.numX * params.numZ)];
							sensitivity_gpu(f_chunk, &params, true);

							params.numZ = numZ_save;
							params.offsetZ = offsetZ_save;
						}
					}
					return true;
				}
			}
			else
				return sensitivity_gpu(f, &params, cpu_to_gpu);
		}
		else
			return sensitivity_gpu(f, &params, cpu_to_gpu);
	}
}

float tomographicModels::get_FBPscalar()
{
	return FBPscalar(&params);
}

bool tomographicModels::set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float helicalPitch)
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
	params.tau = tau;
	params.set_angles(phis, numAngles);
	params.set_helicalPitch(helicalPitch);
	return params.geometryDefined();
}

bool tomographicModels::set_fanbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau)
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
	params.tau = tau;
	params.set_angles(phis, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::set_parallelbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	params.geometry = parameters::PARALLEL;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.set_angles(phis, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::set_modularbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	params.geometry = parameters::MODULAR;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = 0.5*float(numCols-1);
	params.centerRow = 0.5*float(numRows-1);
	params.set_sourcesAndModules(sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in, numAngles);
	return params.geometryDefined();
}

bool tomographicModels::set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	if (voxelWidth <= 0.0)
		voxelWidth = params.default_voxelWidth();
	if (voxelHeight <= 0.0)
		voxelHeight = params.default_voxelHeight();

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

bool tomographicModels::set_default_volume(float scale)
{
	return params.set_default_volume(scale);
}

bool tomographicModels::set_volumeDimensionOrder(int which)
{
	if (parameters::XYZ <= which && which <= parameters::ZYX)
	{
		params.volumeDimensionOrder = which;
		return true;
	}
	else
	{
		printf("Error: volume dimension order must be 0 for XYZ or 1 for ZYX\n");
		return false;
	}
}

int tomographicModels::get_volumeDimensionOrder()
{
	return params.volumeDimensionOrder;
}

bool tomographicModels::set_GPU(int whichGPU)
{
	if (numberOfGPUs() <= 0)
		params.whichGPU = -1;
	else
		params.whichGPU = whichGPU;
	params.whichGPUs.clear();
	params.whichGPUs.push_back(whichGPU);
	return true;
}

bool tomographicModels::set_GPUs(int* whichGPUs, int N)
{
	if (whichGPUs == NULL || N <= 0)
		return false;
	params.whichGPUs.clear();
	for (int i = 0; i < N; i++)
		params.whichGPUs.push_back(whichGPUs[i]);
	params.whichGPU = params.whichGPUs[0];
	return true;
}

int tomographicModels::get_GPU()
{
	return params.whichGPU;
}

bool tomographicModels::set_projector(int which)
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
	params.axisOfSymmetry = axisOfSymmetry;
	return true;
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

bool tomographicModels::set_attenuationMap(float* mu)
{
	params.mu = mu;
	if (params.mu == NULL)
		return false;
	else
	{
		params.muCoeff = 0.0;
		params.muRadius = 0.0;
		return true;
	}
}

bool tomographicModels::set_attenuationMap(float c, float R)
{
	params.muCoeff = c;
	params.muRadius = R;
	if (params.muCoeff != 0.0 && params.muRadius > 0.0)
	{
		params.mu = NULL;
		return true;
	}
	else
	{
		params.muCoeff = 0.0;
		params.muRadius = 0.0;
		return false;
	}
}

bool tomographicModels::clear_attenuationMap()
{
	params.mu = NULL;
	params.muCoeff = 0.0;
	params.muRadius = 0.0;
	return true;
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.project(g, f, &tempParams, cpu_to_gpu);
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.backproject(g, f, &tempParams, cpu_to_gpu);
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.project(g, f, &tempParams, cpu_to_gpu);
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.backproject(g, f, &tempParams, cpu_to_gpu);
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.project(g, f, &tempParams, cpu_to_gpu);
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
	tempParams.set_angles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	return proj.backproject(g, f, &tempParams, cpu_to_gpu);
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

float tomographicModels::get_pixelWidth()
{
	return params.pixelWidth;
}

float tomographicModels::get_pixelHeight()
{
	return params.pixelHeight;
}

bool tomographicModels::set_tau(float tau)
{
	return params.set_tau(tau);
}

bool tomographicModels::set_helicalPitch(float h)
{
	return params.set_helicalPitch(h);
}

bool tomographicModels::set_normalizedHelicalPitch(float h_normalized)
{
	return params.set_normalizedHelicalPitch(h_normalized);
}

float tomographicModels::get_helicalPitch()
{
	return params.helicalPitch;
}

float tomographicModels::get_z_source_offset()
{
	return params.z_source_offset;
}

bool tomographicModels::get_sourcePositions(float* x)
{
	if (x == NULL || params.sourcePositions == NULL)
		return false;
	else
	{
		for (int i = 0; i < 3*params.numAngles; i++)
			x[i] = params.sourcePositions[i];
		return true;
	}
}

bool tomographicModels::get_moduleCenters(float* x)
{
	if (x == NULL || params.moduleCenters == NULL)
		return false;
	else
	{
		for (int i = 0; i < 3 * params.numAngles; i++)
			x[i] = params.moduleCenters[i];
		return true;
	}
}

bool tomographicModels::get_rowVectors(float* x)
{
	if (x == NULL || params.rowVectors == NULL)
		return false;
	else
	{
		for (int i = 0; i < 3 * params.numAngles; i++)
			x[i] = params.rowVectors[i];
		return true;
	}
}

bool tomographicModels::get_colVectors(float* x)
{
	if (x == NULL || params.colVectors == NULL)
		return false;
	else
	{
		for (int i = 0; i < 3 * params.numAngles; i++)
			x[i] = params.colVectors[i];
		return true;
	}
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

float tomographicModels::get_voxelWidth()
{
	return params.voxelWidth;
}

float tomographicModels::get_voxelHeight()
{
	return params.voxelHeight;
}

bool tomographicModels::BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return false;
	}
	else
		return blurFilter(f, N_1, N_2, N_3, FWHM, 3, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return false;
	}
	else
		return medianFilter(f, N_1, N_2, N_3, threshold, cpu_to_gpu, params.whichGPU);
}

float tomographicModels::TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return false;
	}
	else
		return anisotropicTotalVariation_cost(f, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return false;
	}
	else
		return anisotropicTotalVariation_gradient(f, Df, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

float tomographicModels::TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 3.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return 0.0;
	}
	else
		return anisotropicTotalVariation_quadraticForm(f, d, N_1, N_2, N_3, delta, beta, cpu_to_gpu, params.whichGPU);
}

bool tomographicModels::Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu)
{
	float numVol = 1.0;
	if (cpu_to_gpu)
		numVol = 3.0;
	else
		numVol = 1.0;
	if (getAvailableGPUmemory(params.whichGPU) < numVol * params.volumeDataSize())
	{
		printf("Error: Insufficient GPU memory for this operation!\n");
		return false;
	}
	else
		return diffuse(f, N_1, N_2, N_3, delta, numIter, cpu_to_gpu, params.whichGPU);
}
