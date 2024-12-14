////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main API for LEAP
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <algorithm>
#include <omp.h>
#include "tomographic_models.h"
#include "ray_weighting_cpu.h"
#include "ramp_filter_cpu.h"
#include "cuda_utils.h"
#include "sensitivity_cpu.h"
#include "sensitivity.cuh"
#include "ramp_filter_cpu.h"
#include "find_center_cpu.h"
#include "projectors_Joseph_cpu.h"
#include "analytic_ray_tracing.h"
#include "sinogram_replacement.h"
#include "resample_cpu.h"
#include "rebin.h"
#ifndef __USE_CPU
#include "resample.cuh"
#include "ramp_filter.cuh"
#include "noise_filters.cuh"
#include "total_variation.cuh"
#include "matching_pursuit.cuh"
#include "bilateral_filter.cuh"
#include "guided_filter.cuh"
#include "scatter_models.cuh"
#include "geometric_calibration.cuh"
#include "analytic_ray_tracing_gpu.cuh"
#endif

#include "log.h"
//Log::ReportingLevel() = logSTATUS;

tomographicModels::tomographicModels()
{
	className = "tomographicModels";
	params.initialize();
	maxSlicesForChunking = 128;
	//maxSlicesForChunking = 256;
	//maxSlicesForChunking = 512;
	//minSlicesForChunking = std::min(64, maxSlicesForChunking);
	//minSlicesForChunking = std::min(8, maxSlicesForChunking);
	minSlicesForChunking = std::min(32, maxSlicesForChunking);

	/*
	//std::ofstream pfile;

	pfile.open("LEAPCT.log");
	if (!pfile.is_open())
		std::cout << "log file could not be opened" << std::endl;

	Log::Stream() = &pfile;

	Log::ReportingLevel() = logDEBUG;
	LOG(logDEBUG, className, "") << "Error parsing phantom description!" << std::endl;
	//*/
}

tomographicModels::~tomographicModels()
{
	reset();
}

void tomographicModels::set_log_error()
{
	Log::ReportingLevel() = logERROR;
}

void tomographicModels::set_log_warning()
{
	Log::ReportingLevel() = logWARNING;
}

void tomographicModels::set_log_status()
{
	Log::ReportingLevel() = logSTATUS;
}

void tomographicModels::set_log_debug()
{
	Log::ReportingLevel() = logDEBUG;
}

bool tomographicModels::set_maxSlicesForChunking(int N)
{
	if (N >= 1 && N <= 1024)
	{
		maxSlicesForChunking = N;
		//minSlicesForChunking = min(8, maxSlicesForChunking);
		return true;
	}
	else
		return false;
}

bool tomographicModels::print_parameters()
{
	params.printAll();
	return true;
}

const char* tomographicModels::about()
{
	printf("****************************************************************\n");
	printf("LivermorE AI Projector for Computed Tomography (LEAP)\n");
	printf("                     version %s\n", LEAP_VERSION);
	printf("                   LLNL-CODE-848657\n");
	printf("\n");
	printf("             compiled: %s %s\n", __DATE__, __TIME__);
	printf("        written by: Kyle Champley and Hyojin Kim\n");
	printf("****************************************************************\n");

	return LEAP_VERSION;
}

bool tomographicModels::reset()
{
	params.clearAll();
	params.initialize();
	geometricPhantom.clearObjects();
	return true;
}

float* tomographicModels::allocate_volume()
{
	float* retVal = NULL;
	if (params.volumeDefined(false))
		retVal = new float[params.volumeData_numberOfElements()];
	return retVal;
}

float* tomographicModels::allocate_projections()
{
	float* retVal = NULL;
	if (params.geometryDefined(false))
		retVal = new float[params.projectionData_numberOfElements()];
	return retVal;
}

bool tomographicModels::project_gpu(float* g, float* f)
{
#ifndef __USE_CPU
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = project(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::backproject_gpu(float* g, float* f)
{
#ifndef __USE_CPU
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = backproject(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::FBP_gpu(float* g, float* f)
{
#ifndef __USE_CPU
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = doFBP(g, f, false);
	params.whichGPU = whichGPU_save;
	return retVal;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::project_cpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = project(g, f, true);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::project_with_mask_cpu(float* g, float* f, float* mask)
{
	bool data_on_cpu = true;
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = project_with_mask(g, f, mask, true);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::project_with_mask_gpu(float* g, float* f, float* mask)
{
#ifndef __USE_CPU
	bool data_on_cpu = false;
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = project_with_mask(g, f, mask, false);
	params.whichGPU = whichGPU_save;
	return retVal;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::backproject_cpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = backproject(g, f, true);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::filterProjections_cpu(float* g)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = filterProjections(g, g, true);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::filterProjections_gpu(float* g)
{
#ifndef __USE_CPU
	int whichGPU_save = params.whichGPU;
	if (params.whichGPU < 0)
		params.whichGPU = 0;
	bool retVal = filterProjections(g, g, false);
	params.whichGPU = whichGPU_save;
	return retVal;
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::FBP_cpu(float* g, float* f)
{
	int whichGPU_save = params.whichGPU;
	params.whichGPU = -1;
	bool retVal = doFBP(g, f, true);
	params.whichGPU = whichGPU_save;
	return retVal;
}

bool tomographicModels::project(float* g, float* f, bool data_on_cpu)
{
	if (data_on_cpu == true && project_multiGPU(g, f) == true)
		return true;
	else
		return proj.project(g, f, &params, data_on_cpu);
}

bool tomographicModels::project_with_mask(float* g, float* f, float* mask, bool data_on_cpu)
{
	bool retVal = copy_volume_data_to_mask(f, mask, data_on_cpu, true);
	if (retVal == false)
		return false;
	if (data_on_cpu == true && project_multiGPU(g, f) == true)
		retVal = true;
	else
		retVal = proj.project(g, f, &params, data_on_cpu);
	copy_volume_data_to_mask(f, mask, data_on_cpu, false);
	return retVal;
}

bool tomographicModels::copy_volume_data_to_mask(float* f, float* mask, bool data_on_cpu, bool do_forward)
{
	if (f == NULL || mask == NULL)
		return false;

	if (data_on_cpu)
	{
		int N_1 = params.numZ;
		int N_2 = params.numY;
		int N_3 = params.numX;
		if (params.volumeDimensionOrder == parameters::XYZ)
		{
			N_1 = params.numX;
			N_2 = params.numY;
			N_3 = params.numZ;
		}

		omp_set_num_threads(omp_get_num_procs());
		#pragma omp parallel for
		for (int i = 0; i < N_1; i++)
		{
			uint64 ind_offs = uint64(i) * uint64(N_2) * uint64(N_3);
			for (int j = 0; j < N_2; j++)
			{
				for (int k = 0; k < N_3; k++)
				{
					uint64 ind = ind_offs + uint64(j * N_3 + k);
					if (do_forward)
					{
						if (mask[ind] == 0.0)
						{
							if (f[ind] == 0.0)
								mask[ind] = NAN;
							else
							{
								mask[ind] = f[ind];
								f[ind] = 0.0;
							}
						}
						else
						{
							if (f[ind] == 0.0)
								mask[ind] = -1.0;
							//else
							//	mask[ind] = 1.0;
						}
					}
					else
					{
						if (f[ind] == 0.0)
						{
							if (isnan(mask[ind]))
								mask[ind] = 0.0;
							else if (mask[ind] == -1.0)
								mask[ind] = 1.0;
							else
							{
								f[ind] = mask[ind];
								mask[ind] = 0.0;
							}
						}
						else
							mask[ind] = 1.0;
					}
				}
			}
		}
		return true;
	}
	else
	{
#ifndef __USE_CPU
		return copy_volume_data_to_mask_gpu(f, mask, &params, do_forward);
#else
		LOG(logERROR, "", "") << "Error: GPU routines not included in this release!" << std::endl;
		return false;
#endif
	}
}

bool tomographicModels::backproject(float* g, float* f, bool data_on_cpu)
{
	if (data_on_cpu == true && backproject_multiGPU(g, f) == true)
		return true;
	else
	{
		/*
		float* dev_f = copyVolumeDataToGPU(f, &params, params.whichGPU);

		for (int i = 0; i < params.numAngles; i++)
		{
			parameters chunk_params = params;
			chunk_params.removeProjections(i, i);
			float* g_chunk = &g[i*uint64(params.numRows*params.numCols)];
			if (i == 0)
				proj.backproject(g_chunk, dev_f, &chunk_params, data_on_cpu, false, false);
			else
				proj.backproject(g_chunk, dev_f, &chunk_params, data_on_cpu, false, false);
		}
		pullVolumeDataFromGPU(f, &params, dev_f, params.whichGPU);
		cudaFree(dev_f);
		return true;
		//*/

		return proj.backproject(g, f, &params, data_on_cpu);
	}
}

bool tomographicModels::weightedBackproject(float* g, float* f, bool data_on_cpu)
{
	bool doWeight_save = params.doWeightedBackprojection;
	params.doWeightedBackprojection = true;

	bool doExtrapolation_save = params.doExtrapolation;
	if ((params.geometry != parameters::CONE && params.geometry != parameters::CONE_PARALLEL) || params.helicalPitch == 0.0)
		params.doExtrapolation = true;

	bool retVal = backproject(g, f, data_on_cpu);
	params.doWeightedBackprojection = doWeight_save;
	params.doExtrapolation = doExtrapolation_save;
	return retVal;
}

bool tomographicModels::HilbertFilterProjections(float* g, bool data_on_cpu, float scalar)
{
	return FBP.HilbertFilterProjections(g, &params, data_on_cpu, scalar);
}

bool tomographicModels::rampFilterProjections(float* g, bool data_on_cpu, float scalar)
{
	return FBP.rampFilterProjections(g, &params, data_on_cpu, scalar);
}

bool tomographicModels::filterProjections_multiGPU(float* g, float* g_out)
{
#ifndef __USE_CPU
	//return false;
	if (params.whichGPU < 0)
		return false;
	if (params.muSpecified() || params.isSymmetric())
		return false;
	if (params.geometry == parameters::MODULAR && params.modularbeamIsAxiallyAligned() == false)
		return false;
	//if (params.offsetScan)
	//	return false; // FIXME
	int extraCols = zeroPadForOffsetScan_numberOfColsToAdd(&params);

	// First divide numAngles across all GPUs
	int numViewsPerChunk = std::max(1, int(ceil(float(params.numAngles) / std::max(1.0, double(params.whichGPUs.size())))));
	//if (numViewsPerChunk == params.numAngles)
	//	numViewsPerChunk = params.numAngles / 2; // FIXME: this is only temporary!!!

	// Now divide numAngles further to fit on the GPUs
	// reserve some extra memory for filtering (FFT)
	float memAvailable = getAvailableGPUmemory(params.whichGPUs);
	float memNeeded = (1.0+2.0/40.0)*params.projectionDataSize()*float(params.numCols + extraCols)/float(params.numCols);
	if (memNeeded >= memAvailable)
	{
		// memNeeded*N/params.numAngles = memAvailable
		numViewsPerChunk = std::min(numViewsPerChunk, int(floor(memAvailable * float(params.numAngles) / memNeeded)));
	}
	if (numViewsPerChunk < 2)
		return false; // not enough GPU memory for even two projections, ouch!

	if (numViewsPerChunk == params.numAngles)
		return false; // chunking not necessary, just run usual single-gpu routine

	// Now calculate the number of chunks necessary
	int numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));

	// Have to divide data into numChunks, so make the chunks equal
	numViewsPerChunk = int(ceil(float(params.numAngles) / float(numChunks)));

	int numCols = params.numCols;
	float centerCol = params.centerCol;
	float colShiftFromFilter = params.colShiftFromFilter;
	float rowShiftFromFilter = params.rowShiftFromFilter;

	// FIXME: need to copy over some of the changes to params that filter makes
	omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
	#pragma omp parallel for schedule(dynamic)
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstView = ichunk * numViewsPerChunk;
		int lastView = std::min(firstView + numViewsPerChunk - 1, params.numAngles - 1);
		int numViews = lastView - firstView + 1;

		// make a pointer to the start of the view to process
		float* g_chunk = &g[uint64(firstView) * uint64(params.numRows * params.numCols)];
		float* g_out_chunk = &g_out[uint64(firstView) * uint64(params.numRows * (params.numCols+ extraCols))];

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.removeProjections(firstView, lastView);

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();

		//printf("full numAngles = %d, chunk numAngles = %d\n", params.numAngles, chunk_params.numAngles);
		//printf("GPU %d: view range: (%d, %d)    slice range: (%d, %d)\n", chunk_params.whichGPU, firstView, lastView, sliceRange[0], sliceRange[1]);

		LOG(logDEBUG, className, "") << "filtering on GPU " << chunk_params.whichGPU << ": views = (" << firstView << ", " << lastView << ")" << std::endl;

		// Do Computation
		FBP.filterProjections(g_chunk, g_out_chunk, &chunk_params, true);

		if (ichunk == 0)
		{
			numCols = chunk_params.numCols;
			centerCol = chunk_params.centerCol;
			colShiftFromFilter = chunk_params.colShiftFromFilter;
			rowShiftFromFilter = chunk_params.rowShiftFromFilter;
		}
	}

	params.numCols = numCols;
	params.centerCol = centerCol;
	params.colShiftFromFilter = colShiftFromFilter;
	params.rowShiftFromFilter = rowShiftFromFilter;

	return true;
#else
	return false;
#endif
}

bool tomographicModels::filterProjections(float* g, float* g_out, bool data_on_cpu)
{
	//*
	if (params.offsetScan)
	{
		if (zeroPadForOffsetScan_numberOfColsToAdd(&params) <= 0)
			params.offsetScan = false;
	}
	//*/

	if (data_on_cpu == true && filterProjections_multiGPU(g, g_out) == true)
		return true;
	else
		return FBP.filterProjections(g, g_out, &params, data_on_cpu);
}

bool tomographicModels::preRampFiltering(float* g, bool data_on_cpu)
{
	return FBP.preRampFiltering(g, &params, data_on_cpu);
}

bool tomographicModels::postRampFiltering(float* g, bool data_on_cpu)
{
	return FBP.postRampFiltering(g, &params, data_on_cpu);
}

bool tomographicModels::rampFilterVolume(float* f, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: 2D ramp filter only implemented for GPU\n");
		return false;
	}
	if (params.volumeDimensionOrder == parameters::XYZ)
		return rampFilter2D_XYZ(f, &params, data_on_cpu);
	else
	{
		// copies the whole volume onto the GPU and then applies the ramp filter one slice at a time
		// thus very, very little extra memory is required
		int N_1 = params.numZ;

		uint64 numElements = uint64(params.numZ) * uint64(params.numY) * uint64(params.numX);
		double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
		//uint64 maxElements = 2147483646; // 2^31 = 2*1024^3-2

		if (data_on_cpu == true && (getAvailableGPUmemory(params.whichGPU) < dataSize /*|| numElements > maxElements*/))
		{ // if data is on the cpu and there is not enough memory to process the whole thing, then process in chunks
			
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* f_chunk = &f[uint64(sliceStart) * uint64(params.numY * params.numX)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				parameters params_chunk = params;
				params_chunk.numAngles = sliceEnd - sliceStart + 1;
				params_chunk.whichGPU = whichGPU;

				rampFilter2D(f_chunk, &params_chunk, data_on_cpu);
			}

			return true;
		}
		else
		{
			return rampFilter2D(f, &params, data_on_cpu);
		}
	}
#else
	printf("Error: 2D ramp filter only implemented for GPU\n");
	return false;
#endif
}

bool tomographicModels::windowFOV(float* f, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (data_on_cpu == false)
		return windowFOV_gpu(f, &params);
	else
		return windowFOV_cpu(f, &params);
#else
	return windowFOV_cpu(f, &params);
#endif
}

float* tomographicModels::copyRows(float* g, int firstSlice, int lastSlice, int firstView, int lastView)
{
	//Timer clock;
	//clock.tick();

	if (firstView < 0)
		firstView = 0;
	if (lastView < 0 || lastView >= params.numAngles)
		lastView = params.numAngles - 1;
	if (firstView > lastView)
		return NULL;
	int numViews_new = lastView - firstView + 1;

	int numSlices = lastSlice - firstSlice + 1;
	float* g_chunk = (float*)malloc(sizeof(float) * uint64(numViews_new) * uint64(params.numCols) * uint64(numSlices));

	//*
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = firstView; iphi <= lastView; iphi++)
	{
		float* g_proj = &g[uint64(iphi) * uint64(params.numRows * params.numCols) + firstSlice* params.numCols];
		float* g_chunk_proj = &g_chunk[uint64(iphi- firstView) * uint64(numSlices * params.numCols)];

		memcpy(g_chunk_proj, g_proj, sizeof(float) * params.numCols * numSlices);

		//for (int iRow = firstSlice; iRow <= lastSlice; iRow++)
		//{
		//	float* g_line = &g_proj[iRow * params.numCols];
		//	float* g_chunk_line = &g_chunk_proj[(iRow - firstSlice) * params.numCols];
		//	memcpy(g_chunk_line, g_line, sizeof(float) * params.numCols);
		//}
	}
	//*/

	/*
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
			memcpy(g_chunk_line, g_line, sizeof(float)*params.numCols);
			//for (int iCol = 0; iCol < params.numCols; iCol++)
			//	g_chunk_line[iCol] = g_line[iCol];
		}
	}
	//*/

	//clock.tock();
	//printf("Elapsed time: %d\n", clock.duration().count());

	return g_chunk;
}

bool tomographicModels::combineRows(float* g, float* g_chunk, int firstRow, int lastRow, int firstView, int lastView)
{
	if (firstView < 0)
		firstView = 0;
	if (lastView < 0 || lastView >= params.numAngles)
		lastView = params.numAngles - 1;
	if (firstView > lastView)
		return NULL;
	int numViews_new = lastView - firstView + 1;

	int numRows = lastRow - firstRow + 1;
	
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = firstView; iphi <= lastView; iphi++)
	{
		float* g_proj = &g[uint64(iphi) * uint64(params.numRows * params.numCols) + firstRow * params.numCols];
		float* g_chunk_proj = &g_chunk[uint64(iphi - firstView) * uint64(numRows * params.numCols)];

		memcpy(g_proj, g_chunk_proj, sizeof(float) * params.numCols * numRows);
	}

	/*
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
			memcpy(g_line, g_chunk_line, sizeof(float) * params.numCols);
		}
	}
	//*/
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
#ifndef __USE_CPU
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;
	if ((params.geometry == parameters::CONE || params.geometry == parameters::CONE_PARALLEL) && params.helicalPitch != 0.0)
		return project_multiGPU_splitViews(g, f);
	if (params.geometry == parameters::MODULAR)
	{
		if (params.modularbeamIsAxiallyAligned() == false)
			return project_multiGPU_splitViews(g, f);
	}

	int numProjectionData = 1;
	int numVolumeData = 1;

	float memAvailable = getAvailableGPUmemory(params.whichGPUs);

	//memAvailable = 2.0; // FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// Calculate the minimum number of rows one would like to calculate at a time before the volume must be divided further
	// We want to make sure that this lower bound is set low enough to leave extra room for the volume data
	// For now let's make it so it can't occupy more than half the memory
	int minSlicesForChunking_local = std::min(minSlicesForChunking, params.numRows);
	// 0.5*memAvailable >= params.projectionDataSize()*float(minSlicesForChunking_local) / float(params.numRows)
	minSlicesForChunking_local = std::min(minSlicesForChunking_local, int(floor(0.5 * memAvailable * float(params.numRows) / params.projectionDataSize())));
	minSlicesForChunking_local = std::max(1, minSlicesForChunking_local);

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numRowsPerChunk = std::min(64, params.numRows);
	int numRowsPerChunk = std::max(1, int(ceil(float(params.numRows) / std::max(2.0, double(params.whichGPUs.size())) )));
	numRowsPerChunk = std::min(numRowsPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	if (params.hasSufficientGPUmemory(true, 0, numProjectionData, numVolumeData) == false)
	{
		float memNeeded = project_memoryRequired(numRowsPerChunk);

		while (memAvailable < memNeeded)
		{
			numRowsPerChunk = numRowsPerChunk / 2;
			if (numRowsPerChunk <= minSlicesForChunking_local)
			{
				numRowsPerChunk = minSlicesForChunking_local;
				numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
				break;
			}
			memNeeded = project_memoryRequired(numRowsPerChunk);
		}
		numChunks = std::max(1, int(ceil(float(params.numRows) / float(numRowsPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory(0, numProjectionData, numVolumeData) <= params.chunkingMemorySizeThreshold)
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

	//if (params.geometry != parameters::FAN && params.geometry != parameters::PARALLEL && params.geometry != parameters::CONE)
	//	return false;

	bool retVal = true;

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

		if (numSlices > 0)
		{
			// make a copy of the relavent rows
			float* g_chunk = (float*)malloc(sizeof(float) * params.numAngles * params.numCols * numRows);

			float memNeeded = params.projectionDataSize() * float(numRows) / float(params.numRows) + params.volumeDataSize() * float(numSlices) / float(params.numZ) + params.get_extraMemoryReserved();
			if (memNeeded < memAvailable)
			{
				float* f_chunk = &f[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

				// make a copy of the params
				parameters chunk_params;
				chunk_params = params;
				chunk_params.numRows = numRows;
				chunk_params.numZ = numSlices;
				chunk_params.centerRow = params.centerRow - firstRow;

				// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
				chunk_params.offsetZ += sliceRange[0] * params.voxelHeight + params.z_0() - chunk_params.z_0();

				chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
				chunk_params.whichGPUs.clear();
				if (params.mu != NULL)
					chunk_params.mu = &params.mu[uint64(sliceRange[0]) * uint64(params.numX * params.numY)];

				LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": projection: z-slices: (" << sliceRange[0] << ", " << sliceRange[1] << "), rows = (" << firstRow << ", " << lastRow << ")" << std::endl;

				// Do Computation
				proj.project(g_chunk, f_chunk, &chunk_params, true);
			}
			else
			{
				// Further reduce number of volume slices
				// memAvailable => params.projectionDataSize() * float(numRows) / float(params.numRows) + params.volumeDataSize() * float(numSlices_stage2) / float(params.numZ)
				int numSlices_stage2 = std::max(1, int(floor((memAvailable - params.get_extraMemoryReserved() - params.projectionDataSize() * float(numRows) / float(params.numRows)) * float(params.numZ) / params.volumeDataSize())));
				int numChunks_z = std::max(1, int(ceil(float(numSlices) / float(numSlices_stage2))));
				numSlices_stage2 = std::max(1, int(ceil(float(numSlices) / float(numChunks_z)))); // make each chunk approximately equal in size

				LOG(logDEBUG, className, "") << "GPU " << params.whichGPUs[omp_get_thread_num()] << ": numSices = " << numSlices_stage2 << ", numRows = " << numRows << std::endl;

				float* dev_g = 0;
				cudaError_t cudaStatus;
				cudaSetDevice(params.whichGPUs[omp_get_thread_num()]);
				if ((cudaStatus = cudaMalloc((void**)&dev_g, uint64(params.numAngles) * uint64(numRows) * uint64(params.numCols) * sizeof(float))) != cudaSuccess)
				{
					fprintf(stderr, "cudaMalloc(projections) failed!\n");
					retVal = false;
					continue;
				}

				for (int ichunk_z = 0; ichunk_z < numChunks_z; ichunk_z++)
				{
					int sliceRange_stage2[2];
					sliceRange_stage2[0] = sliceRange[0] + ichunk_z * numSlices_stage2;
					sliceRange_stage2[1] = std::min(sliceRange_stage2[0] + numSlices_stage2 - 1, sliceRange[1]);
					int numSlices_stage2 = sliceRange_stage2[1] - sliceRange_stage2[0] + 1;
					if (numSlices_stage2 > 0)
					{
						float* f_chunk = &f[uint64(sliceRange_stage2[0]) * uint64(params.numX * params.numY)];

						// make a copy of the params
						parameters chunk_params;
						chunk_params = params;
						chunk_params.numRows = numRows;
						chunk_params.numZ = numSlices_stage2;
						chunk_params.centerRow = params.centerRow - firstRow;

						// need: chunk_params.z_0() + z_shift = sliceRange_stage2[0]*params.voxelHeight + params.z_0()
						chunk_params.offsetZ += sliceRange_stage2[0] * params.voxelHeight + params.z_0() - chunk_params.z_0();

						chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
						chunk_params.whichGPUs.clear();
						if (params.mu != NULL)
							chunk_params.mu = &params.mu[uint64(sliceRange_stage2[0]) * uint64(params.numX * params.numY)];

						LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": volume: z-slices: (" << sliceRange_stage2[0] << ", " << sliceRange_stage2[1] << "), rows = (" << firstRow << ", " << lastRow << ")" << std::endl;

						bool accumulate = false;
						if (ichunk_z > 0)
							accumulate = true;

						// Do Computation
						proj.project(dev_g, f_chunk, &chunk_params, false, true, accumulate);
						//proj.project(g_chunk, f_chunk, &chunk_params, true);
					}
				}
				pull3DdataFromGPU(g_chunk, make_int3(params.numAngles, numRows, params.numCols), dev_g, params.whichGPUs[omp_get_thread_num()]);
				if (dev_g != 0)
					cudaFree(dev_g);
			}
			combineRows(g, g_chunk, firstRow, lastRow);

			// clean up
			free(g_chunk);
		}
	}
	return retVal;
#else
	return false;
#endif
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

		//float memoryNeeded = 2.0*float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numRows) / float(params.numRows) * params.projectionDataSize();
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

		//float memoryNeeded = 2.0*float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
		float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
		maxMemory = std::max(maxMemory, memoryNeeded);
	}
	return maxMemory + params.get_extraMemoryReserved();
}

bool tomographicModels::project_multiGPU_splitViews(float* g, float* f)
{
#ifndef __USE_CPU
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;

	int numProjectionData = 1;
	int numVolumeData = 1; // need an extra for texture memory (not anymore)

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numRowsPerChunk = std::min(64, params.numRows);
	int numViewsPerChunk = std::max(1, int(ceil(float(params.numAngles) / std::max(2.0, double(params.whichGPUs.size())))));
	//numViewsPerChunk = std::min(numViewsPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
	if (params.hasSufficientGPUmemory(true, 0, numProjectionData, numVolumeData) == false)
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
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory(0, numProjectionData, numVolumeData) <= params.chunkingMemorySizeThreshold)
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

		LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": projection: z-slices: (" << sliceRange[0] << ", " << sliceRange[1] << "), views = (" << firstView << ", " << lastView << ")" << std::endl;

		// Do Computation
		proj.project(g_chunk, f_chunk, &chunk_params, true);
	}
	return true;
#else
	return false;
#endif
}

bool tomographicModels::backproject_FBP_multiGPU(float* g, float* f, bool doFBP)
{
#ifndef __USE_CPU
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;
	if ((params.geometry == parameters::CONE || params.geometry == parameters::CONE_PARALLEL) && params.helicalPitch != 0.0)
		return backproject_FBP_multiGPU_splitViews(g, f, doFBP);
	if (params.geometry == parameters::MODULAR)
	{
		if (params.modularbeamIsAxiallyAligned() == false)
			return backproject_FBP_multiGPU_splitViews(g, f, doFBP);
	}

	int extraCols = 0;
	if (doFBP)
		extraCols = zeroPadForOffsetScan_numberOfColsToAdd(&params);
	//printf("extraCols = %d\n", extraCols);

	if (extraCols > 0)
	{
		LOG(logDEBUG, className, "") << "Extra columns needed for offset scan reconstruction: " << extraCols << std::endl;
	}

	float numProjectionData = 1.0;
	float numVolumeData = 1.0;
	if (params.mu != NULL)
		numVolumeData += 1.0;
	if (doFBP)
		numProjectionData = 2.0; // need an extra for texture memory

	//int numViewsPerChunk = params.numAngles;
	//int numViewChunks = 1;

	float memAvailable = getAvailableGPUmemory(params.whichGPUs);
	LOG(logDEBUG, className, "") << "GPU memory available: " << memAvailable << " GB" << std::endl;

	//memAvailable = 0.5; // FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// Calculate the minimum number of slices one would like to calculate at a time before jobs are broken across views
	// We want to make sure that this lower bound is set low enough so that the volume will still fit into memory
	// and leave extra room for the projection data.  For now let's make it so it can't occupy more than half the memory
	int minSlicesForChunking_local = std::min(minSlicesForChunking, params.numZ);
	// 0.5*memAvailable >= params.volumeDataSize()*float(minSlicesForChunking_local) / float(params.numZ)
	minSlicesForChunking_local = std::min(minSlicesForChunking_local, int(floor(0.5*memAvailable*float(params.numZ) / params.volumeDataSize())));
	minSlicesForChunking_local = std::max(1, minSlicesForChunking_local);

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numSlicesPerChunk = std::min(64, params.numZ);
	int numSlicesPerChunk = std::max(1, int(ceil(float(params.numZ) / std::max(2.0, double(params.whichGPUs.size())))));
	numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	if (params.hasSufficientGPUmemory(true, extraCols, numProjectionData, numVolumeData) == false)
	{
		//*
		float memNeeded = backproject_memoryRequired(numSlicesPerChunk, extraCols, doFBP);
		while (memAvailable < memNeeded)
		{
			numSlicesPerChunk = numSlicesPerChunk / 2;
			if (numSlicesPerChunk <= minSlicesForChunking_local)
			{
				numSlicesPerChunk = minSlicesForChunking_local;
				break;
			}
			memNeeded = backproject_memoryRequired(numSlicesPerChunk, extraCols, doFBP);
		}
		//*/

		/*
		float memNeeded = backproject_memoryRequired(numSlicesPerChunk, extraCols, doFBP, numViewsPerChunk);
		while (memAvailable < memNeeded)
		{
			if (numSlicesPerChunk <= minSlicesForChunking_local)
				numViewsPerChunk = numViewsPerChunk / 2;
			else
				numSlicesPerChunk = numSlicesPerChunk / 2;
			if (numSlicesPerChunk <= 1 || numViewsPerChunk <= 1)
				return false;
			memNeeded = backproject_memoryRequired(numSlicesPerChunk, extraCols, doFBP, numViewsPerChunk);
		}
		numViewChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
		//*/
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory(extraCols, numProjectionData, numVolumeData) <= params.chunkingMemorySizeThreshold)
	{
		// in this case one is only using one GPU and the whole operation can fit on the GPU, so no reason to chunk the data
		if (params.numZ > 1) // well if we are just reconstructing one slice, then we don't want to filter and copy over unnecessary detector rows
			return false;
	}
	else
	{
		numSlicesPerChunk = int(ceil(float(params.numZ) / float(params.whichGPUs.size())));
		numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}

	if (numChunks <= 1)
	{
		bool retVal2 = false;
		if (params.numZ == 1 && params.numRows > 1)
			retVal2 = true;
		if (retVal2 == false)
			return false;
	}

	bool retVal = true;

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
		
		/*
		float* g_chunk = NULL;
		if (rowRange[0] == 0 && rowRange[1] == params.numRows - 1)
			g_chunk = g;
		else
			g_chunk = copyRows(g, rowRange[0], rowRange[1]);
		//*/

		// make a copy of the params
		parameters chunk_params;
		chunk_params = params;
		chunk_params.numRows = rowRange[1] - rowRange[0] + 1;
		chunk_params.numZ = numSlices;

		chunk_params.centerRow = params.centerRow - rowRange[0];
		if (params.mu != NULL)
			chunk_params.mu = &params.mu[uint64(firstSlice) * uint64(params.numX * params.numY)];

		// need: chunk_params.z_0() + z_shift = sliceRange[0]*params.voxelHeight + params.z_0()
		chunk_params.offsetZ += firstSlice * params.voxelHeight + params.z_0() - chunk_params.z_0();

		chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
		chunk_params.whichGPUs.clear();

		// Calculate the view chunking
		float memForVolume = numVolumeData * params.volumeDataSize() * float(numSlices) / float(params.numZ);
		float memForOneProjection = numProjectionData * params.projectionDataSize(extraCols) * (chunk_params.numRows / float(params.numRows)) / float(params.numAngles);
		// memAvailable > memForVolume + numViewsPerChunk * memForOneProjection
		int numViewsPerChunk = std::max(1, std::min(params.numAngles, int(floor((memAvailable - memForVolume - params.get_extraMemoryReserved()) / memForOneProjection))));
		int numViewChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
		if (numViewsPerChunk == params.numAngles)
		{
			LOG(logDEBUG, className, "") << "using volume slab size of " << numSlicesPerChunk << std::endl;

			//*
			float* g_chunk = NULL;
			if (rowRange[0] == 0 && rowRange[1] == params.numRows - 1)
				g_chunk = g;
			else
				g_chunk = copyRows(g, rowRange[0], rowRange[1]);
			//*/

			// Do Computation
			if (doFBP)
			{
				LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": FBP: z-slices: (" << firstSlice << ", " << lastSlice << "), rows = (" << rowRange[0] << ", " << rowRange[1] << ")" << std::endl;
				FBP.execute(g_chunk, f_chunk, &chunk_params, true);
			}
			else
			{
				LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": backprojection: z-slices: (" << firstSlice << ", " << lastSlice << "), rows = (" << rowRange[0] << ", " << rowRange[1] << ")" << std::endl;
				proj.backproject(g_chunk, f_chunk, &chunk_params, true);
			}

			if (g_chunk != g)
				free(g_chunk);
		}
		else
		{
			numViewsPerChunk = int(ceil(float(params.numAngles) / float(numViewChunks)));
			numViewChunks = std::max(1, int(ceil(float(params.numAngles) / float(numViewsPerChunk))));
			LOG(logDEBUG, className, "") << "using volume slab size of " << numSlicesPerChunk << " and number of angles: " << numViewsPerChunk << std::endl;
			//LOG(logDEBUG, className, "") << "memForVolume = " << memForVolume << std::endl;
			//LOG(logDEBUG, className, "") << "memForOneProjection = " << memForOneProjection << std::endl;

			cudaError_t cudaStatus;
			cudaSetDevice(chunk_params.whichGPU);
			float* dev_f = 0;
			if ((cudaStatus = cudaMalloc((void**)&dev_f, uint64(chunk_params.numX) * uint64(chunk_params.numY) * uint64(chunk_params.numZ) * sizeof(float))) != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc(volume) failed!\n");
				//if (g_chunk != g)
				//	free(g_chunk);
				retVal = false;
				//return false;
			}

			for (int ichunk_stage2 = 0; ichunk_stage2 < numViewChunks; ichunk_stage2++)
			{
				int firstView = ichunk_stage2 * numViewsPerChunk;
				int lastView = std::min(firstView + numViewsPerChunk - 1, params.numAngles - 1);
				int numViews = lastView - firstView + 1;

				// make a copy of the relavent rows
				//*
				float* g_chunk_stage2 = NULL;
				bool delete_g_chunk_stage2 = false;
				if (rowRange[0] == 0 && rowRange[1] == params.numRows - 1)
					g_chunk_stage2 = &g[uint64(firstView)*uint64(params.numRows)* uint64(params.numCols)];
				else
				{
					g_chunk_stage2 = copyRows(g, rowRange[0], rowRange[1], firstView, lastView);
					delete_g_chunk_stage2 = true;
				}
				//*/

				//float* g_chunk_stage2 = &g_chunk[uint64(firstView)* uint64(chunk_params.numRows*chunk_params.numCols)];

				parameters chunk_params_stage2;
				chunk_params_stage2 = chunk_params;
				chunk_params_stage2.removeProjections(firstView, lastView);

				bool accumulate = false;
				if (ichunk_stage2 > 0)
					accumulate = true;

				// Do Computation
				if (doFBP)
				{
					LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": FBP: z-slices: (" << firstSlice << ", " << lastSlice << "), rows = (" << rowRange[0] << ", " << rowRange[1] << ")" << std::endl;
					FBP.execute(g_chunk_stage2, dev_f, &chunk_params_stage2, true, false, accumulate);
				}
				else
				{
					LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": backprojection: z-slices: (" << firstSlice << ", " << lastSlice << "), rows = (" << rowRange[0] << ", " << rowRange[1] << ")" << std::endl;
					proj.backproject(g_chunk_stage2, dev_f, &chunk_params_stage2, true, false, accumulate);
				}

				if (delete_g_chunk_stage2 && g_chunk_stage2 != NULL)
					free(g_chunk_stage2);
			}

			pullVolumeDataFromGPU(f_chunk, &chunk_params, dev_f, chunk_params.whichGPU);
			if (dev_f != 0)
				cudaFree(dev_f);
		}

		/* clean up
		if (g_chunk != g)
			free(g_chunk);
		//*/
	}
	//printf("done\n");
	return retVal;
#else
	return false;
#endif
}

int tomographicModels::numRowsRequiredForBackprojectingSlab(int numSlicesPerChunk)
{
	int maxRows = 0;

	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
		int numSlices = lastSlice - firstSlice + 1;

		int rowRange[2];
		params.rowRangeNeededForBackprojection(firstSlice, lastSlice, rowRange);
		int numRows = rowRange[1] - rowRange[0] + 1;

		maxRows = std::max(maxRows, numRows);
	}
	return maxRows;
}

int tomographicModels::extraColumnsForOffsetScan()
{
	return zeroPadForOffsetScan_numberOfColsToAdd(&params);
}

float tomographicModels::backproject_memoryRequired(int numSlicesPerChunk, int extraCols, bool doFBP, int numViews)
{
	float maxMemory = 0.0;

	if (numViews <= 0)
		numViews = params.numAngles;

	float proj_size_scaling = float(numViews) / float(params.numAngles);

	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	for (int ichunk = 0; ichunk < numChunks; ichunk++)
	{
		int firstSlice = ichunk * numSlicesPerChunk;
		int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
		int numSlices = lastSlice - firstSlice + 1;

		int rowRange[2];
		params.rowRangeNeededForBackprojection(firstSlice, lastSlice, rowRange);
		int numRows = rowRange[1] - rowRange[0] + 1;

		if (doFBP)
		{
			// projections are copied to the GPU for filtering, so cannot go directly from CPU memory to texture memory
			float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + 2.0 * proj_size_scaling * float(numRows) / float(params.numRows) * params.projectionDataSize(extraCols);
			maxMemory = std::max(maxMemory, memoryNeeded);
		}
		else
		{
			float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + proj_size_scaling * float(numRows) / float(params.numRows) * params.projectionDataSize(extraCols);
			//float memoryNeeded_A = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numRows) / float(params.numRows) * params.projectionDataSize(extraCols);
			//float memoryNeeded_B = 2.0 * float(numRows) / float(params.numRows) * params.projectionDataSize(extraCols);
			//float memoryNeeded = std::max(memoryNeeded_A, memoryNeeded_B);
			maxMemory = std::max(maxMemory, memoryNeeded);
		}
	}
	return maxMemory + params.get_extraMemoryReserved();
}

float tomographicModels::backproject_memoryRequired_splitViews(int numSlicesPerChunk, bool doFBP)
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

		if (doFBP)
		{
			float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + 2.0 * float(numViews) / float(params.numAngles) * params.projectionDataSize();
			maxMemory = std::max(maxMemory, memoryNeeded);
		}
		else
		{
			float memoryNeeded = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
			//float memoryNeeded_A = float(numSlices) / float(params.numZ) * params.volumeDataSize() + float(numViews) / float(params.numAngles) * params.projectionDataSize();
			//float memoryNeeded_B = 2.0 * float(numViews) / float(params.numAngles) * params.projectionDataSize();
			//float memoryNeeded = std::max(memoryNeeded_A, memoryNeeded_B);
			maxMemory = std::max(maxMemory, memoryNeeded);
		}
	}
	return maxMemory + params.get_extraMemoryReserved();
}

bool tomographicModels::backproject_FBP_multiGPU_splitViews(float* g, float* f, bool doFBP)
{
#ifndef __USE_CPU
	//return false;
	if (params.volumeDimensionOrder != parameters::ZYX || params.isSymmetric())
		return false;

	int numProjectionData = 1;
	int numVolumeData = 1;
	if (doFBP)
		numProjectionData = 2; // need an extra for texture memory

	// if there is sufficient memory for everything and either only one GPU is specified or is a small operation, don't separate into chunks
	//int numSlicesPerChunk = std::min(64, params.numZ);
	int numSlicesPerChunk = std::max(1, int(ceil(float(params.numZ) / std::max(2.0, double(params.whichGPUs.size())))));
	numSlicesPerChunk = std::min(numSlicesPerChunk, maxSlicesForChunking);
	int numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	if (params.hasSufficientGPUmemory(true, 0, numProjectionData, numVolumeData) == false)
	{
		float memAvailable = getAvailableGPUmemory(params.whichGPUs);
		float memNeeded = backproject_memoryRequired_splitViews(numSlicesPerChunk, doFBP);

		while (memAvailable < memNeeded)
		{
			numSlicesPerChunk = numSlicesPerChunk / 2;
			if (numSlicesPerChunk <= 1)
				return false;
			memNeeded = backproject_memoryRequired_splitViews(numSlicesPerChunk, doFBP);
		}
		numChunks = std::max(1, int(ceil(float(params.numZ) / float(numSlicesPerChunk))));
	}
	else if (int(params.whichGPUs.size()) <= 1 || params.requiredGPUmemory(0, numProjectionData, numVolumeData) <= params.chunkingMemorySizeThreshold)
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
		{
			LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": FBP: z-slices: (" << firstSlice << ", " << lastSlice << "), views = (" << viewRange[0] << ", " << viewRange[1] << ")" << std::endl;
			FBP.execute(g_chunk, f_chunk, &chunk_params, true);
		}
		else
		{
			LOG(logDEBUG, className, "") << "GPU " << chunk_params.whichGPU << ": backprojection: z-slices: (" << firstSlice << ", " << lastSlice << "), views = (" << viewRange[0] << ", " << viewRange[1] << ")" << std::endl;
			proj.backproject(g_chunk, f_chunk, &chunk_params, true);
		}
	}
	return true;
#else
	return false;
#endif
}

bool tomographicModels::doFBP(float* g, float* f, bool data_on_cpu)
{
	//*
	if (params.offsetScan)
	{
		if (zeroPadForOffsetScan_numberOfColsToAdd(&params) <= 0)
			params.offsetScan = false;
	}
	//*/
	if (params.helicalPitch != 0.0 && params.tiltAngle != 0.0)
	{
		printf("Error: current implementation of helical FBP cannot handle detector tilt!\n");
		return false;
	}

	//printf("v range: %f to %f\n", params.v(0), params.v(params.numRows-1));
	if (data_on_cpu == true && FBP_multiGPU(g, f) == true)
		return true;
	else
	{
		parameters params_local = params;
		return FBP.execute(g, f, &params_local, data_on_cpu);
		//return FBP.execute(g, f, &params, data_on_cpu);
	}
}

bool tomographicModels::sensitivity(float* f, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.muSpecified() == true || params.isSymmetric() == true || (params.geometry == parameters::MODULAR && usingSFprojectorsForModularBeam(&params) == false))
	{
		if (params.whichGPU < 0 || data_on_cpu == true)
		{
			float* g = params.setToConstant(NULL, uint64(params.numAngles) * uint64(params.numRows) * uint64(params.numCols), 1.0);
			bool retVal = backproject(g, f, data_on_cpu);
			replaceZeros_cpu(f, params.numX, params.numY, params.numZ, 1.0);
			free(g);
			return retVal;
		}
		else
		{
			float* dev_g = 0;
			if (cudaMalloc((void**)&dev_g, uint64(params.numAngles) * uint64(params.numRows) * uint64(params.numCols) * sizeof(float)) != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc(projection[%d,%d,%d]) failed!\n", params.numAngles, params.numRows, params.numCols);
				return false;
			}
			setToConstant(dev_g, 1.0, make_int3(params.numAngles, params.numRows, params.numCols), params.whichGPU);
			bool retVal = backproject(dev_g, f, false);
			replaceZeros(f, make_int3(params.numX, params.numY, params.numZ), params.whichGPU, 1.0);
			cudaFree(dev_g);
			return retVal;
		}
	}

	if (params.whichGPU < 0)
		return sensitivity_CPU(f, &params);
	else
	{
		if (data_on_cpu)
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
					int numSlicesPerChunk = std::max(1, params.numZ / 2);
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
						int lastSlice = std::min(firstSlice + numSlicesPerChunk - 1, params.numZ - 1);
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
				return sensitivity_gpu(f, &params, data_on_cpu);
		}
		else
			return sensitivity_gpu(f, &params, data_on_cpu);
	}
#else
	if (params.muSpecified() == true || params.isSymmetric() == true || (params.geometry == parameters::MODULAR && usingSFprojectorsForModularBeam(&params) == false))
	{
		float* g = params.setToConstant(NULL, uint64(params.numAngles) * uint64(params.numRows) * uint64(params.numCols), 1.0);
		bool retVal = backproject(g, f, data_on_cpu);
		replaceZeros_cpu(f, params.numX, params.numY, params.numZ, 1.0);
		free(g);
		return retVal;
	}
	else
		return sensitivity_CPU(f, &params);
#endif
}

float tomographicModels::get_FBPscalar()
{
	return FBPscalar(&params);
}

bool tomographicModels::set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float tiltAngle, float helicalPitch)
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
	params.set_tiltAngle(tiltAngle);
	if (params.geometryDefined())
	{
		params.set_offsetScan(params.offsetScan);
		return true;
	}
	else
		return false;
}

bool tomographicModels::set_coneparallel(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float helicalPitch)
{
	params.geometry = parameters::CONE_PARALLEL;
	params.detectorType = parameters::FLAT;
	params.tiltAngle = 0.0;
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
	if (params.geometryDefined())
	{
		params.set_tiltAngle(0.0);
		params.set_offsetScan(params.offsetScan);
		return true;
	}
	else
		return false;
}

bool tomographicModels::set_fanbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau)
{
	params.detectorType = parameters::FLAT;
	params.tiltAngle = 0.0;

	params.geometry = parameters::FAN;
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
	if (params.geometryDefined())
	{
		params.set_helicalPitch(0.0);
		params.set_tiltAngle(0.0);
		params.set_offsetScan(params.offsetScan);
		return true;
	}
	else
		return false;
}

bool tomographicModels::set_parallelbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	params.detectorType = parameters::FLAT;
	params.tiltAngle = 0.0;

	params.geometry = parameters::PARALLEL;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.set_angles(phis, numAngles);
	if (params.geometryDefined())
	{
		params.set_helicalPitch(0.0);
		params.set_tiltAngle(0.0);
		params.set_offsetScan(params.offsetScan);
		return true;
	}
	else
		return false;
}

bool tomographicModels::set_modularbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	params.detectorType = parameters::FLAT;
	params.tau = 0.0;
	params.tiltAngle = 0.0;
	params.helicalPitch = 0.0;

	params.geometry = parameters::MODULAR;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = float(0.5*float(numCols-1));
	params.centerRow = float(0.5*float(numRows-1));
	params.set_sourcesAndModules(sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in, numAngles);
	if (params.geometryDefined())
	{
		params.set_helicalPitch(0.0);
		params.set_tiltAngle(0.0);
		params.set_offsetScan(params.offsetScan);
		return true;
	}
	else
		return false;
}

bool tomographicModels::set_flatDetector()
{
	params.detectorType = parameters::FLAT;
	return true;
}

bool tomographicModels::set_curvedDetector()
{
	if (params.geometry != parameters::CONE)
	{
		printf("Error: curved detector only defined for cone-beam geometries\n");
		return false;
	}
	else
	{
		params.detectorType = parameters::CURVED;
		params.tiltAngle = 0.0;
		return true;
	}
}

bool tomographicModels::set_centerCol(float centerCol)
{
	if (params.geometry == parameters::MODULAR)
	{
		printf("Error: centerCol not defined for modular-beam geometry.  Move moduleCenters instead\n");
		return false;
	}
	else
	{
		params.centerCol = centerCol;
		return true;
	}
}

bool tomographicModels::set_centerRow(float centerRow)
{
	if (params.geometry == parameters::MODULAR)
	{
		printf("Error: centerRow not defined for modular-beam geometry.  Move moduleCenters instead\n");
		return false;
	}
	else
	{
		params.centerRow = centerRow;
		return true;
	}
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

int tomographicModels::number_of_gpus()
{
#ifndef __USE_CPU
	return numberOfGPUs();
#else
	return 0;
#endif
}

int tomographicModels::get_gpus(int* list_of_gpus)
{
#ifndef __USE_CPU
	int retVal = params.whichGPUs.size();
	if (list_of_gpus != NULL)
	{
		for (int i = 0; i < retVal; i++)
			list_of_gpus[i] = params.whichGPUs[i];
	}
	return retVal;
#else
	return 0;
#endif
}

bool tomographicModels::set_GPU(int whichGPU)
{
#ifndef __USE_CPU
	if (numberOfGPUs() <= 0)
		params.whichGPU = -1;
	else
		params.whichGPU = whichGPU;
	params.whichGPUs.clear();
	params.whichGPUs.push_back(whichGPU);
	return true;
#else
	return false;
#endif
}

bool tomographicModels::set_GPUs(int* whichGPUs, int N)
{
#ifndef __USE_CPU
	if (whichGPUs == NULL || N <= 0)
		return false;
	params.whichGPUs.clear();
	for (int i = 0; i < N; i++)
		params.whichGPUs.push_back(whichGPUs[i]);
	params.whichGPU = params.whichGPUs[0];
	return true;
#else
	return false;
#endif
}

int tomographicModels::get_GPU()
{
	return params.whichGPU;
}

bool tomographicModels::set_projector(int which)
{
	if (which == parameters::SEPARABLE_FOOTPRINT)
		params.whichProjector = parameters::SEPARABLE_FOOTPRINT;
	else if (which == parameters::VOXEL_DRIVEN)
		params.whichProjector = parameters::VOXEL_DRIVEN;
	else
	{
		printf("Error: currently only SF and VD projectors are implemented!\n");
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
	if (whichRampFilter < 0 || whichRampFilter > 12)
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

bool tomographicModels::flipAttenuationMapSign(bool data_on_cpu)
{
	if (params.muCoeff != 0.0)
	{
		params.muCoeff *= -1.0;
		//printf("flipping sign of muCoeff (%f)\n", params.muCoeff);
	}
	else if (params.mu != NULL && params.numZ > 0 && params.numY > 0 && params.numX > 0)
	{
		//printf("flipping sign of the attenuation map\n");
		scale_cpu(params.mu, -1.0, params.numZ, params.numY, params.numX);
		//scale(params.mu, -1.0, make_int3(params.numZ, params.numY, params.numX), params.whichGPU);
	}
	return true;
}

bool tomographicModels::projectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.project(g, f, &tempParams, data_on_cpu);
}

bool tomographicModels::backprojectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.backproject(g, f, &tempParams, data_on_cpu);
}

bool tomographicModels::projectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.project(g, f, &tempParams, data_on_cpu);
}

bool tomographicModels::backprojectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.backproject(g, f, &tempParams, data_on_cpu);
}

bool tomographicModels::projectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.project(g, f, &tempParams, data_on_cpu);
}

bool tomographicModels::backprojectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
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

	return proj.backproject(g, f, &tempParams, data_on_cpu);
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

bool tomographicModels::set_tiltAngle(float tiltAngle)
{
	return params.set_tiltAngle(tiltAngle);
}

float tomographicModels::get_tiltAngle()
{
	return params.tiltAngle;
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

bool tomographicModels::applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	if (x == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || LUT == NULL || sampleRate <= 0.0 || numSamples <= 0)
	{
		printf("Error: invalid input!\n");
		printf("%d, %d, %d\n", N_1, N_2, N_3);
		printf("%f, %d\n", sampleRate, numSamples);
		return false;
	}

#ifndef __USE_CPU
	// This is a simple algorithm, so only run it on the GPU if the data is already there
	if (data_on_cpu == false)
	{
		return applyTransferFunction_gpu(x, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, params.whichGPU, data_on_cpu);
		//printf("Error: method currently only implemented for data on CPU!\n");
		//return false;
	}
	//return applyTransferFunction_gpu(x, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, params.whichGPU, data_on_cpu);
#endif
	float lastSample = float(numSamples - 1) * sampleRate + firstSample;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* x_2D = &x[uint64(i)*uint64(N_2)* uint64(N_3)];
		for (int j = 0; j < N_2; j++)
		{
			float* x_1D = &x_2D[uint64(j)*uint64(N_3)];
			for (int k = 0; k < N_3; k++)
			{
				float curVal = x_1D[k];
				if (curVal >= lastSample)
				{
					float slope = (LUT[numSamples - 1] - LUT[numSamples - 2]) / sampleRate;
					x_1D[k] = LUT[numSamples - 1] + slope * (curVal - lastSample);
				}
				else if (curVal <= firstSample)
					x_1D[k] = firstSample;
				else
				{
					float ind = curVal / sampleRate - firstSample;
					int ind_low = int(ind);
					float d = ind - float(ind_low);
					x_1D[k] = float((1.0 - d) * LUT[ind_low] + d * LUT[ind_low + 1]);
				}
			}
		}
	}
	return true;
}

bool tomographicModels::beam_hardening_heel_effect(float* g, float* anode_normal, float* LUT, float* takeOffAngles, int numSamples, int numAngles, float sampleRate, float firstSample, bool data_on_cpu)
{
	if (g == NULL || params.geometryDefined() == false || anode_normal == NULL || LUT == NULL || sampleRate <= 0.0 || numSamples <= 0 || numAngles <= 1)
	{
		printf("Error: invalid input!\n");
		return false;
	}
	else if (data_on_cpu == false)
	{
		printf("Error: this function is currently only supported for CPU processing.\n");
		return false;
	}
	else if (params.geometry == parameters::PARALLEL || params.geometry == parameters::CONE_PARALLEL || params.geometry == parameters::FAN)
	{
		printf("Error: this function does not support this CT geometry\n");
		return false;
	}

	//for (int i = 0; i < numSamples; i++)
	//	printf("%f\n", LUT[i]);

	bool normalizeConeAndFanCoordinateFunctions_save = params.normalizeConeAndFanCoordinateFunctions;
	params.normalizeConeAndFanCoordinateFunctions = false;

	float firstAngle = takeOffAngles[0];
	float lastAngle = takeOffAngles[numAngles - 1];
	float T_angle = takeOffAngles[1] - takeOffAngles[0];

	float lastSample = float(numSamples - 1) * sampleRate + firstSample;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < params.numAngles; i++)
	{
		float sdd_cur = params.source_to_detector_distance(i);
		float* aProj = &g[uint64(i) * uint64(params.numRows) * uint64(params.numCols)];
		for (int j = 0; j < params.numRows; j++)
		{
			float* aLine = &aProj[uint64(j) * uint64(params.numCols)];
			for (int k = 0; k < params.numCols; k++)
			{
				float x = aLine[k];

				// Determine takeoff angle for this ray
				float iAngle = 0.5*float(numAngles-1);
				int iAngle_lo = numAngles / 2;
				int iAngle_hi = numAngles / 2;
				float dAngle = 0.0;

				float v = params.v(j,i) / sdd_cur;
				float u = params.u(k,i) / sdd_cur;
				float lineLength = sqrt(1.0 + u * u + v * v);
				float takeoffAngle = 90.0 - acos((u * anode_normal[0] + 1.0 * anode_normal[1] + v * anode_normal[2]) / lineLength) * 180.0 / PI;
				//float takeoffAngle = acos((u * anode_normal[0] + 1.0 * anode_normal[1] + v * anode_normal[2]) / lineLength) * 180.0 / PI;
				//if (i == 0 && j == params.numRows/2)
				//	printf("takeoffAngle = %f (u=%f)\n", takeoffAngle, u);

				if (takeoffAngle <= firstAngle)
				{
					iAngle_lo = 0;
					iAngle_hi = 0;
					dAngle = 0.0;
				}
				else if (takeoffAngle >= lastAngle)
				{
					iAngle_lo = numAngles-1;
					iAngle_hi = numAngles-1;
					dAngle = 0.0;
				}
				else
				{
					// takeoffAngle = firstAngle + iAngle*T_angle
					iAngle = (takeoffAngle - firstAngle) / T_angle;
					iAngle_lo = int(floor(iAngle));
					iAngle_hi = int(ceil(iAngle));
					dAngle = iAngle - float(iAngle_lo);
				}

				float* LUT_phi_lo = &LUT[iAngle_lo * numSamples];
				float* LUT_phi_hi = &LUT[iAngle_hi * numSamples];

				if (x >= lastSample)
				{
					float slope_lo = (LUT_phi_lo[numSamples - 1] - LUT_phi_lo[numSamples - 2]) / sampleRate;
					float slope_hi = (LUT_phi_hi[numSamples - 1] - LUT_phi_hi[numSamples - 2]) / sampleRate;
					float x_lo = LUT_phi_lo[numSamples - 1] + slope_lo * (x - lastSample);
					float x_hi = LUT_phi_hi[numSamples - 1] + slope_hi * (x - lastSample);
					x = (1.0 - dAngle) * x_lo + dAngle * x_hi;
				}
				else if (x <= firstSample)
					x = firstSample;
				else
				{
					float ind = x / sampleRate - firstSample;
					int ind_low = int(ind);
					float d = ind - float(ind_low);
					float x_lo = float((1.0 - d) * LUT_phi_lo[ind_low] + d * LUT_phi_lo[ind_low + 1]);
					float x_hi = float((1.0 - d) * LUT_phi_hi[ind_low] + d * LUT_phi_hi[ind_low + 1]);
					x = (1.0 - dAngle) * x_lo + dAngle * x_hi;
				}

				aLine[k] = x;
			}
		}
	}

	params.normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

	return true;
}

bool tomographicModels::applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	if (x == NULL || y == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || LUT == NULL || sampleRate <= 0.0 || numSamples <= 0)
	{
		printf("Error: invalid input!\n");
		printf("%d, %d, %d\n", N_1, N_2, N_3);
		printf("%f, %d\n", sampleRate, numSamples);
		return false;
	}

#ifndef __USE_CPU
	// This is a simple algorithm, so only run it on the GPU if the data is already there
	if (data_on_cpu == false)
	{
		return applyDualTransferFunction_gpu(x, y, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, params.whichGPU, data_on_cpu);
		//printf("Error: method currently only implemented for data on CPU!\n");
		//return false;
	}
	//return applyDualTransferFunction_gpu(x, y, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, params.whichGPU, data_on_cpu);
#endif
	float lastSample = float(numSamples - 1) * sampleRate + firstSample;

	float* LUT_1 = &LUT[0];
	float* LUT_2 = &LUT[numSamples * numSamples];

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* x_2D = &x[uint64(i) * uint64(N_2) * uint64(N_3)];
		float* y_2D = &y[uint64(i) * uint64(N_2) * uint64(N_3)];
		for (int j = 0; j < N_2; j++)
		{
			float* x_1D = &x_2D[uint64(j) * uint64(N_3)];
			float* y_1D = &y_2D[uint64(j) * uint64(N_3)];
			for (int k = 0; k < N_3; k++)
			{
				float curVal_1 = x_1D[k];
				float curVal_2 = y_1D[k];

				int ind_lo_1, ind_hi_1;
				float d_1;
				if (curVal_1 >= lastSample)
				{
					ind_lo_1 = numSamples - 1;
					ind_hi_1 = numSamples - 1;
					d_1 = 0.0;
				}
				else if (curVal_1 <= firstSample)
				{
					ind_lo_1 = 0;
					ind_hi_1 = 0;
					d_1 = 0.0;
				}
				else
				{
					float ind = curVal_1 / sampleRate - firstSample;
					ind_lo_1 = int(ind);
					ind_hi_1 = ind_lo_1 + 1;
					d_1 = ind - float(ind_lo_1);
				}

				int ind_lo_2, ind_hi_2;
				float d_2;
				if (curVal_2 >= lastSample)
				{
					ind_lo_2 = numSamples - 1;
					ind_hi_2 = numSamples - 1;
					d_2 = 0.0;
				}
				else if (curVal_2 <= firstSample)
				{
					ind_lo_2 = 0;
					ind_hi_2 = 0;
					d_2 = 0.0;
				}
				else
				{
					float ind = curVal_2 / sampleRate - firstSample;
					ind_lo_2 = int(ind);
					ind_hi_2 = ind_lo_2 + 1;
					d_2 = ind - float(ind_lo_2);
				}

				float partA_1 = float((1.0 - d_2)* LUT_1[ind_lo_1 * numSamples + ind_lo_2] + d_2* LUT_1[ind_lo_1 * numSamples + ind_hi_2]);
				float partB_1 = float((1.0 - d_2) * LUT_1[ind_hi_1 * numSamples + ind_lo_2] + d_2 * LUT_1[ind_hi_1 * numSamples + ind_hi_2]);

				float partA_2 = float((1.0 - d_2) * LUT_2[ind_lo_1 * numSamples + ind_lo_2] + d_2 * LUT_2[ind_lo_1 * numSamples + ind_hi_2]);
				float partB_2 = float((1.0 - d_2) * LUT_2[ind_hi_1 * numSamples + ind_lo_2] + d_2 * LUT_2[ind_hi_1 * numSamples + ind_hi_2]);

				x_1D[k] = float((1.0 - d_1) * partA_1 + d_1 * partB_1);
				y_1D[k] = float((1.0 - d_1) * partA_2 + d_1 * partB_2);

				/*
				if (curVal_1 >= lastSample)
				{
					float slope = (LUT[numSamples - 1] - LUT[numSamples - 2]) / sampleRate;
					x_1D[k] = LUT[numSamples - 1] + slope * (curVal - lastSample);
				}
				else if (curVal <= firstSample)
					x_1D[k] = firstSample;
				else
				{
					float ind = curVal / sampleRate - firstSample;
					int ind_low = int(ind);
					float d = ind - float(ind_low);
					x_1D[k] = (1.0 - d) * LUT[ind_low] + d * LUT[ind_low + 1];
				}
				//*/
			}
		}
	}
	return true;
}

bool tomographicModels::convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool data_on_cpu)
{
	if (f_L == NULL || f_H == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || sigma_L == NULL || sigma_H == NULL)
	{
		printf("Error: invalid input!\n");
		return false;
	}

	// This is a simple algorithm, so only run it on the GPU if the data is already there
	if (data_on_cpu == false)
	{
		printf("Error: method currently only implemented for data on CPU!\n");
		return false;
	}

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* slice_L = &f_L[uint64(i) * uint64(N_2) * uint64(N_3)];
		float* slice_H = &f_H[uint64(i) * uint64(N_2) * uint64(N_3)];
		for (int j = 0; j < N_2; j++)
		{
			float* line_L = &slice_L[uint64(j) * uint64(N_3)];
			float* line_H = &slice_H[uint64(j) * uint64(N_3)];
			for (int k = 0; k < N_3; k++)
			{
				float mu_L = line_L[k];
				float mu_H = line_H[k];

				if (mu_L <= 0.0 || mu_H <= 0.0)
				{
					line_L[k] = float(7.31231243248); // effective-Z of air
					line_H[k] = 0.0;
				}
				else
				{
					float theRatio = mu_L / mu_H;

					float prevError = theRatio - sigma_L[0] / sigma_H[0];
					float curError;
					int Z;
					for (Z = 2; Z <= 100; Z++)
					{
						curError = theRatio - sigma_L[Z - 1] / sigma_H[Z - 1];
						if (prevError * curError <= 0.0) // change of sign, so must have root between Z-1 and Z
							break;
						else if (fabs(prevError) < fabs(curError)) // getting worse, so just quit
							break;
						prevError = curError;
					}
					Z = std::min(100 - 2, std::max(1, Z - 1)); // min(98, max(1,Z-1))

					// Now solution is between Z and Z+1

					// Original Calculation
					//double d = (theRatio*sigma(Z+1,ref_H) - sigma(Z+1,ref_L)) / (sigma(Z,ref_L) - sigma(Z+1,ref_L) - theRatio*( sigma(Z,ref_H) - sigma(Z+1,ref_H) ));

					// Revised calculation
					float d;
					if (Z == 1 || 0.5 * (sigma_L[Z - 1 - 1] / sigma_H[Z - 1 - 1] + sigma_L[Z + 1 - 1] / sigma_H[Z + 1 - 1]) > sigma_L[Z - 1] / sigma_H[Z - 1])
						d = (theRatio - sigma_L[Z + 1 - 1] / sigma_H[Z + 1 - 1]) / (sigma_L[Z - 1] / sigma_H[Z - 1] - sigma_L[Z + 1 - 1] / sigma_H[Z + 1 - 1]);
					else
						d = (theRatio * sigma_H[Z + 1 - 1] - sigma_L[Z + 1 - 1]) / (sigma_L[Z - 1] - sigma_L[Z + 1 - 1] - theRatio * (sigma_H[Z - 1] - sigma_H[Z + 1 - 1]));

					d = std::max(float(0.0), std::min(float(1.0), d));
					float Ze = float(Z) + float(1.0) - d;

					int Z_lo = int(Ze);
					int Z_hi = Z_lo + 1;
					d = Ze - float(Z_lo);

					float sigma_Ze_L = (float(1.0) - d) * sigma_L[Z_lo - 1] + d * sigma_L[Z_hi - 1];
					float sigma_Ze_H = (float(1.0) - d) * sigma_H[Z_lo - 1] + d * sigma_H[Z_hi - 1];
					float rhoe = (sigma_Ze_L * mu_L + sigma_Ze_H * mu_H) / (sigma_Ze_L * sigma_Ze_L + sigma_Ze_H * sigma_Ze_H);
					if (mu_L == 0.0 || mu_H == 0.0)
						rhoe = 0.0;
					else if (isnan(rhoe) == 1)
						rhoe = 0.0;

					line_L[k] = Ze;
					line_H[k] = rhoe;
				}
			}
		}
	}
	return true;
}

bool tomographicModels::HighPassFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* f_chunk = &f[uint64(sliceStart) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				highPassFilter(f_chunk, sliceEnd - sliceStart + 1, N_2, N_3, FWHM, 2, 0, true, whichGPU);
			}

			return true;
		}
	}
	else
		return highPassFilter(f, N_1, N_2, N_3, FWHM, 2, 0, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::BlurFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* f_chunk = &f[uint64(sliceStart) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				blurFilter(f_chunk, sliceEnd - sliceStart + 1, N_2, N_3, FWHM, 2, 0, true, whichGPU);
			}

			return true;
		}
	}
	else
		return blurFilter(f, N_1, N_2, N_3, FWHM, 2, 0, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false || FWHM > 2.0)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			float* f_out = f;
			float* f_in = (float*)malloc(sizeof(float) * numElements);
			equal_cpu(f_in, f_out, N_1, N_2, N_3);

			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_out_chunk = &f_out[uint64(sliceStart) * uint64(N_2 * N_3)];
				float* f_in_chunk = &f_in[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				blurFilter(f_in_chunk, numSlices_pad, N_2, N_3, FWHM, 3, 0, true, whichGPU, sliceStart_relative, sliceEnd_relative, f_out_chunk);
			}
			free(f_in);

			return true;
		}
	}
	else
		return blurFilter(f, N_1, N_2, N_3, FWHM, 3, 0, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::HighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false || FWHM > 2.0)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			float* f_out = f;
			float* f_in = (float*)malloc(sizeof(float) * numElements);
			equal_cpu(f_in, f_out, N_1, N_2, N_3);

			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_out_chunk = &f_out[uint64(sliceStart) * uint64(N_2 * N_3)];
				float* f_in_chunk = &f_in[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				highPassFilter(f_in_chunk, numSlices_pad, N_2, N_3, FWHM, 3, 0, true, whichGPU, sliceStart_relative, sliceEnd_relative, f_out_chunk);
			}
			free(f_in);

			return true;
		}
	}
	else
		return highPassFilter(f, N_1, N_2, N_3, FWHM, 3, 0, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::badPixelCorrection(float* g, int N_1, int N_2, int N_3, float* badPixelMap, int w, bool data_on_cpu)
{
	if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numProj = 1.0;
	if (data_on_cpu)
		numProj = 1.0;
	else
		numProj = 0.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (data_on_cpu == true && (getAvailableGPUmemory(params.whichGPU) < dataSize || params.whichGPUs.size() > 1))
	{
		// do chunking
		//int numSlices = std::min(N_1, maxSlicesForChunking);
		int numSlices = int(ceil(double(N_1) / double(int(params.whichGPUs.size()))));
		while (getAvailableGPUmemory(params.whichGPU) < double(numSlices) / double(N_1) * dataSize)
		{
			numSlices = numSlices / 2;
			if (numSlices < 1)
			{
				numSlices = 1;
				break;
			}
		}
		int numChunks = int(ceil(float(N_1) / float(numSlices)));

		//printf("number of slices per chunk: %d\n", numSlices);

		omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
		#pragma omp parallel for schedule(dynamic)
		for (int ichunk = 0; ichunk < numChunks; ichunk++)
		{
			int sliceStart = ichunk * numSlices;
			int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

			float* g_chunk = &g[uint64(sliceStart) * uint64(N_2 * N_3)];
			int whichGPU = params.whichGPUs[omp_get_thread_num()];

			parameters params_chunk = params;
			params_chunk.numAngles = sliceEnd - sliceStart + 1;
			params_chunk.numRows = N_2;
			params_chunk.numCols = N_3;
			//params_chunk.removeProjections(sliceStart, sliceEnd);
			//params_chunk.numAngles = sliceEnd - sliceStart + 1;
			params_chunk.whichGPU = whichGPU;

			badPixelCorrection_gpu(g_chunk, &params_chunk, badPixelMap, w, data_on_cpu);
		}
		return true;
	}
	else
	{
		parameters params_chunk = params;
		params_chunk.numAngles = N_1;
		params_chunk.numRows = N_2;
		params_chunk.numCols = N_3;
		//params_chunk.removeProjections(sliceStart, sliceEnd);
		//params_chunk.numAngles = sliceEnd - sliceStart + 1;

		return badPixelCorrection_gpu(g, &params_chunk, badPixelMap, w, data_on_cpu);
	}
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::MedianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* f_chunk = &f[uint64(sliceStart) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				medianFilter2D(f_chunk, sliceEnd - sliceStart + 1, N_2, N_3, threshold, w, signalThreshold, true, whichGPU);
			}

			return true;
		}
	}
	else
		return medianFilter2D(f, N_1, N_2, N_3, threshold, w, signalThreshold, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			float* f_out = f;
			float* f_in = (float*)malloc(sizeof(float)* numElements);
			equal_cpu(f_in, f_out, N_1, N_2, N_3);

			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_out_chunk = &f_out[uint64(sliceStart) * uint64(N_2 * N_3)];
				float* f_in_chunk = &f_in[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				medianFilter(f_in_chunk, numSlices_pad, N_2, N_3, threshold, w, signalThreshold, true, whichGPU, sliceStart_relative, sliceEnd_relative, f_out_chunk);
			}
			free(f_in);

			return true;
		}
	}
	else
		return medianFilter(f, N_1, N_2, N_3, threshold, w, signalThreshold, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::MeanOrVarianceFilter(float* f, int N_1, int N_2, int N_3, int r, int order, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			float* f_out = f;
			float* f_in = (float*)malloc(sizeof(float) * numElements);
			equal_cpu(f_in, f_out, N_1, N_2, N_3);

			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_out_chunk = &f_out[uint64(sliceStart) * uint64(N_2 * N_3)];
				float* f_in_chunk = &f_in[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				//medianFilter(f_in_chunk, numSlices_pad, N_2, N_3, threshold, w, true, whichGPU, sliceStart_relative, sliceEnd_relative, f_out_chunk);
				momentFilter(f, N_1, N_2, N_3, r, order, data_on_cpu, whichGPU, sliceStart_relative, sliceEnd_relative, f_out_chunk);
			}
			free(f_in);

			return true;
		}
	}
	else
		return momentFilter(f, N_1, N_2, N_3, r, order, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::BilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (f == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || spatialFWHM <= 0.0 || intensityFWHM <= 0.0)
	{
		printf("Error: BilateralFilter invalid arguments!\n");
		return false;
	}

	int numVol;
	if (data_on_cpu)
		numVol = 2;
	else
		numVol = 1;
	if (scale > 1.0)
		numVol += 1;
	double memNeeded = 4.0 * double(numVol) * double(N_1) * double(N_2) * double(N_3) / pow(2.0, 30.0);
	if (getAvailableGPUmemory(params.whichGPU) < memNeeded)
	{
		printf("Error: BilateralFilter not enough GPU memory available for this operation!\n");
		printf("GPU memory needed: %f GB\n", memNeeded);
		printf("GPU memory available: %f GB\n", getAvailableGPUmemory(params.whichGPU));
		printf("It is possible to break this calculation into smaller pieces which would enable this algorithm to work.\n");
		printf("We plan to fix this in a future release, but if you encountered this error, please submit an issue on the github page.\n");
		return false;
	}
	else
		return scaledBilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, scale, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::PriorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* prior, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (f == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || spatialFWHM <= 0.0 || intensityFWHM <= 0.0)
	{
		printf("Error: PriorBilateralFilter invalid arguments!\n");
		return false;
	}

	int numVol;
	if (data_on_cpu)
		numVol = 3;
	else
		numVol = 1;
	double memNeeded = 4.0 * double(numVol) * double(N_1) * double(N_2) * double(N_3) / pow(2.0, 30.0);
	if (getAvailableGPUmemory(params.whichGPU) < memNeeded)
	{
		printf("Error: PriorBilateralFilter not enough GPU memory available for this operation!\n");
		printf("GPU memory needed: %f GB\n", memNeeded);
		printf("GPU memory available: %f GB\n", getAvailableGPUmemory(params.whichGPU));
		printf("It is possible to break this calculation into smaller pieces which would enable this algorithm to work.\n");
		printf("We plan to fix this in a future release, but if you encountered this error, please submit an issue on the github page.\n");
		return false;
	}
	else
		return priorBilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, prior, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::GuidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (f == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || r <= 0 || epsilon <= 0.0)
	{
		printf("Error: GuidedFilter invalid arguments!\n");
		return false;
	}

	int numVol;
	if (data_on_cpu)
		numVol = 3;
	else
		numVol = 2;
	double memNeeded = 4.0 * double(numVol) * double(N_1) * double(N_2) * double(N_3) / pow(2.0, 30.0);
	if (getAvailableGPUmemory(params.whichGPU) < memNeeded)
	{
		printf("Error: GuidedFilter not enough GPU memory available for this operation!\n");
		printf("GPU memory needed: %f GB\n", memNeeded);
		printf("GPU memory available: %f GB\n", getAvailableGPUmemory(params.whichGPU));
		printf("It is possible to break this calculation into smaller pieces which would enable this algorithm to work.\n");
		printf("We plan to fix this in a future release, but if you encountered this error, please submit an issue on the github page.\n");
		return false;
	}
	else
		return guidedFilter(f, N_1, N_2, N_3, r, epsilon, numIter, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (f == NULL || dictionary == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || numElements <= 0 || N_d1 <= 0 || N_d2 <= 0 || N_d3 <= 0)
	{
		printf("Error: dictionaryDenoising invalid arguments!\n");
		return false;
	}
	if (sparsityThreshold < 1 || sparsityThreshold > numElements || epsilon < 0.0)
	{
		printf("Error: dictionaryDenoising invalid arguments!\n");
		return false;
	}
	double memNeeded = matchingPursuit_memory(N_1, N_2, N_3, numElements, N_d1, N_d2, N_d3, sparsityThreshold);
	//printf("GPU memory needed: %f GB\n", memNeeded);
	if (getAvailableGPUmemory(params.whichGPU) < memNeeded)
	{
		printf("Error: dictionaryDenoising not enough GPU memory available for this operation!\n");
		printf("GPU memory needed: %f GB\n", memNeeded);
		printf("GPU memory available: %f GB\n", getAvailableGPUmemory(params.whichGPU));
		printf("It is possible to break this calculation into smaller pieces which would enable this algorithm to work.\n");
		printf("We plan to fix this in a future release, but if you encountered this error, please submit an issue on the github page.\n");
		return false;
	}
	else
		return matchingPursuit(f, N_1, N_2, N_3, dictionary, numElements, N_d1, N_d2, N_d3, epsilon, sparsityThreshold, data_on_cpu, params.whichGPU);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

float tomographicModels::TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return 0.0;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return 0.0;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices)/double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			float* costs = (float*)calloc(size_t(numChunks), sizeof(float));
			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1-1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_chunk = &f[uint64(sliceStart_pad) * uint64(N_2*N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				costs[ichunk] = anisotropicTotalVariation_cost(f_chunk, numSlices_pad, N_2, N_3, delta, beta, p, true, whichGPU, sliceStart_relative, sliceEnd_relative, params.numTVneighbors);
			}

			float retVal = 0.0;
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
				retVal += costs[ichunk];

			free(costs);
			return retVal;
		}
	}
	else
		return anisotropicTotalVariation_cost(f, N_1, N_2, N_3, delta, beta, p, data_on_cpu, params.whichGPU, -1, -1, params.numTVneighbors);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool doMean, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	
	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else if (Df == f)
		{
			printf("Error: Insufficient GPU memory for this operation when doing in-place processing!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* Df_chunk = &Df[uint64(sliceStart) * uint64(N_2 * N_3)]; // not padded

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_chunk = &f[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				anisotropicTotalVariation_gradient(f_chunk, Df_chunk, numSlices_pad, N_2, N_3, delta, beta, p, true, whichGPU, sliceStart_relative, sliceEnd_relative, params.numTVneighbors, doMean);
			}

			return true;
		}
	}
	else
		return anisotropicTotalVariation_gradient(f, Df, N_1, N_2, N_3, delta, beta, p, data_on_cpu, params.whichGPU, -1, -1, params.numTVneighbors, doMean);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

float tomographicModels::TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return 0.0;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 3.0;
	else
		numVol = 1.0;
	
	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			float* costs = (float*)calloc(size_t(numChunks), sizeof(float));
			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				int sliceStart_pad = std::max(0, sliceStart - 1);
				int sliceEnd_pad = std::min(N_1 - 1, sliceEnd + 1);
				int numSlices_pad = sliceEnd_pad - sliceStart_pad + 1;

				int sliceStart_relative = sliceStart - sliceStart_pad;
				int sliceEnd_relative = sliceStart_relative + (sliceEnd - sliceStart);

				float* f_chunk = &f[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				float* d_chunk = &d[uint64(sliceStart_pad) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				costs[ichunk] = anisotropicTotalVariation_quadraticForm(f_chunk, d_chunk, numSlices_pad, N_2, N_3, delta, beta, p, true, whichGPU, sliceStart_relative, sliceEnd_relative, params.numTVneighbors);
			}

			float retVal = 0.0;
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
				retVal += costs[ichunk];

			free(costs);
			return retVal;
		}
	}
	else
		return anisotropicTotalVariation_quadraticForm(f, d, N_1, N_2, N_3, delta, beta, p, data_on_cpu, params.whichGPU, -1, -1, params.numTVneighbors);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 3.0;
	else
		numVol = 2.0;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			//printf("diffuse: processing in chunks...\n");
			// will process in chunks
			float* f_0 = (float*)malloc(sizeof(float) * uint64(N_1) * uint64(N_2) * uint64(N_3));
			equal_cpu(f_0, f, N_1, N_2, N_3);
			float* d = (float*)malloc(sizeof(float) * uint64(N_1) * uint64(N_2) * uint64(N_3));
			for (int iter = 0; iter < numIter; iter++)
			{
				TVgradient(f, d, N_1, N_2, N_3, delta, beta, p, doMean, true);
				float num = innerProduct_cpu(d, d, N_1, N_2, N_3);
				float denom = TVquadForm(f, d, N_1, N_2, N_3, delta, beta, p, true);
				if (denom <= 1.0e-16)
					break;
				float stepSize = num / denom;
				scale_cpu(f, 1.0 - stepSize, N_1, N_2, N_3);
				sub_cpu(d, f_0, N_1, N_2, N_3);
				scalarAdd_cpu(f, -stepSize, d, N_1, N_2, N_3);
			}
			free(f_0);
			free(d);
			return true;
		}
	}
	else
		return TVdenoise(f, N_1, N_2, N_3, delta, beta, p, numIter, data_on_cpu, params.whichGPU, params.numTVneighbors, doMean);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::Diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;
	
	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			//printf("diffuse: processing in chunks...\n");
			// will process in chunks
			float* d = (float*)malloc(sizeof(float) * uint64(N_1) * uint64(N_2) * uint64(N_3));
			for (int iter = 0; iter < numIter; iter++)
			{
				TVgradient(f, d, N_1, N_2, N_3, delta, 1.0, p, false, true);
				float num = innerProduct_cpu(d, d, N_1, N_2, N_3);
				float denom = TVquadForm(f, d, N_1, N_2, N_3, delta, 1.0, p, true);
				if (denom <= 1.0e-16)
					break;
				float stepSize = num / denom;
				scalarAdd_cpu(f, -stepSize, d, N_1, N_2, N_3);
			}
			free(d);
			return true;
		}
	}
	else
		return diffuse(f, N_1, N_2, N_3, delta, p, numIter, data_on_cpu, params.whichGPU, params.numTVneighbors);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::rayTrace(float* g, int oversampling, bool data_on_cpu)
{
	/*
	analyticRayTracing simulator;
	return simulator.rayTrace(g, &params, &geometricPhantom, oversampling);
	//*/
	//*
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		if (data_on_cpu == false)
		{
			LOG(logERROR, "", "") << "Error: GPU routines not included in this release!" << std::endl;
			return false;
		}
		analyticRayTracing simulator;
		return simulator.rayTrace(g, &params, &geometricPhantom, oversampling);
	}
	else
	{
		int N_1 = params.numAngles;

		uint64 numElements = params.projectionData_numberOfElements();
		//uint64 numElements_filter = uint64(N_H1) * uint64(N_H2);
		double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
		//uint64 maxElements = 2147483646;

		if (data_on_cpu == true && (getAvailableGPUmemory(params.whichGPU) < dataSize || params.whichGPUs.size() > 1))
		{
			// do chunking
			//int numSlices = std::min(N_1, maxSlicesForChunking);
			int numSlices = int(ceil(double(N_1) / double(int(params.whichGPUs.size()))));
			while (getAvailableGPUmemory(params.whichGPU) < double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* g_chunk = &g[uint64(sliceStart) * uint64(params.numRows * params.numCols)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];

				parameters params_chunk = params;
				params_chunk.removeProjections(sliceStart, sliceEnd);
				//params_chunk.numAngles = sliceEnd - sliceStart + 1;
				params_chunk.whichGPU = whichGPU;

				rayTrace_gpu(g_chunk, &params_chunk, &geometricPhantom, data_on_cpu, oversampling);
			}
			return true;
		}
		else
			return rayTrace_gpu(g, &params, &geometricPhantom, data_on_cpu, oversampling);
	}
#else
	if (data_on_cpu == false)
	{
		LOG(logERROR, "", "") << "Error: GPU routines not included in this release!" << std::endl;
		return false;
	}
	analyticRayTracing simulator;
	return simulator.rayTrace(g, &params, &geometricPhantom, oversampling);
#endif
	//*/
}

bool tomographicModels::rebin_curved(float* g, float* fanAngles, int order)
{
	rebin rebinningRoutines;
	return rebinningRoutines.rebin_curved(g, &params, fanAngles, order);
}

bool tomographicModels::rebin_parallel(float* g, int order)
{
	rebin rebinningRoutines;
	return rebinningRoutines.rebin_parallel(g, &params, order);
}

bool tomographicModels::sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize)
{
	return sinogramReplacement(g, priorSinogram, metalTrace, &params, windowSize);
}

bool tomographicModels::down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (data_on_cpu)
		return downSample_cpu(I, N, I_dn, N_dn, factors);
	else
		return downSample(I, N, I_dn, N_dn, factors, params.whichGPU);
#else
	return downSample_cpu(I, N, I_dn, N_dn, factors);
#endif
}

bool tomographicModels::up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (data_on_cpu)
		return upSample_cpu(I, N, I_up, N_up, factors);
	else
		return upSample(I, N, I_up, N_up, factors, params.whichGPU);
#else
	return upSample_cpu(I, N, I_up, N_up, factors);
#endif
}

bool tomographicModels::scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType)
{
#ifndef __USE_CPU
	//*
	int numChunks = int(params.whichGPUs.size());
	if (params.numAngles >= numChunks)
	{
		int numViewsPerChunk = std::max(1, int(ceil(float(params.numAngles) / double(numChunks))));
		//printf("numChunks = %d\n", numChunks);

		omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
		#pragma omp parallel for schedule(dynamic)
		for (int ichunk = 0; ichunk < numChunks; ichunk++)
		{
			int firstView = ichunk * numViewsPerChunk;
			int lastView = std::min(firstView + numViewsPerChunk - 1, params.numAngles - 1);
			int numViews = lastView - firstView + 1;

			// make a copy of the relavent rows
			float* g_chunk = &g[uint64(firstView) * uint64(params.numRows * params.numCols)];

			// make a copy of the params
			parameters chunk_params;
			chunk_params = params;
			chunk_params.removeProjections(firstView, lastView);

			chunk_params.whichGPU = params.whichGPUs[omp_get_thread_num()];
			chunk_params.whichGPUs.clear();

			//printf("full numAngles = %d, chunk numAngles = %d\n", params.numAngles, chunk_params.numAngles);
			//printf("GPU %d: view range: (%d, %d)    slice range: (%d, %d)\n", chunk_params.whichGPU, firstView, lastView, sliceRange[0], sliceRange[1]);

			// Do Computation
			simulateScatter_firstOrder_singleMaterial(g_chunk, f, &chunk_params, source, energies, N_energies, detector, sigma, scatterDist, data_on_cpu, jobType);
		}
		return true;
	}
	else
		return simulateScatter_firstOrder_singleMaterial(g, f, &params, source, energies, N_energies, detector, sigma, scatterDist, data_on_cpu, jobType);
	//*/

	//return simulateScatter_firstOrder_singleMaterial(g, f, &params, source, energies, N_energies, detector, sigma, scatterDist, data_on_cpu, jobType);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::synthesize_symmetry(float* f_radial, float* f)
{
	phantom symObject(&params);
	return symObject.synthesizeSymmetry(f_radial, f);
}

float tomographicModels::find_centerCol(float* g, int iRow, float* searchBounds, bool data_on_cpu)
{
	if (data_on_cpu)
		return findCenter_cpu(g, &params, iRow, false, searchBounds);
	else
	{
		printf("Error: find_centerCol not yet implemented for data on the GPU\n");
		return 0.0;
	}
}

float tomographicModels::find_tau(float* g, int iRow, float* searchBounds, bool data_on_cpu)
{
	if (data_on_cpu)
		return findCenter_cpu(g, &params, iRow, true, searchBounds);
	else
	{
		printf("Error: find_tau not yet implemented for data on the GPU\n");
		return 0.0;
	}
}

float tomographicModels::estimate_tilt(float* g, bool data_on_cpu)
{
	if (data_on_cpu)
		return estimateTilt(g, &params);
	else
	{
		printf("Error: estimate_tilt not yet implemented for data on the GPU\n");
		return false;
	}
}

bool tomographicModels::conjugate_difference(float* g, float alpha, float centerCol, float* diff, bool data_on_cpu)
{
	if (data_on_cpu)
		return getConjugateDifference(g, &params, alpha, centerCol, diff);
	else
	{
		printf("Error: conjugate_difference not yet implemented for data on the GPU\n");
		return false;
	}
}

float tomographicModels::consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: consistency_cost not yet implemented for data on the CPU\n");
		return -1.0;
	}
	else
	{
		return consistencyCost(g, &params, data_on_cpu, Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt);
	}
#else
	printf("Error: consistency_cost not yet implemented for data on the CPU\n");
	return -1.0;
#endif
}

bool tomographicModels::Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu)
{
	if (g == NULL || params.geometryDefined() == false)
		return false;
	//BlurFilter2D(g, params.numAngles, params.numRows, params.numCols, 2.0, data_on_cpu);
#ifndef __USE_CPU
	if (data_on_cpu)
		return Laplacian_cpu(g, numDims, smooth, &params, 1.0);
	else
		return Laplacian_gpu(g, numDims, smooth, &params, data_on_cpu, 1.0);
#else
	return Laplacian_cpu(g, numDims, smooth, &params, 1.0);
#endif
}

bool tomographicModels::transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (g == NULL || H == NULL || N_H1 <= 0 || N_H2 <= 0 || params.geometryDefined() == false)
		return false;
	//return transmissionFilter_gpu(g, &params, data_on_cpu, H, N_H1, N_H2);
	
	//#####################################################################################################
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}

	int N_1 = params.numAngles;

	uint64 numElements = uint64(params.numAngles) * uint64(params.numRows) * uint64(params.numCols);
	//uint64 numElements_filter = uint64(N_H1) * uint64(N_H2);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (data_on_cpu == true && (getAvailableGPUmemory(params.whichGPU) < dataSize /*|| numElements > maxElements*/))
	{
		// do chunking
		int numSlices = std::min(N_1, maxSlicesForChunking);
		while (getAvailableGPUmemory(params.whichGPU) < double(numSlices) / double(N_1) * dataSize)
		{
			numSlices = numSlices / 2;
			if (numSlices < 1)
			{
				numSlices = 1;
				break;
			}
		}
		int numChunks = int(ceil(float(N_1) / float(numSlices)));

		//printf("number of slices per chunk: %d\n", numSlices);

		omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
		#pragma omp parallel for schedule(dynamic)
		for (int ichunk = 0; ichunk < numChunks; ichunk++)
		{
			int sliceStart = ichunk * numSlices;
			int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

			float* g_chunk = &g[uint64(sliceStart) * uint64(params.numRows * params.numCols)];
			int whichGPU = params.whichGPUs[omp_get_thread_num()];

			parameters params_chunk = params;
			params_chunk.numAngles = sliceEnd - sliceStart + 1;
			params_chunk.whichGPU = whichGPU;

			transmissionFilter_gpu(g_chunk, &params_chunk, data_on_cpu, H, N_H1, N_H2, isAttenuationData);
		}

		return true;
	}
	else
	{
		return transmissionFilter_gpu(g, &params, data_on_cpu, H, N_H1, N_H2, isAttenuationData);
	}
	//#####################################################################################################
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}

bool tomographicModels::AzimuthalBlur(float* f, float FWHM, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params.whichGPU < 0)
	{
		printf("Error: this function is currently only implemented for GPU processing!\n");
		return false;
	}
	float numVol = 1.0;
	if (data_on_cpu)
		numVol = 2.0;
	else
		numVol = 1.0;

	int N_1 = params.numZ;
	int N_2 = params.numY;
	int N_3 = params.numX;

	uint64 numElements = uint64(N_1) * uint64(N_2) * uint64(N_3);
	double dataSize = 4.0 * double(numElements) / pow(2.0, 30.0);
	//uint64 maxElements = 2147483646;

	if (getAvailableGPUmemory(params.whichGPU) < numVol * dataSize /*|| numElements > maxElements*/)
	{
		if (data_on_cpu == false)
		{
			printf("Error: Insufficient GPU memory for this operation!\n");
			return false;
		}
		else
		{
			// do chunking
			int numSlices = std::min(N_1, maxSlicesForChunking);
			while (getAvailableGPUmemory(params.whichGPU) < numVol * double(numSlices) / double(N_1) * dataSize)
			{
				numSlices = numSlices / 2;
				if (numSlices < 1)
				{
					numSlices = 1;
					break;
				}
			}
			int numChunks = int(ceil(float(N_1) / float(numSlices)));

			//printf("number of slices per chunk: %d\n", numSlices);

			omp_set_num_threads(std::min(int(params.whichGPUs.size()), omp_get_num_procs()));
			#pragma omp parallel for schedule(dynamic)
			for (int ichunk = 0; ichunk < numChunks; ichunk++)
			{
				int sliceStart = ichunk * numSlices;
				int sliceEnd = std::min(N_1 - 1, sliceStart + numSlices - 1);

				float* f_chunk = &f[uint64(sliceStart) * uint64(N_2 * N_3)];
				int whichGPU = params.whichGPUs[omp_get_thread_num()];
				parameters params_chunk = params;
				params_chunk.numZ = sliceEnd - sliceStart + 1;
				params_chunk.whichGPU = whichGPU;

				azimuthalBlur(f_chunk, &params_chunk, FWHM, data_on_cpu);
			}

			return true;
		}
	}
	else
		return azimuthalBlur(f, &params, FWHM, data_on_cpu);
#else
	printf("Error: GPU routines not included in this release!\n");
	return false;
#endif
}
