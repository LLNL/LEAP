////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "tomographic_models_c_interface.h"
#include "list_of_tomographic_models.h"
#include "tomographic_models.h"
#include "ray_tracing/phantom.h"
#include "fbp/ray_weighting_cpu.h"
#include "geometry/rebin.h"
#include "file_io.h"
#include "ring_removal.h"
#include "inpainting.h"
#include "segmentation.h"
#include "statistics.h"
#include "cpu_utils.h"
#include "resample_cpu.h"
#include "geometry/find_center_cpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <omp.h>

#ifndef __USE_CPU
#include "cuda_utils.h"
#endif

#if defined(_MSC_VER) || defined(_WIN32)
    #include <malloc.h>
#endif

//#include "Log.h"
//#include <torch/torch.h>
//#include <torch/extension.h>
//#include <pybind11/pybind11.h>
//namespace py = pybind11;

#ifndef PI
#define PI 3.141592653589793
#endif

//#ifdef DEFINE_STATIC_UI
listOfTomographicModels list_models;
int whichModel = 0;

bool set_model(int i)
{
	whichModel = i;
	return true;
}

int create_new_model()
{
	whichModel = list_models.append();
	return whichModel;
}

tomographicModels* tomo()
{
	return list_models.get(whichModel);
}

bool copy_parameters(int param_id, bool copy_volume_params)
{
	if (0 <= param_id && param_id < list_models.size())
	{
		if (whichModel != param_id)
		{
			//printf("copy %d => %d\n", param_id, whichModel);
			//list_models.get(param_id)->params.assign(tomo()->params);

			int volumeDimensionOrder;
			int numX, numY, numZ;
			float voxelWidth, voxelHeight;
			float offsetX, offsetY, offsetZ;

			parameters* params_source = &(list_models.get(param_id)->params);
			parameters* params_target = &(tomo()->params);

			if (!copy_volume_params)
			{
				volumeDimensionOrder = params_target->volumeDimensionOrder;
				numX = params_target->numX;
				numY = params_target->numY;
				numZ = params_target->numZ;
				voxelWidth = params_target->voxelWidth;
				voxelHeight = params_target->voxelHeight;
				offsetX = params_target->offsetX;
				offsetY = params_target->offsetY;
				offsetZ = params_target->offsetZ;
			}

			params_target->assign(*params_source);

			if (!copy_volume_params)
			{
				params_target->volumeDimensionOrder = volumeDimensionOrder;
				params_target->numX = numX;
				params_target->numY = numY;
				params_target->numZ = numZ;
				params_target->voxelWidth = voxelWidth;
				params_target->voxelHeight = voxelHeight;
				params_target->offsetX = offsetX;
				params_target->offsetY = offsetY;
				params_target->offsetZ = offsetZ;
			}

			phantom* phantom_source = &(list_models.get(param_id)->geometricPhantom);
			phantom* phantom_target = &(tomo()->geometricPhantom);
			phantom_target->assign(*phantom_source);
		}
		return true;
	}
	else
		return false;
}

bool copy_volume_parameters(int param_id)
{
	if (0 <= param_id && param_id < list_models.size())
	{
		if (whichModel != param_id)
		{
			parameters* params_source = &(list_models.get(param_id)->params);
			parameters* params_target = &(tomo()->params);

			params_target->volumeDimensionOrder = params_source->volumeDimensionOrder;
			params_target->numX = params_source->numX;
			params_target->numY = params_source->numY;
			params_target->numZ = params_source->numZ;
			params_target->voxelWidth = params_source->voxelWidth;
			params_target->voxelHeight = params_source->voxelHeight;
			params_target->offsetX = params_source->offsetX;
			params_target->offsetY = params_source->offsetY;
			params_target->offsetZ = params_source->offsetZ;
		}
		return true;
	}
	else
		return false;
}

float* allocate_3D_array(int N_1, int N_2, int N_3, bool pinned)
{
	float* data = NULL;
	if (N_1 > 0 && N_2 > 0 && N_3 > 0)
	{
		size_t size_bytes = size_t(N_1) * size_t(N_2) * size_t(N_3) * sizeof(float);
		#ifndef __USE_CPU
		if (pinned)
		{
			cudaError_t err = cudaHostAlloc((void**)&data, size_bytes, cudaHostAllocDefault);
			if (err != cudaSuccess)
			{
				std::fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
				return nullptr;
			}
		}
		else
		{
			data = malloc_aligned(size_bytes);
		}
		#else
		data = malloc_aligned(size_bytes);
		#endif
	}
	return data;
}

bool free_3D_array(float* data, bool pinned)
{
	if (data == nullptr)
		return false;
	#ifndef __USE_CPU
	if (pinned)
	{
		cudaFreeHost(data);
		return true;
	}
	else
		return free_aligned(data);
	#else
	return free_aligned(data);
	#endif
}

void about()
{
	tomo()->about();
}

void version(char* versionText)
{
	sprintf(versionText, "%s", LEAP_VERSION);
}

bool print_parameters()
{
	return tomo()->print_parameters();
}

bool reset()
{
	return tomo()->reset();
}

bool all_defined()
{
	return tomo()->params.allDefined(false);
}

bool ct_geometry_defined()
{
	return tomo()->params.geometryDefined(false);
}

bool ct_volume_defined()
{
	return tomo()->params.volumeDefined(false);
}

void set_log_error()
{
	tomo()->set_log_error();
}

void set_log_warning()
{
	tomo()->set_log_warning();
}

void set_log_status()
{
	tomo()->set_log_status();
}

void set_log_debug()
{
	tomo()->set_log_debug();
}

bool include_cufft()
{
	#ifdef __INCLUDE_CUFFT
	return true;
	#else
	return false;
	#endif
}

int getOptimalFFTsize(int N)
{
	return optimalFFTsize(N);
}

bool set_maxSlicesForChunking(int N)
{
	return tomo()->set_maxSlicesForChunking(N);
}

int get_maxSlicesForChunking()
{
	return tomo()->get_maxSlicesForChunking();
}

bool verify_input_sizes(int numAngles, int numRows, int numCols, int numZ, int numY, int numX)
{
	parameters* params = &(tomo()->params);
	if (params->numAngles != numAngles || params->numRows != numRows || params->numCols != numCols || params->numZ != numZ || params->numY != numY || params->numX != numX)
		return false;
	else
		return true;
}

bool project_gpu(float* g, float* f)
{
	return tomo()->project_gpu(g, f);
}

bool project_with_mask_gpu(float* g, float* f, float* mask)
{
	return tomo()->project_with_mask_gpu(g, f, mask);
}

bool backproject_gpu(float* g, float* f)
{
	return tomo()->backproject_gpu(g, f);
}

bool project_cpu(float* g, float* f)
{
	return tomo()->project_cpu(g, f);
}

bool project_with_mask_cpu(float* g, float* f, float* mask)
{
	return tomo()->project_with_mask_cpu(g, f, mask);
}

bool backproject_cpu(float* g, float* f)
{
	return tomo()->backproject_cpu(g, f);
}

bool project(float* g, float* f, bool data_on_cpu)
{
	return tomo()->project(g, f, data_on_cpu);
}

bool project_with_mask(float* g, float* f, float* mask, bool data_on_cpu)
{
	return tomo()->project_with_mask(g, f, mask, data_on_cpu);
}

bool backproject(float* g, float* f, bool data_on_cpu)
{
	return tomo()->backproject(g, f, data_on_cpu);
}

bool FBP_cpu(float* g, float* f)
{
	return tomo()->FBP_cpu(g, f);
}

bool FBP_gpu(float* g, float* f)
{
	return tomo()->FBP_gpu(g, f);
}

bool weightedBackproject(float* g, float* f, bool doDBP, bool data_on_cpu)
{
	return tomo()->weightedBackproject(g, f, doDBP, data_on_cpu);
}

bool fmad(float* g, int N_1, int N_2, int N_3, float* scale, float* shift, int M_1, int M_2, int M_3, float clip_low, float clip_high)
{
	if (g == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || scale == NULL || shift == NULL || M_1 <= 0 || M_2 <= 0 || M_3 <= 0)
		return false;
	else
	{
		bool just_clip = false;
		if (M_1 == 1 && M_2 == 1 && M_3 == 1 && scale[0] == 1.0 && shift[0] == 0.0 && clip_low == 0.0 && isnan(clip_high))
			just_clip = true;

		if (isnan(clip_low))
			clip_low = -1.0*pow(2.0, 32);
		if (isnan(clip_high))
			clip_high = pow(2.0, 32);
		omp_set_num_threads(num_cpu_threads());
		#pragma omp parallel for
		for (int i = 0; i < N_1; i++)
		{
			float* aProj = &g[uint64(i)*uint64(N_2*N_3)];
			float* scale_slice = NULL;
			float* shift_slice = NULL;
			if (M_1 == 1)
			{
				scale_slice = scale;
				shift_slice = shift;
			}
			else
			{
				scale_slice = &scale[uint64(i)*uint64(M_2*M_3)];
				shift_slice = &shift[uint64(i)*uint64(M_2*M_3)];
			}
			for (int j = 0; j < N_2; j++)
			{
				float* scale_line = NULL;
				float* shift_line = NULL;
				if (M_2 == 1)
				{
					scale_line = scale_slice;
					shift_line = shift_slice;
				}
				else
				{
					scale_line = &scale_slice[j*M_3];
					shift_line = &shift_slice[j*M_3];
				}
				if (just_clip)
				{
					for (int k = 0; k < N_3; k++)
					{
						if (aProj[j*N_3 + k] < 0.0)
							aProj[j*N_3 + k] = 0.0;
					}
				}
				else
				{
					for (int k = 0; k < N_3; k++)
					{
						float m = scale_line[min(k, M_3-1)];
						float b = shift_line[min(k, M_3-1)];
						int ind = j*N_3 + k;
						aProj[ind] = std::min(clip_high, std::max(clip_low, aProj[ind]*m + b));
					}
				}
			}
		}
		return true;
	}
}

bool multiply(float* out, float* y, float* x, int N_1, int N_2, int N_3)
{
	if (out == nullptr || y == nullptr || x == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	uint64 img_sz = uint64(N_2*N_3);
	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* out_i = &out[uint64(i)*img_sz];
		float* y_i = &y[uint64(i)*img_sz];
		float* x_i = &x[uint64(i)*img_sz];
		for (int j = 0; j < N_2; j++)
		{
			for (int k = 0; k < N_3; k++)
				out_i[j*N_3+k] = y_i[j*N_3+k] * x_i[j*N_3+k];
		}
	}
	return true;
}

bool scalar_add(float* out, float* y, float a, float* x, int N_1, int N_2, int N_3, bool do_clip)
{
	if (out == nullptr || y == nullptr || x == nullptr || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	uint64 img_sz = uint64(N_2*N_3);
	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* out_i = &out[uint64(i)*img_sz];
		float* y_i = &y[uint64(i)*img_sz];
		float* x_i = &x[uint64(i)*img_sz];
		for (int j = 0; j < N_2; j++)
		{
			for (int k = 0; k < N_3; k++)
			{
				float val = y_i[j*N_3+k] + a * x_i[j*N_3+k];
				if (do_clip)
					out_i[j*N_3+k] = max(float(0.0), val);
				else
					out_i[j*N_3+k] = val;
			}
		}
	}
	return true;
}

bool negLog(float* g, int N_1, int N_2, int N_3, float gray_value, float clip_low, float clip_high)
{
	if (g == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	else
	{
		if (isnan(clip_low))
			clip_low = pow(2.0,-16)*gray_value;
		if (isnan(clip_high))
			clip_high = pow(2.0, 16)*gray_value;
		omp_set_num_threads(num_cpu_threads());
		#pragma omp parallel for
		for (int i = 0; i < N_1; i++)
		{
			float* aProj = &g[uint64(i)*uint64(N_2*N_3)];
			for (int j = 0; j < N_2; j++)
			{
				for (int k = 0; k < N_3; k++)
				{
					aProj[j*N_3+k] = -log(std::min(clip_high, std::max(clip_low, aProj[j*N_3+k]))/gray_value);
				}
			}
		}
		return true;
	}
}

bool expNeg(float* g, int N_1, int N_2, int N_3, float gray_value)
{
	if (g == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	else
	{
		omp_set_num_threads(num_cpu_threads());
		#pragma omp parallel for
		for (int i = 0; i < N_1; i++)
		{
			float* aProj = &g[uint64(i)*uint64(N_2*N_3)];
			for (int j = 0; j < N_2; j++)
			{
				for (int k = 0; k < N_3; k++)
					aProj[j*N_3+k] = exp(-aProj[j*N_3+k])*gray_value;
			}
		}
		return true;
	}
}

bool DBP_filter(float* g, bool data_on_cpu)
{
	return tomo()->DBPfilter(g, data_on_cpu);
}

bool DBP_filter_cpu(float* g)
{
	return tomo()->DBPfilter_cpu(g);
}

bool filterProjections(float* g, float* g_out, bool inconsistency, bool data_on_cpu)
{
	tomo()->params.inconsistencyReconstruction = inconsistency;
	bool retVal = tomo()->filterProjections(g, g_out, data_on_cpu);
	tomo()->params.inconsistencyReconstruction = false;
	return retVal;
}

bool filterProjections_gpu(float* g, bool inconsistency)
{
	tomo()->params.inconsistencyReconstruction = inconsistency;
	bool retVal = tomo()->filterProjections_gpu(g);
	tomo()->params.inconsistencyReconstruction = false;
	return retVal;
}

bool filterProjections_cpu(float* g, float* g_out, bool inconsistency)
{
	tomo()->params.inconsistencyReconstruction = inconsistency;
	bool retVal = tomo()->filterProjections_cpu(g, g_out);
	tomo()->params.inconsistencyReconstruction = false;
	return retVal;
}

int extraColumnsForOffsetScan()
{
	return tomo()->extraColumnsForOffsetScan();
}

bool get_offsetScan_weights(float* w)
{
	if (w == NULL)
		return false;
	float* w_temp = setOffsetScanWeights(&(tomo()->params));
	if (w_temp != NULL)
	{
		memcpy(w, w_temp, sizeof(float) * tomo()->params.numRows * tomo()->params.numCols);
		free(w_temp);
		return true;
	}
	else
		return false;
}

bool apply_projection_weights(float* g, float* w, int expNeg_or_negLog, bool data_on_cpu)
{
	if (g == NULL || w == NULL || data_on_cpu == false)
		return false;

	int numAngles = get_numAngles();
	int numRows = get_numRows();
	int numCols = get_numCols();

	float lowerBound = pow(2.0,-16);

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < numAngles; i++)
	{
		float* aProj = &g[uint64(i)*uint64(numRows)*uint64(numCols)];
		for (int j = 0; j < numRows; j++)
		{
			for (int k = 0; k < numCols; k++)
			{
				if (expNeg_or_negLog == -1)
				{
					aProj[j*numCols + k] = exp(-aProj[j*numCols + k]) * w[j*numCols + k];
				}
				else if (expNeg_or_negLog == 1)
				{
					aProj[j*numCols + k] = -log(std::max(lowerBound, aProj[j*numCols + k] * w[j*numCols + k]));
				}
				else
					aProj[j*numCols + k] *= w[j*numCols + k];
			}
		}
	}
	return true;
}

bool HilbertFilterProjections(float* g, bool data_on_cpu, float scalar, float sampleShift)
{
	return tomo()->HilbertFilterProjections(g, data_on_cpu, scalar, sampleShift);
}

bool rampFilterProjections(float* g, bool data_on_cpu, float scalar)
{
	return tomo()->rampFilterProjections(g, data_on_cpu, scalar);
}

bool preRampFiltering(float* g, bool data_on_cpu)
{
	return tomo()->preRampFiltering(g, data_on_cpu);
}

bool postRampFiltering(float* g, bool data_on_cpu)
{
	return tomo()->postRampFiltering(g, data_on_cpu);
}

bool rampFilterVolume(float* f, bool data_on_cpu)
{
	return tomo()->rampFilterVolume(f, data_on_cpu);
}

bool FBP(float* g, float* f, bool data_on_cpu)
{
	return tomo()->doFBP(g, f, data_on_cpu);
}

bool DBP(float* g, float* f, bool data_on_cpu)
{
	return tomo()->DBP(g, f, data_on_cpu);
}

bool inconsistencyReconstruction(float* g, float* f, bool data_on_cpu)
{
	tomo()->params.inconsistencyReconstruction = true;
	bool retVal = FBP(g, f, data_on_cpu);
	tomo()->params.inconsistencyReconstruction = false;
	return retVal;
}

bool lambdaTomography(float* g, float* f, bool data_on_cpu)
{
	bool offsetScan_save = tomo()->params.offsetScan;
	tomo()->params.offsetScan = false;
	tomo()->params.lambdaTomography = true;
	bool retVal = FBP(g, f, data_on_cpu);
	tomo()->params.lambdaTomography = false;
	tomo()->params.offsetScan = offsetScan_save;
	return retVal;
}

bool sensitivity(float* f, bool data_on_cpu)
{
	return tomo()->sensitivity(f, data_on_cpu);
}

bool windowFOV(float* f, bool data_on_cpu)
{
	return tomo()->windowFOV(f, data_on_cpu);
}

float get_FBPscalar()
{
	return tomo()->get_FBPscalar();
}

bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float tiltAngle, float pitchAngle, float helicalPitch)
{
	return tomo()->set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, tiltAngle, pitchAngle, helicalPitch);
}

bool set_fanbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau)
{
	return tomo()->set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau);
}

bool set_parallelbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	return tomo()->set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
}

bool set_modularbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	return tomo()->set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in);
}

bool set_coneparallel(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float helicalPitch)
{
	return tomo()->set_coneparallel(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch);
}

bool rotate_detector(float alpha)
{
	return tomo()->params.rotateDetector(alpha);
}

bool shift_detector(float r, float c)
{
	return tomo()->params.shiftDetector(r, c);
}

bool set_flatDetector()
{
	return tomo()->set_flatDetector();
}

bool set_curvedDetector()
{
	return tomo()->set_curvedDetector();
}

int get_detectorType()
{
	return tomo()->params.detectorType;
}

bool set_numCols(int numCols)
{
	if (numCols >= 0)
	{
		tomo()->params.numCols = numCols;
		return true;
	}
	else
		return false;
}

bool set_numRows(int numRows)
{
	if (numRows >= 0)
	{
		tomo()->params.numRows = numRows;
		return true;
	}
	else
		return false;
}

bool set_numAngles(int numAngles)
{
	if (numAngles >= 0)
	{
		if (tomo()->params.numAngles != numAngles)
		{
			if (tomo()->params.phis != NULL)
				delete[] tomo()->params.phis;
			tomo()->params.phis = NULL;
			tomo()->params.numAngles = numAngles;
		}
		return true;
	}
	else
		return false;
}

bool set_pixelHeight(float H)
{
	if (H >= 0.0)
	{
		tomo()->params.pixelHeight = H;
		return true;
	}
	else
		return false;
}

bool set_pixelWidth(float W)
{
	if (W >= 0.0)
	{
		tomo()->params.pixelWidth = W;
		return true;
	}
	else
		return false;
}

bool set_centerCol(float centerCol)
{
	return tomo()->set_centerCol(centerCol);
}

bool set_centerRow(float centerRow)
{
	return tomo()->set_centerRow(centerRow);
}

bool set_sod(float sod)
{
	return tomo()->params.set_sod(sod);
}
bool set_sdd(float sdd)
{
	return tomo()->params.set_sdd(sdd);
}

bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool set_default_volume(float scale)
{
	return tomo()->set_default_volume(scale);
}

bool set_numZ(int numZ)
{
	if (numZ >= 0)
	{
		tomo()->params.numZ = numZ;
		return true;
	}
	else
		return false;
}

bool set_numY(int numY)
{
	if (numY >= 0)
	{
		tomo()->params.numY = numY;
		return true;
	}
	else
		return false;
}

bool set_numX(int numX)
{
	if (numX >= 0)
	{
		tomo()->params.numX = numX;
		return true;
	}
	else
		return false;
}

bool set_offsetX(float offsetX)
{
	tomo()->params.offsetX = offsetX;
	return true;
}

bool set_offsetY(float offsetY)
{
	tomo()->params.offsetY = offsetY;
	return true;
}

bool set_offsetZ(float offsetZ)
{
	tomo()->params.offsetZ = offsetZ;
	return true;
}

bool set_voxelWidth(float W)
{
	if (W >= 0.0)
	{
		tomo()->params.voxelWidth = W;
		return true;
	}
	else
		return false;
}

bool set_voxelHeight(float H)
{
	if (H >= 0.0 /*&& (tomo()->params.geometry == parameters::CONE || tomo()->params.geometry == parameters::MODULAR)*/)
	{
		tomo()->params.voxelHeight = H;
		return true;
	}
	else
		return false;
}

float default_voxelWidth()
{
	return tomo()->params.default_voxelWidth();
}

bool set_volumeDimensionOrder(int which)
{
	return tomo()->set_volumeDimensionOrder(which);
}

int get_volumeDimensionOrder()
{
	return tomo()->get_volumeDimensionOrder();
}

bool set_max_cpu_threads(int n)
{
	max_threads = max(1, n);
	if (max_threads < number_of_gpus())
		printf("WARNING: number of usable GPUs is limited by the maximum number of CPU threads\n");
	return true;
}

bool set_max_gpu_memory(float c)
{
	#ifndef __USE_CPU
	if (c < 0.1)
	{
		max_gpu_memory = 0.1;
		return false;
	}
	else
	{
		max_gpu_memory = c;
		return true;
	}
	#else
	return false;
	#endif
}

bool get_physically_shared_memory()
{
	#ifndef __USE_CPU
	return physically_shared_memory(0);
	#else
	return false;
	#endif
}

float get_available_system_memory()
{
	return getAvailableSystemMemory();
}

int number_of_gpus()
{
	return tomo()->number_of_gpus();
}

int get_gpus(int* list_of_gpus)
{
	return tomo()->get_gpus(list_of_gpus);
}

bool set_GPU(int whichGPU)
{
	return tomo()->set_GPU(whichGPU);
}

bool set_GPUs(int* whichGPUs, int N)
{
	return tomo()->set_GPUs(whichGPUs, N);
}

int get_GPU()
{
	return tomo()->get_GPU();
}

float get_available_gpu_memory(int whichGPU)
{
	#ifdef __USE_CPU
	return 0.0;
	#else
	return getAvailableGPUmemory(whichGPU);
	#endif
}

bool set_projector(int which)
{
	return tomo()->set_projector(which);
}

int get_projector()
{
	return tomo()->params.whichProjector;
}

bool set_axisOfSymmetry(float axisOfSymmetry)
{
	return tomo()->set_axisOfSymmetry(axisOfSymmetry);
}

float get_axisOfSymmetry()
{
	return tomo()->params.axisOfSymmetry;
}

bool clear_axisOfSymmetry()
{
	return tomo()->clear_axisOfSymmetry();
}

bool set_rFOV(float rFOV_in)
{
	return tomo()->set_rFOV(rFOV_in);
}

float get_rFOV(bool get_default_value)
{
	if (get_default_value)
	{
		float rFOVspecified_save = tomo()->params.rFOVspecified;
		tomo()->params.rFOVspecified = 0.0;
		float retVal = tomo()->params.rFOV();
		tomo()->params.rFOVspecified = rFOVspecified_save;
		return retVal;
	}
	else
		return tomo()->params.rFOV();
}

float get_rFOV_min()
{
	return tomo()->params.rFOV_min();
}

float get_rFOV_max()
{
	return tomo()->params.rFOV_max();
}

float get_zFOV_min()
{
	return tomo()->params.zFOV_min();
}

float get_zFOV_max()
{
	return tomo()->params.zFOV_max();
}

bool set_offsetScan(bool aFlag)
{
	return tomo()->params.set_offsetScan(aFlag);
}

bool get_offsetScan()
{
	return tomo()->params.offsetScan;
}

bool set_truncatedScan(bool aFlag)
{
	return tomo()->params.set_truncatedScan(aFlag);
}

bool get_truncatedScan()
{
	return tomo()->params.truncatedScan;
}

bool set_cornerPatching(bool aFlag)
{
	return tomo()->params.set_cornerPatching(aFlag);
}

bool set_clipWeightedBackprojection(bool aFlag)
{
	tomo()->params.clipWeightedBackprojection = aFlag;
	return true;
}

bool get_clipWeightedBackprojection()
{
	return tomo()->params.clipWeightedBackprojection;
}

bool set_numRowsExtrapolate(int N)
{
	return tomo()->params.set_numRowsExtrapolate(N);
}

bool set_numTVneighbors(int N)
{
	return tomo()->params.set_numTVneighbors(N);
}

int get_numTVneighbors()
{
	return tomo()->params.numTVneighbors;
}

bool set_rampID(int whichRampFilter)
{
	return tomo()->set_rampID(whichRampFilter);
}

int get_rampID()
{
	return tomo()->params.rampID;
}

bool set_helicalFilterParameter(float epsilon)
{
	tomo()->params.helicalFilterParameter = std::max(float(0.0), epsilon);
	return true;
}

bool set_FBPlowpass(float W)
{
	tomo()->params.FBPlowpass = W;
	return true;
}

float get_FBPlowpass()
{
	return tomo()->params.FBPlowpass;
}

bool set_tau(float tau)
{
	return tomo()->set_tau(tau);
}

bool set_tiltAngle(float tiltAngle)
{
	return tomo()->set_tiltAngle(tiltAngle);
}

bool set_pitchAngle(float pitchAngle)
{
	return tomo()->set_pitchAngle(pitchAngle);
}

bool set_helicalPitch(float h)
{
	return tomo()->set_helicalPitch(h);
}

bool set_normalizedHelicalPitch(float h_normalized)
{
	return tomo()->set_normalizedHelicalPitch(h_normalized);
}

bool set_source_size(float height, float width)
{
	return tomo()->params.set_source_size(height, width);
}

bool get_source_size(float* height_and_width)
{
	if (height_and_width == NULL)
		return false;
	else
	{
		height_and_width[0] = tomo()->params.source_size[0];
		height_and_width[1] = tomo()->params.source_size[1];
		return true;
	}
}

bool set_helicalFBPWeight(float q)
{
	if (q <= 0.0 || q > 1.0)
		return false;
	else
	{
		tomo()->params.helicalFBPWeight = q;
		return true;
	}
}

bool set_DBPparameter(float epsilon)
{
	tomo()->params.DBPparameter = epsilon;
	return true;
}

bool set_attenuationMap(float* mu)
{
	return tomo()->set_attenuationMap(mu);
}

bool set_cylindircalAttenuationMap(float c, float R)
{
	return tomo()->set_attenuationMap(c, R);
}

bool convert_conebeam_to_modularbeam()
{
	return tomo()->params.convert_conebeam_to_modularbeam();
}

bool convert_parallelbeam_to_modularbeam()
{
	return tomo()->params.convert_parallelbeam_to_modularbeam();
}

bool clear_attenuationMap()
{
	return tomo()->clear_attenuationMap();
}

bool muSpecified()
{
	return tomo()->params.muSpecified();
}

bool flipAttenuationMapSign(bool data_on_cpu)
{
	return tomo()->flipAttenuationMapSign(data_on_cpu);
}

bool angles_are_defined()
{
	return tomo()->params.angles_are_defined();
}

bool angles_are_equispaced()
{
	return tomo()->params.anglesAreEquispaced();
}

bool set_geometry(int which)
{
	//CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3
	if (which < parameters::CONE || which > parameters::CONE_PARALLEL)
		return false;
	else
	{
		tomo()->params.geometry = which;
		if (which != parameters::CONE)
			tomo()->params.detectorType = parameters::FLAT;
		return true;
	}
}

bool projectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->projectConeBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool backprojectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->backprojectConeBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool projectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->projectFanBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool backprojectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->backprojectFanBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool projectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->projectParallelBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool backprojectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->backprojectParallelBeam(g, f, data_on_cpu, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool rowRangeNeededForBackprojection(int* rowsNeeded)
{
	if (rowsNeeded == NULL || tomo()->params.allDefined() == false)
		return false;
	else
		return tomo()->params.rowRangeNeededForBackprojection(0, tomo()->params.numZ - 1, rowsNeeded);
}

bool viewRangeNeededForBackprojection(int* viewsNeeded)
{
	if (viewsNeeded == NULL || tomo()->params.allDefined() == false)
		return false;
	else
		return tomo()->params.viewRangeNeededForBackprojection(0, tomo()->params.numZ - 1, viewsNeeded);
}

bool sliceRangeNeededForProjection(int* slicesNeeded, bool doClip, bool restrictToVolume)
{
	if (slicesNeeded == NULL || tomo()->params.allDefined() == false)
		return false;
	else
	{
		int rowRange[2] = {0, tomo()->params.numRows - 1};
		if (restrictToVolume)
			rowRangeNeededForBackprojection(rowRange);
		return tomo()->params.sliceRangeNeededForProjection(rowRange[0], rowRange[1], slicesNeeded, doClip);
	}
}

int numRowsRequiredForBackprojectingSlab(int numSlicesPerChunk)
{
	return tomo()->numRowsRequiredForBackprojectingSlab(numSlicesPerChunk);
}

int get_geometry()
{
	return tomo()->params.geometry;
}

float get_sod()
{
	return tomo()->params.sod;
}

float get_sdd()
{
	return tomo()->params.sdd;
}

int get_numAngles()
{
	return tomo()->get_numAngles();
}

int get_numRows()
{
	return tomo()->get_numRows();
}

int get_numCols()
{
	return tomo()->get_numCols();
}

float get_pixelWidth()
{
	return tomo()->get_pixelWidth();
}

float get_pixelHeight()
{
	return tomo()->get_pixelHeight();
}

float get_centerRow()
{
	return tomo()->params.centerRow;
}

float get_centerCol()
{
	return tomo()->params.centerCol;
}

float get_tau()
{
	return tomo()->params.tau;
}

float get_tiltAngle()
{
	return tomo()->params.tiltAngle;
}

float get_pitchAngle()
{
	return tomo()->params.pitchAngle;
}

float get_helicalPitch()
{
	return tomo()->get_helicalPitch();
}

float get_normalizedHelicalPitch()
{
	return tomo()->params.normalizedHelicalPitch();
}

float get_helicalFBPWeight()
{
	return tomo()->params.helicalFBPWeight;
}

float get_z_source_offset()
{
	return tomo()->get_z_source_offset();
}

bool set_z_source_offset(float z_offs)
{
	return tomo()->set_z_source_offset(z_offs);
}

bool get_sourcePositions(float* x)
{
	return tomo()->get_sourcePositions(x);
}

bool get_moduleCenters(float* x)
{
	return tomo()->get_moduleCenters(x);
}

bool get_rowVectors(float* x)
{
	return tomo()->get_rowVectors(x);
}

bool get_colVectors(float* x)
{
	return tomo()->get_colVectors(x);
}

bool set_angles(float* phis, int N)
{
	return tomo()->params.set_angles(phis, N);
}

bool get_angles(float* phis)
{
	return tomo()->params.get_angles(phis);
}

float get_angularRange(bool get_sign)
{
	if (get_sign)
	{
		if (tomo()->params.T_phi() < 0.0)
			return -1.0*tomo()->params.angularRange;
		else
			return tomo()->params.angularRange;
	}
	else
		return tomo()->params.angularRange;
}

int get_numX()
{
	return tomo()->get_numX();
}

int get_numY()
{
	return tomo()->get_numY();
}

int get_numZ()
{
	return tomo()->get_numZ();
}

float get_voxelWidth()
{
	return tomo()->get_voxelWidth();
}

float get_voxelHeight()
{
	return tomo()->get_voxelHeight();
}

float get_offsetX()
{
	return tomo()->params.offsetX;
}

float get_offsetY()
{
	return tomo()->params.offsetY;
}

float get_offsetZ()
{
	return tomo()->params.offsetZ;
}

float get_z0()
{
	return tomo()->params.z_0();
}

float get_y0()
{
	return tomo()->params.y_0();
}

float get_x0()
{
	return tomo()->params.x_0();
}

float find_centerCol(float* g, int iRow, float* searchBounds, bool data_on_cpu)
{
	return tomo()->find_centerCol(g, iRow, searchBounds, data_on_cpu);
}

float find_tau(float* g, int iRow, float* searchBounds, bool data_on_cpu)
{
	return tomo()->find_tau(g, iRow, searchBounds, data_on_cpu);
}

float consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu)
{
	return tomo()->consistency_cost(g, Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt, data_on_cpu);
}

float estimate_tilt(float* g, bool data_on_cpu)
{
	return tomo()->estimate_tilt(g, data_on_cpu);
}

float conjugate_difference(float* g, float alpha, float centerCol, float* diff, bool data_on_cpu)
{
	return tomo()->conjugate_difference(g, alpha, centerCol, diff, data_on_cpu);
}

bool inconsistency_sweep(float* g, float* shifts, int numShifts, float* tilts, int numTilts, int which_param, float* costValues, bool data_on_cpu)
{
	return tomo()->inconsistency_sweep(g, shifts, numShifts, tilts, numTilts, which_param, costValues, data_on_cpu);
}

bool Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu)
{
	return tomo()->Laplacian(g, numDims, smooth, data_on_cpu);
}

bool ring_removal(float* g, float delta, float beta, int numIter, float maxChange, int angle_downsampling_factor)
{
	parameters* params = &(tomo()->params);
	ringRemoval ringo;
	return ringo.execute(g, params->numAngles, params->numRows, params->numCols, delta, beta, numIter, maxChange, angle_downsampling_factor);
}

bool transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, float FWHM, bool data_on_cpu)
{
	return tomo()->transmissionFilter(g, H, N_H1, N_H2, isAttenuationData, FWHM, data_on_cpu);
}

bool apply_polynomial_bhc(float* g, int N_1, int N_2, int N_3, float* coeff, int N_coeff, bool data_on_cpu)
{
	if (g == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || coeff == NULL || N_coeff <= 0)
		return false;
	if (data_on_cpu == false)
		return false;

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N_1; i++)
	{
		float* aProj = &g[uint64(i) * uint64(N_2*N_3)];
		for (int j = 0; j < N_2; j++)
		{
			float* aLine = &aProj[uint64(j)*uint64(N_3)];
			for (int k = 0; k < N_3; k++)
			{
				float cur_val = aLine[k];
				float x = cur_val;
				float new_val = coeff[0];
				for (int l = 1; l < N_coeff; l++)
				{
					new_val += coeff[l] * x;
					x *= cur_val;
				}
				aLine[k] = new_val;
			}
		}
	}
	return true;
}

bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	return tomo()->applyTransferFunction(x, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, data_on_cpu);
}

bool beam_hardening_heel_effect(float* g, float* anode_normal, float* LUT, float* takeOffAngles, int numSamples, int numAngles, float sampleRate, float firstSample, bool data_on_cpu)
{
	return tomo()->beam_hardening_heel_effect(g, anode_normal, LUT, takeOffAngles, numSamples, numAngles, sampleRate, firstSample, data_on_cpu);
}

bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool scalar_LUT, bool data_on_cpu)
{
	return tomo()->applyDualTransferFunction(x, y, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, scalar_LUT, data_on_cpu);
}

bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool constrain, bool data_on_cpu)
{
	return tomo()->convertToRhoeZe(f_L, f_H, N_1, N_2, N_3, sigma_L, sigma_H, constrain, data_on_cpu);
}

bool applyThreeMaterialBHC(float* sum, float* w_1, float* w_2, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	return tomo()->applyThreeMaterialBHC(sum, w_1, w_2, LUT, firstSample, sampleRate, numSamples, data_on_cpu);
}

bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->BlurFilter(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool HighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->HighPassFilter(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu)
{
	return tomo()->MedianFilter(f, N_1, N_2, N_3, threshold, w, signalThreshold, data_on_cpu);
}

bool MeanOrVarianceFilter(float* f, int N_1, int N_2, int N_3, int r, int order, bool data_on_cpu)
{
	return tomo()->MeanOrVarianceFilter(f, N_1, N_2, N_3, r, order, data_on_cpu);
}

bool BlurFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->BlurFilter2D(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool HighPassFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->HighPassFilter2D(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool MedianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, float signalThreshold, bool data_on_cpu)
{
	return tomo()->MedianFilter2D(f, N_1, N_2, N_3, threshold, w, signalThreshold, data_on_cpu);
}

bool BlurFilter1D(float* f, int N_1, int N_2, int N_3, float FWHM, int axis, bool isPeriodic, bool data_on_cpu)
{
	return tomo()->BlurFilter1D(f, N_1, N_2, N_3, FWHM, axis, isPeriodic, data_on_cpu);
}

bool badPixelCorrection(float* g, int N_1, int N_2, int N_3, float* badPixelMap, int w, bool data_on_cpu)
{
	return tomo()->badPixelCorrection(g, N_1, N_2, N_3, badPixelMap, w, data_on_cpu);
}

bool BilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu)
{
	return tomo()->BilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, scale, data_on_cpu);
}

bool PriorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* prior, bool data_on_cpu)
{
	return tomo()->PriorBilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, prior, data_on_cpu);
}

bool GuidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu)
{
	return tomo()->GuidedFilter(f, N_1, N_2, N_3, r, epsilon, numIter, data_on_cpu);
}

bool dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu)
{
	return tomo()->dictionaryDenoising(f, N_1, N_2, N_3, dictionary, numElements, N_d1, N_d2, N_d3, epsilon, sparsityThreshold, data_on_cpu);
}

float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
	return tomo()->TVcost(f, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
}

bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
	return tomo()->TVgradient(f, Df, N_1, N_2, N_3, delta, beta, p, false, data_on_cpu);
}

float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
	return tomo()->TVquadForm(f, d, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
}

bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu)
{
	return tomo()->Diffuse(f, N_1, N_2, N_3, delta, p, numIter, data_on_cpu);
}

bool TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu)
{
	return tomo()->TV_denoise(f, N_1, N_2, N_3, delta, beta, p, numIter, doMean, data_on_cpu);
}

bool TV_fast(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool data_on_cpu)
{
	return tomo()->TV_fast(f, N_1, N_2, N_3, delta, beta, p, numIter, data_on_cpu);
}

bool addMesh(float* triangles, int numTriangles, float val, const char* chemForm)
{
	return tomo()->geometricPhantom.addMesh(triangles, numTriangles, val, chemForm);
}

bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip, const char* chemForm, int oversampling)
{
	return tomo()->geometricPhantom.addObject(f, &(tomo()->params), type, c, r, val, A, clip, chemForm, oversampling);
}

bool voxelize(float* f, int oversampling)
{
	return tomo()->geometricPhantom.voxelize(f, &(tomo()->params), oversampling);
}

bool voxelizeMesh(float* f, float val, int oversampling)
{
	return tomo()->voxelizeMesh(f, val, true, oversampling);
}

bool scalePhantom(float scale_x, float scale_y, float scale_z)
{
	return tomo()->geometricPhantom.scale_phantom(scale_x, scale_y, scale_z);
}

bool shiftPhantom(float shift_x, float shift_y, float shift_z)
{
	return tomo()->geometricPhantom.shift_phantom(shift_x, shift_y, shift_z);
}

bool clearPhantom()
{
	tomo()->geometricPhantom.clearAll();
	return true;
}

bool rayTrace(float* g, int oversampling, bool data_on_cpu)
{
	return tomo()->rayTrace(g, NULL, NULL, 0, oversampling, data_on_cpu);
}

bool rayTrace_polychromatic(float* g, float* spectralResponse, float* energies, int N_energies, int oversampling, bool data_on_cpu)
{
	return tomo()->rayTrace(g, spectralResponse, energies, N_energies, oversampling, data_on_cpu);
}

bool rayTraceMesh(float* g, int oversampling, bool data_on_cpu)
{
	return tomo()->rayTraceMesh(g, NULL, NULL, 0, oversampling, data_on_cpu);
}

bool rayTraceMesh_polychromatic(float* g, float* spectralResponse, float* energies, int N_energies, int oversampling, bool data_on_cpu)
{
	return tomo()->rayTraceMesh(g, spectralResponse, energies, N_energies, oversampling, data_on_cpu);
}

bool double_cone(float* f, int N_1, int N_2, int N_3, float beta, float minimum_radius)
{
	return tomo()->geometricPhantom.double_cone(f, N_1, N_2, N_3, beta, minimum_radius);
}

bool patch_corners(float* f, float* f_top, int numZ_top, float* f_bot, int numZ_bot, int window_width)
{
	if (f == NULL || f_top == NULL || numZ_top <= 0 || f_bot == NULL || numZ_bot <= 0)
		return false;

	uint64 img_sz = uint64(tomo()->params.numX) * uint64(tomo()->params.numY);
	float R = tomo()->params.sod;
	float w = window_width * tomo()->params.voxelHeight;

	float z_max_center = tomo()->params.zFOV_max();
	float z_min_center = tomo()->params.zFOV_min();

	// whole volume: [0, numZ-1]
	// top cap: 	 [numZ-numZ_top, numZ-1]

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int n = 0; n < numZ_top; n++)
	{
		int iz = n + (tomo()->params.numZ - numZ_top);
		float z_cur = iz*tomo()->params.voxelHeight + tomo()->params.z_0();

		float* aSlice = &f[uint64(iz)*img_sz];
		float* aSlice_top = &f_top[uint64(n)*img_sz];
		for (int iy = 0; iy < tomo()->params.numY; iy++)
		{
			float y = iy*tomo()->params.voxelWidth + tomo()->params.y_0();
			for (int ix = 0; ix < tomo()->params.numX; ix++)
			{
				uint64 ind = uint64(iy)*uint64(tomo()->params.numX) + uint64(ix);

				float x = ix*tomo()->params.voxelWidth + tomo()->params.x_0();
				float r = sqrt(x*x + y*y);

				float dist = max(float(0.0), min(float(1.0), (z_cur - (z_max_center-w)*(R - r)/R)/w));

				if (dist > 0.0)
					aSlice[ind] = (1.0-dist)*aSlice[ind] + dist*aSlice_top[ind];
			}
		}
	}

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int n = 0; n < numZ_bot; n++)
	{
		int iz = n;
		float z_cur = iz*tomo()->params.voxelHeight + tomo()->params.z_0();

		float* aSlice = &f[uint64(iz)*img_sz];
		float* aSlice_bot = &f_bot[uint64(n)*img_sz];
		for (int iy = 0; iy < tomo()->params.numY; iy++)
		{
			float y = iy*tomo()->params.voxelWidth + tomo()->params.y_0();
			for (int ix = 0; ix < tomo()->params.numX; ix++)
			{
				uint64 ind = uint64(iy)*uint64(tomo()->params.numX) + uint64(ix);

				float x = ix*tomo()->params.voxelWidth + tomo()->params.x_0();
				float r = sqrt(x*x + y*y);

				float dist = max(float(0.0), min(float(1.0), ((z_min_center+w)*(R - r)/R - z_cur)/w));

				if (dist > 0.0)
					aSlice[ind] = (1.0-dist)*aSlice[ind] + dist*aSlice_bot[ind];
			}
		}
	}

	return true;
}

bool rebin_curved(float* g, float* fanAngles, int order)
{
	return tomo()->rebin_curved(g, fanAngles, order);
}

bool rebin_parallel(float* g, int order)
{
	return tomo()->rebin_parallel(g, order);
}

int rebin_parallel_sinogram(float* g, float* output, int order, int desiredRow)
{
	rebin rebinningRoutines;
	return rebinningRoutines.rebin_parallel_singleSinogram(g, &(tomo()->params), output, order, desiredRow);
}

bool sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize, int padSide)
{
	return tomo()->sinogram_replacement(g, priorSinogram, metalTrace, windowSize, padSide);
}

bool sinogram_replacement_interp(float* g, int* windowSize, int padSide)
{
	return tomo()->sinogram_replacement(g, NULL, NULL, windowSize, padSide);
}

bool down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, int order, float* offset, float maxWidth, bool data_on_cpu)
{
	return tomo()->down_sample(I, N, I_dn, N_dn, factors, order, offset, maxWidth, data_on_cpu);
}

bool up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, int order, int set_type, bool data_on_cpu)
{
	return tomo()->up_sample(I, N, I_up, N_up, factors, order, set_type, data_on_cpu);
}

bool resample_projection_angles(float* g, float* g_new, float* phis_new, int N_phis_new)
{
	return resampleProjectionAngles_cpu(g, &(tomo()->params), g_new, phis_new, N_phis_new);
}

bool antialiasFilter(float* volume, int* N, float L, int order, float* h, int N_h, bool* axis, bool data_on_cpu)
{
	if (data_on_cpu)
		return antialias_filter(volume, N, L, order, h, N_h, axis);
	else
		return false;
}

bool finiteDifference(float* volume, int* N, int order, int shift, bool* axis, float scalar, bool data_on_cpu)
{
	if (data_on_cpu)
		return finite_difference(volume, N, order, shift, axis, scalar);
	else
		return false;
}

bool scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType)
{
	return tomo()->scatter_model(g, f, source, energies, N_energies, detector, sigma, scatterDist, data_on_cpu, jobType);
}

bool scatter_simulation(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float reference_energy, const char** chemForms, int num_materials, float* densities, float* b_L, float* b_H, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters)
{
	return tomo()->scatter_simulation(g, f, source, energies, N_energies, detector, reference_energy, chemForms, num_materials, densities, b_L, b_H, data_on_cpu, num_photons_per_pixel, min_scatters, max_scatters);
}

bool detector_scatter_simulation(
	float* events, float thickness, float mass_density, float* source, float* energies, int N_energies, const char* chemForm, int num_photons, int max_scatters, float* direction)
{
	return tomo()->detector_scatter_simulation(
		events, thickness, mass_density, source, energies, N_energies, chemForm, num_photons, max_scatters, direction);
}

bool polychromatic_attenuation(float* spectralResponse, float* gammas, float referenceEnergy, float* g_1, float* sigma_1, float* g_2, float* sigma_2, float* g_3, float* sigma_3, float* g_poly, int N_gamma)
{
	if ((g_2 == nullptr && sigma_2 != nullptr) || (g_2 != nullptr && sigma_2 == nullptr))
		return false;
	if ((g_3 == nullptr && sigma_3 != nullptr) || (g_3 != nullptr && sigma_3 == nullptr))
		return false;
	int N_1 = tomo()->params.numAngles;
	int N_2 = tomo()->params.numRows;
	int N_3 = tomo()->params.numCols;

	uint64 img_size = uint64(N_2)*uint64(N_3);

	float ind_ref = 0.0;
	if (referenceEnergy <= gammas[0])
        ind_ref = 0.0;
	else if (referenceEnergy >= gammas[N_gamma-1])
		ind_ref = float(N_gamma-1);
	else
	{
		for (int i = 1; i < N_gamma; i++)
		{
			if (referenceEnergy <= gammas[i])
			{
				float d = (referenceEnergy - gammas[i - 1]) / (gammas[i] - gammas[i - 1]);
				ind_ref = float(i - 1) + d;
				break;
			}
		}
	}

    int ind_lo = int(ind_ref);
    int ind_hi = min(N_gamma - 1, ind_lo + 1);
    float h = ind_ref - float(ind_lo);
	float sigma_1_ref, sigma_2_ref, sigma_3_ref;
    sigma_1_ref = (1.0 - h) * sigma_1[ind_lo] + h * sigma_1[ind_hi];
	if (sigma_2 != nullptr)
    	sigma_2_ref = (1.0 - h) * sigma_2[ind_lo] + h * sigma_2[ind_hi];
	if (sigma_3 != nullptr)
	    sigma_3_ref = (1.0 - h) * sigma_3[ind_lo] + h * sigma_3[ind_hi];

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N_1; i++)
	{
		float* proj_1 = &g_1[uint64(i)*img_size];
		float* proj_poly = &g_poly[uint64(i)*img_size];
		float* proj_2 = nullptr;
		float* proj_3 = nullptr;
		if (g_2 != nullptr)
			proj_2 = &g_2[uint64(i)*img_size];
		if (g_3 != nullptr)
			proj_3 = &g_3[uint64(i)*img_size];
		for (uint64 j = 0; j < img_size; j++)
		{
			double val = 0.0;

			float a_1, a_2, a_3;
			a_1 = proj_1[j] / sigma_1_ref;
			if (proj_2 != nullptr)
				a_2 = proj_2[j] / sigma_2_ref;
			if (proj_3 != nullptr)
				a_3 = proj_3[j] / sigma_3_ref;

			for (int l = 0; l < N_gamma; l++)
			{
				double cur = a_1*sigma_1[l];
				if (proj_2 != nullptr)
					cur += a_2*sigma_2[l];
				if (proj_3 != nullptr)
					cur += a_3*sigma_3[l];
				val += spectralResponse[l] * exp(-cur);
			}
			proj_poly[j] = -log(val);
		}
	}

	return true;
}

bool synthesize_symmetry(float* f_radial, float* f)
{
	return tomo()->synthesize_symmetry(f_radial, f);
}

bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu)
{
	return tomo()->AzimuthalBlur(f, FWHM, data_on_cpu);
}


bool divide(float* I, float* J, int N_1, int N_2, int N_3, float divide_by_zero_value, bool skip_zero_denominator, bool data_on_cpu)
{
	if (I == NULL || J == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	else
	{
		if (data_on_cpu)
		{
			uint64 img_size = uint64(N_2)*uint64(N_3);

			omp_set_num_threads(num_cpu_threads());
			#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < N_1; i++)
			{
				float* lhs = &I[uint64(i)*img_size];
				float* rhs = &J[uint64(i)*img_size];
				for (uint64 j = 0; j < img_size; j++)
				{
					if (rhs[j] == 0.0)
					{
						if (!skip_zero_denominator)
							lhs[j] = divide_by_zero_value;
					}
					else
						lhs[j] = lhs[j] / rhs[j];
				}
			}
			return true;
		}
		else
		{
			#ifdef __USE_CPU
				printf("GPU operations not available in this release!\n");
				return false;
			#else
			divide(I, J, make_int3(N_1, N_2, N_3), tomo()->params.whichGPU, skip_zero_denominator);
			return true;
			#endif
		}
	}
}

bool reciprocal(float* I, int N_1, int N_2, int N_3, float divide_by_zero_value, bool data_on_cpu)
{
	if (I == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	else
	{
		if (data_on_cpu)
		{
			uint64 img_size = uint64(N_2)*uint64(N_3);

			omp_set_num_threads(num_cpu_threads());
			#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < N_1; i++)
			{
				float* lhs = &I[uint64(i)*img_size];
				for (uint64 j = 0; j < img_size; j++)
				{
					if (lhs[j] == 0.0)
					{
						lhs[j] = divide_by_zero_value;
					}
					else
						lhs[j] = 1.0 / lhs[j];
				}
			}
			return true;
		}
		else
		{
			#ifdef __USE_CPU
				printf("GPU operations not available in this release!\n");
				return false;
			#else
			reciprocal(I, make_int3(N_1, N_2, N_3), divide_by_zero_value, tomo()->params.whichGPU);
			return true;
			#endif
		}
	}
}

bool has_nan_or_inf(float* I, int N_1, int N_2, int N_3)
{
	return has_nan(I, N_1, N_2, N_3);
}

bool replaceNAN(float* I, int N_1, int N_2, int N_3, float newValue)
{
	return replace_nan(I, N_1, N_2, N_3, newValue);
}

bool boundingBox(float* I, int N_1, int N_2, int N_3, int boundary_type, int* AABB)
{
	return bounding_box(I, N_1, N_2, N_3, boundary_type, AABB);
}

bool heaviside(float* I, int N_1, int N_2, int N_3, float scale, float shift)
{
	return step_function(I, N_1, N_2, N_3, scale, shift);
}

bool dirac(float* I, int N_1, int N_2, int N_3, float scale, float shift)
{
	return dirac_function(I, N_1, N_2, N_3, scale, shift);
}

bool quantize(float* I, int N_1, int N_2, int N_3, float* targets, int numTargets)
{
	if (I == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || targets == NULL || numTargets <= 0)
		return false;
	else if (numTargets == 1)
		return equal_cpu(I, targets[0], N_1, N_2, N_3);
	else
	{
		float firstDivision = 0.5*(targets[0] + targets[1]);

		omp_set_num_threads(num_cpu_threads());
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < N_1; i++)
		{
			float* anImage = &I[uint64(i)*uint64(N_2)*uint64(N_3)];
			for (uint64 ind = 0; ind < uint64(N_2*N_3); ind++)
			{
				float curVal = anImage[ind];
				int min_ind = 0;
				if (curVal > firstDivision)
				{
					float minDiff = fabs(curVal - targets[0]);
					for (int n = 1; n < numTargets; n++)
					{
						float curDiff = fabs(curVal - targets[n]);
						if (curDiff < minDiff)
						{
							minDiff = curDiff;
							min_ind = n;
						}
					}
				}
				anImage[ind] = targets[min_ind];
			}
		}
		return true;
	}
}

bool inpaint(float* I, int N_1, int N_2, int N_3)
{
	return inpaint3D(I, N_1, N_2, N_3);
}

bool threshold(float* I, int N_1, int N_2, int N_3, float value, bool greater_than, int fill_type, int num_pixel_dilate)
{
	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N_1; i++)
	{
		segmentation segRoutines;
		float* anImage = &I[uint64(i)*uint64(N_2)*uint64(N_3)];
		segRoutines.init(anImage, N_2, N_3);
		segRoutines.threshold(value, greater_than, fill_type, num_pixel_dilate);
	}
	return true;
}

bool region_growing(float* I, int N_1, int N_2, int N_3, float startThreshold, float endThreshold, int fill_type, int num_pixel_dilate)
{
	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N_1; i++)
	{
		segmentation segRoutines;
		float* anImage = &I[uint64(i)*uint64(N_2)*uint64(N_3)];
		float* anImage_lo = NULL;
		float* anImage_hi = NULL;
		//*
		if (i > 0)
			anImage_lo = &I[uint64(i-1)*uint64(N_2)*uint64(N_3)];
		if (i < N_1-1)
			anImage_hi = &I[uint64(i+1)*uint64(N_2)*uint64(N_3)];
		//*/
		segRoutines.init(anImage, N_2, N_3, anImage_lo, anImage_hi);
		segRoutines.region_growing(startThreshold, endThreshold, fill_type, num_pixel_dilate);
	}
	return true;
}

bool dilate(float* I, int N_1, int N_2, int N_3, int pixelRadius, int fill_type)
{
	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < N_1; i++)
	{
		segmentation segRoutines;
		float* anImage = &I[uint64(i)*uint64(N_2)*uint64(N_3)];
		segRoutines.init(anImage, N_2, N_3);
		segRoutines.dilate(pixelRadius, fill_type);
	}
	return true;
}

bool k_means(float* I, int N_1, int N_2, int N_3, float* means, int K)
{
	return kmeans(I, N_1, N_2, N_3, means, K);
}

bool Otsu_thresholds(float* I, int N_1, int N_2, int N_3, float* thresholds, int K)
{
	return Otsu(I, N_1, N_2, N_3, thresholds, K);
}

bool extrema(float* I, int N_1, int N_2, int N_3, float* minmax)
{
	if (minmax == NULL)
		return false;
	else
	{
		float minValue, maxValue;
		if (range(I, N_1, N_2, N_3, minValue, maxValue))
		{
			minmax[0] = minValue;
			minmax[1] = maxValue;
			return true;
		}
		else
			return false;
	}
}

bool basic_stats(float* I, int N_1, int N_2, int N_3, float* stats)
{
	return basicStats(I, N_1, N_2, N_3, stats);
}

bool percentile2D(float* I, int N_images, int N, float q, float* percentiles)
{
	return percentile_2D(I, N_images, N, q, percentiles);
}

float histogram3D(float* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

float histogram3D_uint8(uint8_t* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

float histogram3D_int16(int16_t* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

float histogram3D_uint16(uint16_t* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

float histogram3D_int32(int32_t* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

float histogram3D_uint32(uint32_t* I, int N_1, int N_2, int N_3, float* h, float* bins, int numBins, float rangeMin, float rangeMax)
{
	if (I == NULL || h == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || bins == NULL)
		return 0.0;
	else
	{
		float binSize = 0.0;
		histogram(I, N_1, N_2, N_3, numBins, binSize, h, bins, true, rangeMin, rangeMax);
		return binSize;
	}
}

bool covariance(float* I, int N_1, int N_2, int N_3, float threshold, float* cov, float* com)
{
	return calculate_covariance(I, N_1, N_2, N_3, threshold, cov, com);
}

bool center_of_mass(float* I, int N_1, int N_2, int N_3, float threshold, float* com)
{
	return calculate_centroid(I, N_1, N_2, N_3, threshold, com);
}

bool valueMask(float* I, int N_1, int N_2, int N_3, float a, float b, float c, float d, bool deriv)
{
	if (I == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || a > b || b > c || c > d)
		return false;
	else
	{
		omp_set_num_threads(num_cpu_threads());
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < N_1; i++)
		{
			float* anImage = &I[uint64(i)*uint64(N_2)*uint64(N_3)];
			if (deriv)
			{
				for (uint64 ind = 0; ind < uint64(N_2*N_3); ind++)
				{
					float curVal = anImage[ind];
					if (curVal <= a || curVal >= d)
						anImage[ind] = 0.0;
					else if (a < curVal && curVal < b)
						anImage[ind] = (2.0*curVal - a) / (b - a);
					else if (c < curVal && curVal < d)
						anImage[ind] = (d - 2.0*curVal) / (d - c);
					else //if (b <= curVal && curVal <= c)
						anImage[ind] = 1.0;
				}
			}
			else
			{
				for (uint64 ind = 0; ind < uint64(N_2*N_3); ind++)
				{
					float curVal = anImage[ind];
					if (curVal <= a || curVal >= d)
						anImage[ind] = 0.0;
					else if (a < curVal && curVal < b)
						anImage[ind] = curVal * (curVal - a) / (b - a);
					else if (c < curVal && curVal < d)
						anImage[ind] = curVal * (d - curVal) / (d - c);
				}
			}
		}
		return true;
	}
}

bool sum_first_axis(float* I, int N_1, int N_2, int N_3, float* sums)
{
	return sum_first_dimension(I, N_1, N_2, N_3, sums);
}

bool sum_axis(float* I, int N_1, int N_2, int N_3, float* sums, int axis)
{
	return sum_dimension(I, N_1, N_2, N_3, sums, axis);
}

bool saveParamsToFile(const char* param_fn)
{
	return saveParametersToFile(param_fn, &(tomo()->params));
}

bool save_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight, float pixelWidth, int dtype, float wmin, float wmax)
{
	return write_tif(fileName, data, numRows, numCols, pixelHeight, pixelWidth, dtype, wmin, wmax);
}

bool read_tif_header(char* fileName, int* shape, float* size, float* slope_and_offset)
{
	return read_header(fileName, shape, size, slope_and_offset);
}

bool read_tif(char* fileName, float* data)
{
	if (load_tif(fileName, data) == NULL)
		return false;
	else
		return true;
}

bool read_tif_rows(char* fileName, int firstRow, int lastRow, float* data)
{
	if (load_tif_rows(fileName, firstRow, lastRow, data) == NULL)
		return false;
	else
		return true;
}

bool read_tif_cols(char* fileName, int firstCol, int lastCol, float* data)
{
	if (load_tif_cols(fileName, firstCol, lastCol, data) == NULL)
		return false;
	else
		return true;
}

bool read_tif_roi(char* fileName, int firstRow, int lastRow, int firstCol, int lastCol, float* data)
{
	if (load_tif_roi(fileName, firstRow, lastRow, firstCol, lastCol, data) == NULL)
		return false;
	else
		return true;
}

void test_script()
{
	
}

/*
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(leapct, m) {
    m.def("set_model", &set_model, "");
    m.def("create_new_model", &create_new_model, "");
    m.def("copy_parameters", &copy_parameters, "");
    m.def("about", &about, "");
	m.def("version", &version, "");
    m.def("print_parameters", &print_parameters, "");
    m.def("reset", &reset, "");
    m.def("include_cufft", &include_cufft, "");
	m.def("set_log_error", &set_log_error, "");
	m.def("set_log_warning", &set_log_warning, "");
	m.def("set_log_status", &set_log_status, "");
	m.def("set_log_debug", &set_log_debug, "");
    m.def("getOptimalFFTsize", &getOptimalFFTsize, "");
	m.def("set_maxSlicesForChunking", &set_maxSlicesForChunking, "");
	m.def("get_maxSlicesForChunking", &get_maxSlicesForChunking, "");
    m.def("verify_input_sizes", &verify_input_sizes, "");
    m.def("project_gpu", &project_gpu, "");
    m.def("backproject_gpu", &backproject_gpu, "");
    m.def("project_cpu", &project_cpu, "");
	m.def("project_with_mask", &project_with_mask, "");
	m.def("project_with_mask_gpu", &project_with_mask_gpu, "");
	m.def("project_with_mask_cpu", &project_with_mask_cpu, "");
    m.def("backproject_cpu", &backproject_cpu, "");
    m.def("FBP_cpu", &FBP_cpu, "");
    m.def("FBP_gpu", &FBP_gpu, "");
    m.def("project", &project, "");
    m.def("backproject", &backproject, "");
    m.def("weightedBackproject", &weightedBackproject, "");
	m.def("negLog", &negLog, "");
	m.def("expNeg", &expNeg, "");
    m.def("HilbertFilterProjections", &HilbertFilterProjections, "");
    m.def("rampFilterProjections", &rampFilterProjections, "");
    m.def("filterProjections", &filterProjections, "");
	m.def("filterProjections_gpu", &filterProjections_gpu, "");
	m.def("filterProjections_cpu", &filterProjections_cpu, "");
	m.def("extraColumnsForOffsetScan", &extraColumnsForOffsetScan, "");
	m.def("get_offsetScan_weights", &get_offsetScan_weights, "");
	m.def("preRampFiltering", &preRampFiltering, "");
	m.def("postRampFiltering", &postRampFiltering, "");
    m.def("rampFilterVolume", &rampFilterVolume, "");
    m.def("get_FBPscalar", &get_FBPscalar, "");
    m.def("FBP", &FBP, "");
    m.def("inconsistencyReconstruction", &inconsistencyReconstruction, "");
	m.def("lambdaTomography", &lambdaTomography, "");
    m.def("sensitivity", &sensitivity, "");
    m.def("windowFOV", &windowFOV, "");
    m.def("set_conebeam", &set_conebeam, "");
	m.def("set_coneparallel", &set_coneparallel, "");
    m.def("set_fanbeam", &set_fanbeam, "");
    m.def("set_parallelbeam", &set_parallelbeam, "");
    m.def("set_modularbeam", &set_modularbeam, "");
    m.def("rotate_detector", &rotate_detector, "");
    m.def("shift_detector", &shift_detector, "");
    m.def("set_flatDetector", &set_flatDetector, "");
    m.def("set_curvedDetector", &set_curvedDetector, "");
    m.def("get_detectorType", &get_detectorType, "");
    m.def("set_numCols", &set_numCols, "");
    m.def("set_numRows", &set_numRows, "");
	m.def("set_numAngles", &set_numAngles, "");
    m.def("set_pixelHeight", &set_pixelHeight, "");
    m.def("set_pixelWidth", &set_pixelWidth, "");
    m.def("set_centerCol", &set_centerCol, "");
    m.def("set_centerRow", &set_centerRow, "");
    m.def("set_volume", &set_volume, "");
    m.def("set_volumeDimensionOrder", &set_volumeDimensionOrder, "");
    m.def("get_volumeDimensionOrder", &get_volumeDimensionOrder, "");
    m.def("set_default_volume", &set_default_volume, "");
    m.def("set_numZ", &set_numZ, "");
    m.def("set_numY", &set_numY, "");
    m.def("set_numX", &set_numX, "");
	m.def("set_offsetX", &set_offsetX, "");
	m.def("set_offsetY", &set_offsetY, "");
    m.def("set_offsetZ", &set_offsetZ, "");
    m.def("set_voxelWidth", &set_voxelWidth, "");
    m.def("set_voxelHeight", &set_voxelHeight, "");
	m.def("set_geometry", &set_geometry, "");
	m.def("angles_are_defined", &angles_are_defined, "");
	m.def("angles_are_equispaced", &angles_are_equispaced, "");
    m.def("projectConeBeam", &projectConeBeam, "");
    m.def("backprojectConeBeam", &backprojectConeBeam, "");
    m.def("projectFanBeam", &projectFanBeam, "");
    m.def("backprojectFanBeam", &backprojectFanBeam, "");
    m.def("projectParallelBeam", &projectParallelBeam, "");
    m.def("backprojectParallelBeam", &backprojectParallelBeam, "");
    m.def("rowRangeNeededForBackprojection", &rowRangeNeededForBackprojection, "");
	m.def("viewRangeNeededForBackprojection", &viewRangeNeededForBackprojection, "");
	m.def("sliceRangeNeededForProjection", &sliceRangeNeededForProjection, "");
	m.def("numRowsRequiredForBackprojectingSlab", &numRowsRequiredForBackprojectingSlab, "");
    m.def("number_of_gpus", &number_of_gpus, "");
	m.def("get_gpus", &get_gpus, "");
	m.def("set_GPU", &set_GPU, "");
    m.def("set_GPUs", &set_GPUs, "");
    m.def("get_GPU", &get_GPU, "");
    m.def("set_axisOfSymmetry", &set_axisOfSymmetry, "");
	m.def("get_axisOfSymmetry", &get_axisOfSymmetry, "");
    m.def("clear_axisOfSymmetry", &clear_axisOfSymmetry, "");
    m.def("set_projector", &set_projector, "");
	m.def("get_projector", &get_projector, "");
    m.def("set_rFOV", &set_rFOV, "");
	m.def("get_rFOV", &get_rFOV, "");
	m.def("get_rFOV_min", &get_rFOV_min, "");
	m.def("get_rFOV_max", &get_rFOV_max, "");
    m.def("set_offsetScan", &set_offsetScan, "");
	m.def("get_offsetScan", &get_offsetScan, "");
    m.def("set_truncatedScan", &set_truncatedScan, "");
	m.def("get_truncatedScan", &get_truncatedScan, "");
    m.def("set_numTVneighbors", &set_numTVneighbors, "");
    m.def("get_numTVneighbors", &get_numTVneighbors, "");
    m.def("set_rampID", &set_rampID, "");
	m.def("get_rampID", &get_rampID, "");
	m.def("set_FBPlowpass", &set_FBPlowpass, "");
	m.def("get_FBPlowpass", &get_FBPlowpass, "");
    m.def("set_tau", &set_tau, "");
    m.def("set_helicalPitch", &set_helicalPitch, "");
    m.def("set_normalizedHelicalPitch", &set_normalizedHelicalPitch, "");
    m.def("set_attenuationMap", &set_attenuationMap, "");
    m.def("set_cylindircalAttenuationMap", &set_cylindircalAttenuationMap, "");
    m.def("convert_conebeam_to_modularbeam", &convert_conebeam_to_modularbeam, "");
    m.def("convert_parallelbeam_to_modularbeam", &convert_parallelbeam_to_modularbeam, "");
    m.def("clear_attenuationMap", &clear_attenuationMap, "");
    m.def("muSpecified", &muSpecified, "");
    m.def("flipAttenuationMapSign", &flipAttenuationMapSign, "");
    m.def("get_geometry", &get_geometry, "");
    m.def("get_sod", &get_sod, "");
    m.def("get_sdd", &get_sdd, "");
    m.def("get_numAngles", &get_numAngles, "");
    m.def("get_numRows", &get_numRows, "");
    m.def("get_numCols", &get_numCols, "");
    m.def("get_pixelWidth", &get_pixelWidth, "");
    m.def("get_pixelHeight", &get_pixelHeight, "");
    m.def("get_centerRow", &get_centerRow, "");
    m.def("get_centerCol", &get_centerCol, "");
    m.def("get_tau", &get_tau, "");
	m.def("get_tiltAngle", &get_tiltAngle, "");
    m.def("get_helicalPitch", &get_helicalPitch, "");
    m.def("get_normalizedHelicalPitch", &get_normalizedHelicalPitch, "");
    m.def("get_z_source_offset", &get_z_source_offset, "");
    m.def("get_sourcePositions", &get_sourcePositions, "");
    m.def("get_moduleCenters", &get_moduleCenters, "");
    m.def("get_rowVectors", &get_rowVectors, "");
    m.def("get_colVectors", &get_colVectors, "");
    m.def("set_angles", &set_angles, "");
    m.def("get_angles", &get_angles, "");
    m.def("get_angularRange", &get_angularRange, "");
    m.def("get_numX", &get_numX, "");
    m.def("get_numY", &get_numY, "");
    m.def("get_numZ", &get_numZ, "");
    m.def("get_voxelWidth", &get_voxelWidth, "");
    m.def("get_voxelHeight", &get_voxelHeight, "");
    m.def("get_offsetX", &get_offsetX, "");
    m.def("get_offsetY", &get_offsetY, "");
    m.def("get_offsetZ", &get_offsetZ, "");
    m.def("get_z0", &get_z0, "");
    m.def("find_centerCol", &find_centerCol, "");
	m.def("find_tau", &find_tau, "");
	m.def("estimate_tilt", &estimate_tilt, "");
	m.def("conjugate_difference", &conjugate_difference, "");
    m.def("Laplacian", &Laplacian, "");
	m.def("ring_removal", &ring_removal, "");
    m.def("transmissionFilter", &transmissionFilter, "");
    m.def("applyTransferFunction", &applyTransferFunction, "");
	m.def("beam_hardening_heel_effect", &beam_hardening_heel_effect, "");
    m.def("applyDualTransferFunction", &applyDualTransferFunction, "");
    m.def("convertToRhoeZe", &convertToRhoeZe, "");
    m.def("BlurFilter", &BlurFilter, "");
	m.def("HighPassFilter", &HighPassFilter, "");
    m.def("MedianFilter", &MedianFilter, "");
	m.def("MeanOrVarianceFilter", &MeanOrVarianceFilter, "");
    m.def("BlurFilter2D", &BlurFilter2D, "");
	m.def("HighPassFilter2D", &HighPassFilter2D, "");
    m.def("MedianFilter2D", &MedianFilter2D, "");
	m.def("badPixelCorrection", &badPixelCorrection, "");
    m.def("BilateralFilter", &BilateralFilter, "");
	m.def("PriorBilateralFilter", &PriorBilateralFilter, "");
	m.def("GuidedFilter", &GuidedFilter, "");
    m.def("dictionaryDenoising", &dictionaryDenoising, "");
    m.def("TVcost", &TVcost, "");
    m.def("TVgradient", &TVgradient, "");
    m.def("TVquadForm", &TVquadForm, "");
    m.def("Diffuse", &Diffuse, "");
	m.def("TV_denoise", &TV_denoise, "");
    m.def("addObject", &addObject, "");
	m.def("voxelize", &voxelize, "");
    m.def("clearPhantom", &clearPhantom, "");
	m.def("scalePhantom", &scalePhantom, "");
    m.def("rayTrace", &rayTrace, "");
    m.def("rebin_curved", &rebin_curved, "");
	m.def("rebin_parallel", &rebin_parallel, "");
	m.def("rebin_parallel_sinogram", &rebin_parallel_sinogram, "");
    m.def("sinogram_replacement", &sinogram_replacement, "");
    m.def("down_sample", &down_sample, "");
    m.def("up_sample", &up_sample, "");
	m.def("scatter_model", &scatter_model, "");
	m.def("scatter_simulation", &scatter_simulation, "");
	m.def("detector_scatter_simulation", &detector_scatter_simulation, "");
    m.def("synthesize_symmetry", &synthesize_symmetry, "");
    m.def("AzimuthalBlur", &AzimuthalBlur, "");
	m.def("inpaint", &inpaint, "");
	m.def("threshold", &threshold, "");
	m.def("region_growing", &region_growing, "");
	m.def("dilate", &dilate, "");
	m.def("k_means", &k_means, "");
    m.def("saveParamsToFile", &saveParamsToFile, "");
	m.def("save_tif", &save_tif, "");
	m.def("read_tif_header", &read_tif_header, "");
	m.def("read_tif", &read_tif, "");
	m.def("read_tif_rows", &read_tif_rows, "");
	m.def("read_tif_cols", &read_tif_cols, "");
}
//*/
