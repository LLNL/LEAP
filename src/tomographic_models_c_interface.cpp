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
#include "phantom.h"
#include "file_io.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

bool copy_parameters(int param_id)
{
	if (0 <= param_id && param_id < list_models.size())
	{
		if (whichModel != param_id)
		{
			//printf("copy %d => %d\n", param_id, whichModel);
			//list_models.get(param_id)->params.assign(tomo()->params);
			tomo()->params.assign(list_models.get(param_id)->params);
		}
		return true;
	}
	else
		return false;
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
	return tomo()->params.allDefined();
}

bool ct_geometry_defined()
{
	return tomo()->params.geometryDefined();
}

bool ct_volume_defined()
{
	return tomo()->params.volumeDefined();
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

bool backproject_gpu(float* g, float* f)
{
	return tomo()->backproject_gpu(g, f);
}

bool project_cpu(float* g, float* f)
{
	return tomo()->project_cpu(g, f);
}

bool backproject_cpu(float* g, float* f)
{
	return tomo()->backproject_cpu(g, f);
}

bool project(float* g, float* f, bool data_on_cpu)
{
	return tomo()->project(g, f, data_on_cpu);
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

bool weightedBackproject(float* g, float* f, bool data_on_cpu)
{
	return tomo()->weightedBackproject(g, f, data_on_cpu);
}

bool filterProjections(float* g, bool data_on_cpu)
{
	return tomo()->filterProjections(g, data_on_cpu);
}

bool HilbertFilterProjections(float* g, bool data_on_cpu, float scalar)
{
	return tomo()->HilbertFilterProjections(g, data_on_cpu, scalar);
}

bool rampFilterProjections(float* g, bool data_on_cpu, float scalar)
{
	return tomo()->rampFilterProjections(g, data_on_cpu, scalar);
}

bool rampFilterVolume(float* f, bool data_on_cpu)
{
	return tomo()->rampFilterVolume(f, data_on_cpu);
}

bool FBP(float* g, float* f, bool data_on_cpu)
{
	return tomo()->doFBP(g, f, data_on_cpu);
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

bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float helicalPitch)
{
	return tomo()->set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch);
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

bool set_volumeDimensionOrder(int which)
{
	return tomo()->set_volumeDimensionOrder(which);
}

int get_volumeDimensionOrder()
{
	return tomo()->get_volumeDimensionOrder();
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

bool set_projector(int which)
{
	return tomo()->set_projector(which);
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

bool set_offsetScan(bool aFlag)
{
	return tomo()->params.set_offsetScan(aFlag);
}

bool set_truncatedScan(bool aFlag)
{
	return tomo()->params.set_truncatedScan(aFlag);
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

bool set_tau(float tau)
{
	return tomo()->set_tau(tau);
}

bool set_helicalPitch(float h)
{
	return tomo()->set_helicalPitch(h);
}

bool set_normalizedHelicalPitch(float h_normalized)
{
	return tomo()->set_normalizedHelicalPitch(h_normalized);
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

bool set_geometry(int which)
{
	//CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3
	if (which < parameters::CONE || which > parameters::MODULAR)
		return false;
	else
	{
		tomo()->params.geometry = which;
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

bool sliceRangeNeededForProjection(int* slicesNeeded, bool doClip)
{
	if (slicesNeeded == NULL || tomo()->params.allDefined() == false)
		return false;
	else
		return tomo()->params.sliceRangeNeededForProjection(0, tomo()->params.numRows - 1, slicesNeeded, doClip);
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

float get_helicalPitch()
{
	return tomo()->get_helicalPitch();
}

float get_normalizedHelicalPitch()
{
	return tomo()->params.normalizedHelicalPitch();
}

float get_z_source_offset()
{
	return tomo()->get_z_source_offset();
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

float get_angularRange()
{
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

bool find_centerCol(float* g, int iRow, bool data_on_cpu)
{
	return tomo()->find_centerCol(g, iRow, data_on_cpu);
}

float consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu)
{
	return tomo()->consistency_cost(g, Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt, data_on_cpu);
}

bool Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu)
{
	return tomo()->Laplacian(g, numDims, smooth, data_on_cpu);
}

bool transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, bool data_on_cpu)
{
	return tomo()->transmissionFilter(g, H, N_H1, N_H2, isAttenuationData, data_on_cpu);
}

bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	return tomo()->applyTransferFunction(x, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, data_on_cpu);
}

bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
{
	return tomo()->applyDualTransferFunction(x, y, N_1, N_2, N_3, LUT, firstSample, sampleRate, numSamples, data_on_cpu);
}

bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool data_on_cpu)
{
	return tomo()->convertToRhoeZe(f_L, f_H, N_1, N_2, N_3, sigma_L, sigma_H, data_on_cpu);
}

bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->BlurFilter(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool HighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->HighPassFilter(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, bool data_on_cpu)
{
	return tomo()->MedianFilter(f, N_1, N_2, N_3, threshold, w, data_on_cpu);
}

bool MeanOrVarianceFilter(float* f, int N_1, int N_2, int N_3, int r, int order, bool data_on_cpu)
{
	return tomo()->MeanOrVarianceFilter(f, N_1, N_2, N_3, r, order, data_on_cpu);
}

bool BlurFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->BlurFilter2D(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool MedianFilter2D(float* f, int N_1, int N_2, int N_3, float threshold, int w, bool data_on_cpu)
{
	return tomo()->MedianFilter2D(f, N_1, N_2, N_3, threshold, w, data_on_cpu);
}

bool BilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu)
{
	return tomo()->BilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, scale, data_on_cpu);
}

bool PriorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* prior, bool data_on_cpu)
{
	return tomo()->PriorBilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, prior, data_on_cpu);
}

bool GuidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, bool data_on_cpu)
{
	return tomo()->GuidedFilter(f, N_1, N_2, N_3, r, epsilon, data_on_cpu);
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
	return tomo()->TVgradient(f, Df, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
}

float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu)
{
	return tomo()->TVquadForm(f, d, N_1, N_2, N_3, delta, beta, p, data_on_cpu);
}

bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu)
{
	return tomo()->Diffuse(f, N_1, N_2, N_3, delta, p, numIter, data_on_cpu);
}

bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip, int oversampling)
{
	return tomo()->geometricPhantom.addObject(f, &(tomo()->params), type, c, r, val, A, clip, oversampling);
}

bool clearPhantom()
{
	tomo()->geometricPhantom.clearObjects();
	return true;
}

bool rayTrace(float* g, int oversampling)
{
	return tomo()->rayTrace(g, oversampling);
}

bool rebin_curved(float* g, float* fanAngles, int order)
{
	return tomo()->rebin_curved(g, fanAngles, order);
}

bool sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize)
{
	return tomo()->sinogram_replacement(g, priorSinogram, metalTrace, windowSize);
}

bool down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, bool data_on_cpu)
{
	return tomo()->down_sample(I, N, I_dn, N_dn, factors, data_on_cpu);
}

bool up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, bool data_on_cpu)
{
	return tomo()->up_sample(I, N, I_up, N_up, factors, data_on_cpu);
}

bool scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType)
{
	return tomo()->scatter_model(g, f, source, energies, N_energies, detector, sigma, scatterDist, data_on_cpu, jobType);
}

bool synthesize_symmetry(float* f_radial, float* f)
{
	return tomo()->synthesize_symmetry(f_radial, f);
}

bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu)
{
	return tomo()->AzimuthalBlur(f, FWHM, data_on_cpu);
}

bool saveParamsToFile(const char* param_fn)
{
	return saveParametersToFile(param_fn, &(tomo()->params));
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
    m.def("verify_input_sizes", &verify_input_sizes, "");
    m.def("project_gpu", &project_gpu, "");
    m.def("backproject_gpu", &backproject_gpu, "");
    m.def("project_cpu", &project_cpu, "");
    m.def("backproject_cpu", &backproject_cpu, "");
    m.def("FBP_cpu", &FBP_cpu, "");
    m.def("FBP_gpu", &FBP_gpu, "");
    m.def("project", &project, "");
    m.def("backproject", &backproject, "");
    m.def("weightedBackproject", &weightedBackproject, "");
    m.def("HilbertFilterProjections", &HilbertFilterProjections, "");
    m.def("rampFilterProjections", &rampFilterProjections, "");
    m.def("filterProjections", &filterProjections, "");
    m.def("rampFilterVolume", &rampFilterVolume, "");
    m.def("get_FBPscalar", &get_FBPscalar, "");
    m.def("FBP", &FBP, "");
    m.def("inconsistencyReconstruction", &inconsistencyReconstruction, "");
	m.def("lambdaTomography", &lambdaTomography, "");
    m.def("sensitivity", &sensitivity, "");
    m.def("windowFOV", &windowFOV, "");
    m.def("set_conebeam", &set_conebeam, "");
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
    m.def("set_offsetZ", &set_offsetZ, "");
    m.def("set_voxelWidth", &set_voxelWidth, "");
    m.def("set_voxelHeight", &set_voxelHeight, "");
	m.def("set_geometry", &set_geometry, "");
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
    m.def("set_GPU", &set_GPU, "");
    m.def("set_GPUs", &set_GPUs, "");
    m.def("get_GPU", &get_GPU, "");
    m.def("set_axisOfSymmetry", &set_axisOfSymmetry, "");
	m.def("get_axisOfSymmetry", &get_axisOfSymmetry, "");
    m.def("clear_axisOfSymmetry", &clear_axisOfSymmetry, "");
    m.def("set_projector", &set_projector, "");
    m.def("set_rFOV", &set_rFOV, "");
    m.def("set_offsetScan", &set_offsetScan, "");
    m.def("set_truncatedScan", &set_truncatedScan, "");
    m.def("set_numTVneighbors", &set_numTVneighbors, "");
    m.def("get_numTVneighbors", &get_numTVneighbors, "");
    m.def("set_rampID", &set_rampID, "");
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
    m.def("Laplacian", &Laplacian, "");
    m.def("transmissionFilter", &transmissionFilter, "");
    m.def("applyTransferFunction", &applyTransferFunction, "");
    m.def("applyDualTransferFunction", &applyDualTransferFunction, "");
    m.def("convertToRhoeZe", &convertToRhoeZe, "");
    m.def("BlurFilter", &BlurFilter, "");
	m.def("HighPassFilter", &HighPassFilter, "");
    m.def("MedianFilter", &MedianFilter, "");
	m.def("MeanOrVarianceFilter", &MeanOrVarianceFilter, "");
    m.def("BlurFilter2D", &BlurFilter2D, "");
    m.def("MedianFilter2D", &MedianFilter2D, "");
    m.def("BilateralFilter", &BilateralFilter, "");
	m.def("PriorBilateralFilter", &PriorBilateralFilter, "");
	m.def("GuidedFilter", &GuidedFilter, "");
    m.def("dictionaryDenoising", &dictionaryDenoising, "");
    m.def("TVcost", &TVcost, "");
    m.def("TVgradient", &TVgradient, "");
    m.def("TVquadForm", &TVquadForm, "");
    m.def("Diffuse", &Diffuse, "");
    m.def("addObject", &addObject, "");
    m.def("clearPhantom", &clearPhantom, "");
    m.def("rayTrace", &rayTrace, "");
    m.def("rebin_curved", &rebin_curved, "");
    m.def("sinogram_replacement", &sinogram_replacement, "");
    m.def("down_sample", &down_sample, "");
    m.def("up_sample", &up_sample, "");
    m.def("scatter_model", &scatter_model, "");
    m.def("synthesize_symmetry", &synthesize_symmetry, "");
    m.def("AzimuthalBlur", &AzimuthalBlur, "");
    m.def("saveParamsToFile", &saveParamsToFile, "");
}
//*/
