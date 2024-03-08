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

bool print_parameters()
{
	return tomo()->print_parameters();
}

bool reset()
{
	return tomo()->reset();
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
	if (numCols >= 1)
	{
		tomo()->params.numCols = numCols;
		return true;
	}
	else
		return false;
}

bool set_numRows(int numRows)
{
	if (numRows >= 1)
	{
		tomo()->params.numRows = numRows;
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
	if (numZ > 0)
	{
		tomo()->params.numZ = numZ;
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

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int w, bool data_on_cpu)
{
	return tomo()->MedianFilter(f, N_1, N_2, N_3, threshold, w, data_on_cpu);
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

bool dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu)
{
	return tomo()->dictionaryDenoising(f, N_1, N_2, N_3, dictionary, numElements, N_d1, N_d2, N_d3, epsilon, sparsityThreshold, data_on_cpu);
}

float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu)
{
	return tomo()->TVcost(f, N_1, N_2, N_3, delta, beta, data_on_cpu);
}

bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu)
{
	return tomo()->TVgradient(f, Df, N_1, N_2, N_3, delta, beta, data_on_cpu);
}

float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu)
{
	return tomo()->TVquadForm(f, d, N_1, N_2, N_3, delta, beta, data_on_cpu);
}

bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool data_on_cpu)
{
	return tomo()->Diffuse(f, N_1, N_2, N_3, delta, numIter, data_on_cpu);
}

bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip, int oversampling)
{
	phantom testObject;
	return testObject.addObject(f, &(tomo()->params), type, c, r, val, A, clip, oversampling);
}

bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu)
{
	return tomo()->AzimuthalBlur(f, FWHM, data_on_cpu);
}

bool saveParamsToFile(const char* param_fn)
{
	return saveParametersToFile(param_fn, &(tomo()->params));
}
