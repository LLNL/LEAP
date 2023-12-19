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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>

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

bool print_parameters()
{
	return tomo()->print_parameters();
}

bool reset()
{
	return tomo()->reset();
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

bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo()->set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool set_default_volume(float scale)
{
	return tomo()->set_default_volume(scale);
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

bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu)
{
	return tomo()->BlurFilter(f, N_1, N_2, N_3, FWHM, data_on_cpu);
}

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool data_on_cpu)
{
	return tomo()->MedianFilter(f, N_1, N_2, N_3, threshold, data_on_cpu);
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

bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip)
{
	phantom testObject;
	return testObject.addObject(f, &(tomo()->params), type, c, r, val, A, clip);
}

bool saveParamsToFile(const char* param_fn)
{
	tomographicModels* p_model = tomo();
	if (p_model == NULL)
		return false;

	parameters* params = &(p_model->params);

	std::string phis_strs;
	for (int i = 0; i < params->numAngles; i++) {
		float phis = (params->phis[i] + 0.5 * PI) * 180.0 / PI;
		char phis_str[64];
		sprintf(phis_str, " %f", phis);
		phis_strs += phis_str;
		if (i != params->numAngles - 1)
			phis_strs += ",";
	}

	std::ofstream param_file;
	param_file.open(param_fn);
	//*
	param_file << "img_dimx = " << params->numX << std::endl;
	param_file << "img_dimy = " << params->numY << std::endl;
	param_file << "img_dimz = " << params->numZ << std::endl;
	param_file << "img_pwidth = " << params->voxelWidth << std::endl;
	param_file << "img_pheight = " << params->voxelHeight << std::endl;
	param_file << "img_offsetx = " << params->offsetX << std::endl;
	param_file << "img_offsety = " << params->offsetY << std::endl;
	param_file << "img_offsetz = " << params->offsetZ << std::endl;

	if (params->geometry == parameters::CONE)
		param_file << "proj_geometry = " << "cone" << std::endl;
	else if (params->geometry == parameters::PARALLEL)
		param_file << "proj_geometry = " << "parallel" << std::endl;
	else if (params->geometry == parameters::FAN)
		param_file << "proj_geometry = " << "fan" << std::endl;
	else if (params->geometry == parameters::MODULAR)
		param_file << "proj_geometry = " << "modular" << std::endl;
	param_file << "proj_arange = " << params->angularRange << std::endl;
	param_file << "proj_nangles = " << params->numAngles << std::endl;
	param_file << "proj_nrows = " << params->numRows << std::endl;
	param_file << "proj_ncols = " << params->numCols << std::endl;
	param_file << "proj_pheight = " << params->pixelHeight << std::endl;
	param_file << "proj_pwidth = " << params->pixelWidth << std::endl;
	param_file << "proj_crow = " << params->centerRow << std::endl;
	param_file << "proj_ccol = " << params->centerCol << std::endl;
	param_file << "proj_phis = " << phis_strs << std::endl;
	param_file << "proj_sod = " << params->sod << std::endl;
	param_file << "proj_sdd = " << params->sdd << std::endl;
	//*/
	/*
	param_file << "numX = " << params->numX << std::endl;
	param_file << "numY = " << params->numY << std::endl;
	param_file << "numZ = " << params->numZ << std::endl;
	param_file << "voxelWidth = " << params->voxelWidth << std::endl;
	param_file << "voxelHeight = " << params->voxelHeight << std::endl;
	param_file << "offsetX = " << params->offsetX << std::endl;
	param_file << "offsetY = " << params->offsetY << std::endl;
	param_file << "offsetZ = " << params->offsetZ << std::endl;

	if (params->geometry == parameters::CONE)
		param_file << "geometry = " << "CONE" << std::endl;
	else if (params->geometry == parameters::PARALLEL)
		param_file << "geometry = " << "PARALLEL" << std::endl;
	else if (params->geometry == parameters::FAN)
		param_file << "geometry = " << "FAN" << std::endl;
	else if (params->geometry == parameters::MODULAR)
		param_file << "geometry = " << "MODULAR" << std::endl;
	param_file << "angularRange = " << params->angularRange << std::endl;
	param_file << "numAngles = " << params->numAngles << std::endl;
	param_file << "numRows = " << params->numRows << std::endl;
	param_file << "numCols = " << params->numCols << std::endl;
	param_file << "pixelHeight = " << params->pixelHeight << std::endl;
	param_file << "pixelWidth = " << params->pixelWidth << std::endl;
	param_file << "centerRow = " << params->centerRow << std::endl;
	param_file << "centerCol = " << params->centerCol << std::endl;
	param_file << "phis = " << phis_strs << std::endl;
	param_file << "sod = " << params->sod << std::endl;
	param_file << "sdd = " << params->sdd << std::endl;
	//*/
	param_file.close();

	return true;
}
