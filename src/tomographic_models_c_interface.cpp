////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module for ctype binding
////////////////////////////////////////////////////////////////////////////////

#include "tomographic_models_c_interface.h"
#include "tomographic_models.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef PI
#define PI 3.141592653589793
#endif

//#ifdef DEFINE_STATIC_UI
tomographicModels tomo;

bool printParameters()
{
	return tomo.printParameters();
}

bool reset()
{
	return tomo.reset();
}

bool project(float* g, float* f, bool cpu_to_gpu)
{
	return tomo.project(g, f, cpu_to_gpu);
}

bool backproject(float* g, float* f, bool cpu_to_gpu)
{
	return tomo.backproject(g, f, cpu_to_gpu);
}

bool rampFilterProjections(float* g, bool cpu_to_gpu, float scalar)
{
	return tomo.rampFilterProjections(g, cpu_to_gpu, scalar);
}

bool rampFilterVolume(float* f, bool cpu_to_gpu)
{
	return tomo.rampFilterVolume(f, cpu_to_gpu);
}

bool FBP(float* g, float* f, bool cpu_to_gpu)
{
	return tomo.FBP(g, f, cpu_to_gpu);
}

float get_FBPscalar()
{
	return tomo.get_FBPscalar();
}

bool setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd)
{
	return tomo.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
}

bool setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	return tomo.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
}

bool setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	return tomo.setModularBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in);
}

bool setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	return tomo.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool setDefaultVolumeParameters()
{
	return tomo.setDefaultVolumeParameters();
}

bool setVolumeDimensionOrder(int which)
{
	return tomo.setVolumeDimensionOrder(which);
}

int getVolumeDimensionOrder()
{
	return tomo.getVolumeDimensionOrder();
}

bool setGPU(int whichGPU)
{
	return tomo.setGPU(whichGPU);
}

bool setProjector(int which)
{
	return tomo.setProjector(which);
}

bool set_axisOfSymmetry(float axisOfSymmetry)
{
	return tomo.set_axisOfSymmetry(axisOfSymmetry);
}

bool clear_axisOfSymmetry()
{
	return tomo.clear_axisOfSymmetry();
}

bool set_rFOV(float rFOV_in)
{
	return tomo.set_rFOV(rFOV_in);
}

bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	tomographicModels tomo_temp;
	tomo_temp.params.geometry = parameters::CONE;
	tomo_temp.params.detectorType = parameters::FLAT;
	tomo_temp.params.sod = sod;
	tomo_temp.params.sdd = sdd;
	tomo_temp.params.pixelWidth = pixelWidth;
	tomo_temp.params.pixelHeight = pixelHeight;
	tomo_temp.params.numCols = numCols;
	tomo_temp.params.numRows = numRows;
	tomo_temp.params.numAngles = numAngles;
	tomo_temp.params.centerCol = centerCol;
	tomo_temp.params.centerRow = centerRow;
	tomo_temp.params.setAngles(phis, numAngles);

	tomo_temp.params.numX = numX;
	tomo_temp.params.numY = numY;
	tomo_temp.params.numZ = numZ;
	tomo_temp.params.voxelWidth = voxelWidth;
	tomo_temp.params.voxelHeight = voxelHeight;
	tomo_temp.params.offsetX = offsetX;
	tomo_temp.params.offsetY = offsetY;
	tomo_temp.params.offsetZ = offsetZ;

	return tomo_temp.project(g, f, cpu_to_gpu);
}

bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	tomographicModels tomo_temp;
	tomo_temp.params.geometry = parameters::CONE;
	tomo_temp.params.detectorType = parameters::FLAT;
	tomo_temp.params.sod = sod;
	tomo_temp.params.sdd = sdd;
	tomo_temp.params.pixelWidth = pixelWidth;
	tomo_temp.params.pixelHeight = pixelHeight;
	tomo_temp.params.numCols = numCols;
	tomo_temp.params.numRows = numRows;
	tomo_temp.params.numAngles = numAngles;
	tomo_temp.params.centerCol = centerCol;
	tomo_temp.params.centerRow = centerRow;
	tomo_temp.params.setAngles(phis, numAngles);

	tomo_temp.params.numX = numX;
	tomo_temp.params.numY = numY;
	tomo_temp.params.numZ = numZ;
	tomo_temp.params.voxelWidth = voxelWidth;
	tomo_temp.params.voxelHeight = voxelHeight;
	tomo_temp.params.offsetX = offsetX;
	tomo_temp.params.offsetY = offsetY;
	tomo_temp.params.offsetZ = offsetZ;

	return tomo_temp.backproject(g, f, cpu_to_gpu);
}

bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	tomographicModels tomo_temp;
	tomo_temp.params.geometry = parameters::PARALLEL;
	tomo_temp.params.detectorType = parameters::FLAT;
	tomo_temp.params.pixelWidth = pixelWidth;
	tomo_temp.params.pixelHeight = pixelHeight;
	tomo_temp.params.numCols = numCols;
	tomo_temp.params.numRows = numRows;
	tomo_temp.params.numAngles = numAngles;
	tomo_temp.params.centerCol = centerCol;
	tomo_temp.params.centerRow = centerRow;
	tomo_temp.params.setAngles(phis, numAngles);

	tomo_temp.params.numX = numX;
	tomo_temp.params.numY = numY;
	tomo_temp.params.numZ = numZ;
	tomo_temp.params.voxelWidth = voxelWidth;
	tomo_temp.params.voxelHeight = voxelHeight;
	tomo_temp.params.offsetX = offsetX;
	tomo_temp.params.offsetY = offsetY;
	tomo_temp.params.offsetZ = offsetZ;

	return tomo_temp.project(g, f, cpu_to_gpu);
}

bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	tomographicModels tomo_temp;
	tomo_temp.params.geometry = parameters::PARALLEL;
	tomo_temp.params.detectorType = parameters::FLAT;
	tomo_temp.params.pixelWidth = pixelWidth;
	tomo_temp.params.pixelHeight = pixelHeight;
	tomo_temp.params.numCols = numCols;
	tomo_temp.params.numRows = numRows;
	tomo_temp.params.numAngles = numAngles;
	tomo_temp.params.centerCol = centerCol;
	tomo_temp.params.centerRow = centerRow;
	tomo_temp.params.setAngles(phis, numAngles);

	tomo_temp.params.numX = numX;
	tomo_temp.params.numY = numY;
	tomo_temp.params.numZ = numZ;
	tomo_temp.params.voxelWidth = voxelWidth;
	tomo_temp.params.voxelHeight = voxelHeight;
	tomo_temp.params.offsetX = offsetX;
	tomo_temp.params.offsetY = offsetY;
	tomo_temp.params.offsetZ = offsetZ;

	return tomo_temp.backproject(g, f, cpu_to_gpu);
}

int get_numAngles()
{
	return tomo.get_numAngles();
}

int get_numRows()
{
	return tomo.get_numRows();
}

int get_numCols()
{
	return tomo.get_numCols();
}

int get_numX()
{
	return tomo.get_numX();
}

int get_numY()
{
	return tomo.get_numY();
}

int get_numZ()
{
	return tomo.get_numZ();
}

bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool cpu_to_gpu)
{
	return tomo.BlurFilter(f, N_1, N_2, N_3, FWHM, cpu_to_gpu);
}

bool MedianFilter(float* f, int N_1, int N_2, int N_3, float threshold, bool cpu_to_gpu)
{
	return tomo.MedianFilter(f, N_1, N_2, N_3, threshold, cpu_to_gpu);
}

float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return tomo.TVcost(f, N_1, N_2, N_3, delta, beta, cpu_to_gpu);
}

bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return tomo.TVgradient(f, Df, N_1, N_2, N_3, delta, beta, cpu_to_gpu);
}

float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu)
{
	return tomo.TVquadForm(f, d, N_1, N_2, N_3, delta, beta, cpu_to_gpu);
}

bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu)
{
	return tomo.Diffuse(f, N_1, N_2, N_3, delta, numIter, cpu_to_gpu);
}
