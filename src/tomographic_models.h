#pragma once

#include <stdlib.h>
#include "parameters.h"

class tomographicModels
{
public:
	tomographicModels();
	~tomographicModels();

	bool project_gpu(float* g, float* f);
	bool backproject_gpu(float* g, float* f);

	bool project_cpu(float* g, float* f);
	bool backproject_cpu(float* g, float* f);

	bool project(float* g, float* f, bool cpu_to_gpu);
	bool backproject(float* g, float* f, bool cpu_to_gpu);

	bool project(float* g, float* f, parameters* ctParams, bool cpu_to_gpu);
	bool backproject(float* g, float* f, parameters* ctParams, bool cpu_to_gpu);

	bool rampFilterProjections(float* g, parameters* ctParams, bool cpu_to_gpu, float scalar);
	bool rampFilterProjections(float* g, bool cpu_to_gpu, float scalar);
	bool rampFilterVolume(float* f, bool cpu_to_gpu);

	bool FBP(float* g, float* f, bool cpu_to_gpu);
	bool FBP(float* g, float* f, parameters* ctParams, bool cpu_to_gpu);

	bool project_multiGPU(float* g, float* f);
	bool backproject_multiGPU(float* g, float* f);
	bool FBP_multiGPU(float* g, float* f);

	bool printParameters();

	bool setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd);
	bool setFanBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd);
	bool setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis);
	bool setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float*, float*, float*, float*);
	bool setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool setVolumeDimensionOrder(int which);
	int getVolumeDimensionOrder();
	bool setDefaultVolumeParameters(float scale = 1.0);

	bool projectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool setGPU(int whichGPU);
	bool setGPUs(int* whichGPUs, int N);
	int getGPU();
	bool set_axisOfSymmetry(float axisOfSymmetry);
	bool clear_axisOfSymmetry();
	bool setProjector(int which);
	bool set_rFOV(float rFOV_in);
	bool set_rampID(int);
	bool reset();

	int get_numAngles();
	int get_numRows();
	int get_numCols();
	float get_FBPscalar();

	int get_numX();
	int get_numY();
	int get_numZ();

	// Filters for 3D data
	bool BlurFilter(float* f, int, int, int, float FWHM, bool cpu_to_gpu);
	bool MedianFilter(float* f, int, int, int, float threshold, bool cpu_to_gpu);

	// Anisotropic Total Variation for 3D data
	float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
	bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
	float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
	bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu);
	
	parameters params;
private:
	bool backproject_FBP_multiGPU(float* g, float* f, bool doFBP);
	float* copyRows(float*, int, int);
	bool combineRows(float*, float*, int, int);
};
