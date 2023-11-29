////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ header for ctype binding
////////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
    #pragma once

    #ifdef PROJECTOR_EXPORTS
        #define PROJECTOR_API __declspec(dllexport)
    #else
        #define PROJECTOR_API __declspec(dllimport)
    #endif
#else
    #define PROJECTOR_API
#endif

extern "C" PROJECTOR_API bool printParameters();
extern "C" PROJECTOR_API bool reset();

extern "C" PROJECTOR_API bool project_gpu(float* g, float* f);
extern "C" PROJECTOR_API bool backproject_gpu(float* g, float* f);

extern "C" PROJECTOR_API bool project_cpu(float* g, float* f);
extern "C" PROJECTOR_API bool backproject_cpu(float* g, float* f);

extern "C" PROJECTOR_API bool project(float* g, float* f, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool backproject(float* g, float* f, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool weightedBackproject(float* g, float* f, bool cpu_to_gpu);

extern "C" PROJECTOR_API bool HilbertFilterProjections(float* g, bool cpu_to_gpu, float scalar);
extern "C" PROJECTOR_API bool rampFilterProjections(float* g, bool cpu_to_gpu, float scalar);
extern "C" PROJECTOR_API bool filterProjections(float* g, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool rampFilterVolume(float* f, bool cpu_to_gpu);
extern "C" PROJECTOR_API float get_FBPscalar();

extern "C" PROJECTOR_API bool FBP(float* g, float* f, bool cpu_to_gpu);

extern "C" PROJECTOR_API bool sensitivity(float* f, bool cpu_to_gpu);

extern "C" PROJECTOR_API bool set_coneBeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd);
extern "C" PROJECTOR_API bool set_fanBeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd);
extern "C" PROJECTOR_API bool set_parallelBeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis);
extern "C" PROJECTOR_API bool set_modularBeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float*, float*, float*, float*);
extern "C" PROJECTOR_API bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool set_volumeDimensionOrder(int which);
extern "C" PROJECTOR_API int get_volumeDimensionOrder();
extern "C" PROJECTOR_API bool set_defaultVolume(float scale);

extern "C" PROJECTOR_API bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool projectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectFanBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool set_GPU(int whichGPU);
extern "C" PROJECTOR_API bool set_GPUs(int* whichGPUs, int N);
extern "C" PROJECTOR_API int get_GPU();
extern "C" PROJECTOR_API bool set_axisOfSymmetry(float axisOfSymmetry);
extern "C" PROJECTOR_API bool clear_axisOfSymmetry();
extern "C" PROJECTOR_API bool set_projector(int which);
extern "C" PROJECTOR_API bool set_rFOV(float rFOV_in);
extern "C" PROJECTOR_API bool set_rampID(int whichRampFilter);
extern "C" PROJECTOR_API bool set_helicalPitch(float h);
extern "C" PROJECTOR_API bool set_attenuationMap(float*);
extern "C" PROJECTOR_API bool set_cylindircalAttenuationMap(float, float);
extern "C" PROJECTOR_API bool clear_attenuationMap();

extern "C" PROJECTOR_API int get_geometry();
extern "C" PROJECTOR_API float get_sod();
extern "C" PROJECTOR_API float get_sdd();
extern "C" PROJECTOR_API int get_numAngles();
extern "C" PROJECTOR_API int get_numRows();
extern "C" PROJECTOR_API int get_numCols();
extern "C" PROJECTOR_API float get_pixelWidth();
extern "C" PROJECTOR_API float get_pixelHeight();
extern "C" PROJECTOR_API float get_centerRow();
extern "C" PROJECTOR_API float get_centerCol();

extern "C" PROJECTOR_API bool get_sourcePositions(float*);
extern "C" PROJECTOR_API bool get_moduleCenters(float*);
extern "C" PROJECTOR_API bool get_rowVectors(float*);
extern "C" PROJECTOR_API bool get_colVectors(float*);

extern "C" PROJECTOR_API bool set_angles(float* phis_in, int numAngles_in);
extern "C" PROJECTOR_API bool get_angles(float*);

extern "C" PROJECTOR_API int get_numX();
extern "C" PROJECTOR_API int get_numY();
extern "C" PROJECTOR_API int get_numZ();
extern "C" PROJECTOR_API float get_voxelWidth();
extern "C" PROJECTOR_API float get_voxelHeight();
extern "C" PROJECTOR_API float get_offsetX();
extern "C" PROJECTOR_API float get_offsetY();
extern "C" PROJECTOR_API float get_offsetZ();

// Filters for 3D data
extern "C" PROJECTOR_API bool BlurFilter(float* f, int, int, int, float FWHM, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool MedianFilter(float* f, int, int, int, float threshold, bool cpu_to_gpu);

// Anisotropic Total Variation for 3D data
extern "C" PROJECTOR_API float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
extern "C" PROJECTOR_API float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool cpu_to_gpu);
