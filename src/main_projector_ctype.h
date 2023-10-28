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

extern "C" PROJECTOR_API bool project(float* g, float* f, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool backproject(float* g, float* f, bool cpu_to_gpu);

extern "C" PROJECTOR_API bool rampFilterProjections(float* g, bool cpu_to_gpu);
extern "C" PROJECTOR_API bool rampFilterVolume(float* f, bool cpu_to_gpu);

extern "C" PROJECTOR_API bool printParameters();

extern "C" PROJECTOR_API bool setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd);
extern "C" PROJECTOR_API bool setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis);
extern "C" PROJECTOR_API bool setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float*, float*, float*, float*);
extern "C" PROJECTOR_API bool setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool setVolumeDimensionOrder(int which);
extern "C" PROJECTOR_API int getVolumeDimensionOrder();
extern "C" PROJECTOR_API bool setDefaultVolumeParameters();

extern "C" PROJECTOR_API bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool setGPU(int whichGPU);
extern "C" PROJECTOR_API bool set_axisOfSymmetry(float axisOfSymmetry);
extern "C" PROJECTOR_API bool setProjector(int which);
extern "C" PROJECTOR_API bool set_rFOV(float rFOV_in);
extern "C" PROJECTOR_API bool reset();

extern "C" PROJECTOR_API int get_numAngles();
extern "C" PROJECTOR_API int get_numRows();
extern "C" PROJECTOR_API int get_numCols();
extern "C" PROJECTOR_API float get_FBPscalar();

extern "C" PROJECTOR_API int get_numX();
extern "C" PROJECTOR_API int get_numY();
extern "C" PROJECTOR_API int get_numZ();

extern "C" PROJECTOR_API bool BlurFilter(float* f, int, int, int, float FWHM);
extern "C" PROJECTOR_API bool MedianFilter(float* f, int, int, int, float threshold);

extern "C" PROJECTOR_API float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta);
extern "C" PROJECTOR_API bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
extern "C" PROJECTOR_API float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta);
