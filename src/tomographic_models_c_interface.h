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


/**
 * This header and source file's sole purpose is to provide an ANSI C wrapper around the tomographicModels class.
 * This makes it possible to use with the python ctypes module and other interfaces which do not work with C++ classes
 * such as C#, MATLAB, etc.
 */

extern "C" PROJECTOR_API bool set_model(int);
extern "C" PROJECTOR_API int create_new_model();

extern "C" PROJECTOR_API bool copy_parameters(int);

extern "C" PROJECTOR_API void about();
extern "C" PROJECTOR_API void version(char*);
extern "C" PROJECTOR_API bool print_parameters();
extern "C" PROJECTOR_API bool reset();
extern "C" PROJECTOR_API void set_log_error();
extern "C" PROJECTOR_API void set_log_warning();
extern "C" PROJECTOR_API void set_log_status();
extern "C" PROJECTOR_API void set_log_debug();
extern "C" PROJECTOR_API bool include_cufft();
extern "C" PROJECTOR_API int getOptimalFFTsize(int N);
extern "C" PROJECTOR_API bool set_maxSlicesForChunking(int N);

extern "C" PROJECTOR_API bool all_defined();
extern "C" PROJECTOR_API bool ct_geometry_defined();
extern "C" PROJECTOR_API bool ct_volume_defined();

extern "C" PROJECTOR_API bool verify_input_sizes(int, int, int, int, int, int);

extern "C" PROJECTOR_API bool project_gpu(float* g, float* f);
extern "C" PROJECTOR_API bool project_with_mask_gpu(float* g, float* f, float* mask);
extern "C" PROJECTOR_API bool backproject_gpu(float* g, float* f);

extern "C" PROJECTOR_API bool project_cpu(float* g, float* f);
extern "C" PROJECTOR_API bool project_with_mask_cpu(float* g, float* f, float* mask);
extern "C" PROJECTOR_API bool backproject_cpu(float* g, float* f);

extern "C" PROJECTOR_API bool FBP_cpu(float* g, float* f);
extern "C" PROJECTOR_API bool FBP_gpu(float* g, float* f);

extern "C" PROJECTOR_API bool project(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool project_with_mask(float* g, float* f, float* mask, bool data_on_cpu);
extern "C" PROJECTOR_API bool backproject(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool weightedBackproject(float* g, float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool negLog(float* g, int, int, int, float gray_value);
extern "C" PROJECTOR_API bool expNeg(float* g, int, int, int);

extern "C" PROJECTOR_API bool HilbertFilterProjections(float* g, bool data_on_cpu, float scalar);
extern "C" PROJECTOR_API bool rampFilterProjections(float* g, bool data_on_cpu, float scalar);
extern "C" PROJECTOR_API bool filterProjections(float* g, float* g_out, bool data_on_cpu);
extern "C" PROJECTOR_API bool filterProjections_gpu(float* g);
extern "C" PROJECTOR_API bool filterProjections_cpu(float* g);

extern "C" PROJECTOR_API int extraColumnsForOffsetScan();

extern "C" PROJECTOR_API bool preRampFiltering(float* g, bool data_on_cpu);
extern "C" PROJECTOR_API bool postRampFiltering(float* g, bool data_on_cpu);

extern "C" PROJECTOR_API bool rampFilterVolume(float* f, bool data_on_cpu);
extern "C" PROJECTOR_API float get_FBPscalar();

extern "C" PROJECTOR_API bool FBP(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool inconsistencyReconstruction(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool lambdaTomography(float* g, float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool sensitivity(float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool windowFOV(float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float tiltAngle, float helicalPitch);
extern "C" PROJECTOR_API bool set_fanbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau);
extern "C" PROJECTOR_API bool set_parallelbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis);
extern "C" PROJECTOR_API bool set_modularbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float*, float*, float*, float*);
extern "C" PROJECTOR_API bool set_coneparallel(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float helicalPitch);

extern "C" PROJECTOR_API bool rotate_detector(float alpha);
extern "C" PROJECTOR_API bool shift_detector(float r, float c);

extern "C" PROJECTOR_API bool set_flatDetector();
extern "C" PROJECTOR_API bool set_curvedDetector();
extern "C" PROJECTOR_API int get_detectorType();

extern "C" PROJECTOR_API bool set_numAngles(int);
extern "C" PROJECTOR_API bool set_numCols(int);
extern "C" PROJECTOR_API bool set_numRows(int);

extern "C" PROJECTOR_API bool set_sod(float);
extern "C" PROJECTOR_API bool set_sdd(float);

extern "C" PROJECTOR_API bool set_pixelHeight(float);
extern "C" PROJECTOR_API bool set_pixelWidth(float);

extern "C" PROJECTOR_API bool set_centerCol(float);
extern "C" PROJECTOR_API bool set_centerRow(float);

extern "C" PROJECTOR_API bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool set_volumeDimensionOrder(int which);
extern "C" PROJECTOR_API int get_volumeDimensionOrder();
extern "C" PROJECTOR_API bool set_default_volume(float scale);

extern "C" PROJECTOR_API bool set_numZ(int numZ);
extern "C" PROJECTOR_API bool set_numY(int numY);
extern "C" PROJECTOR_API bool set_numX(int numX);
extern "C" PROJECTOR_API bool set_offsetX(float offsetX);
extern "C" PROJECTOR_API bool set_offsetY(float offsetY);
extern "C" PROJECTOR_API bool set_offsetZ(float offsetZ);

extern "C" PROJECTOR_API bool set_voxelWidth(float W);
extern "C" PROJECTOR_API bool set_voxelHeight(float H);

extern "C" PROJECTOR_API bool set_geometry(int);

extern "C" PROJECTOR_API bool angles_are_defined();
extern "C" PROJECTOR_API bool angles_are_equispaced();

extern "C" PROJECTOR_API bool projectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool projectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool projectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
extern "C" PROJECTOR_API bool backprojectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

extern "C" PROJECTOR_API bool rowRangeNeededForBackprojection(int* rowsNeeded);
extern "C" PROJECTOR_API bool viewRangeNeededForBackprojection(int* viewsNeeded);
extern "C" PROJECTOR_API bool sliceRangeNeededForProjection(int* slicesNeeded, bool doClip);
extern "C" PROJECTOR_API int numRowsRequiredForBackprojectingSlab(int numSlicesPerChunk);

extern "C" PROJECTOR_API int number_of_gpus();
extern "C" PROJECTOR_API int get_gpus(int* list_of_gpus);
extern "C" PROJECTOR_API bool set_GPU(int whichGPU);
extern "C" PROJECTOR_API bool set_GPUs(int* whichGPUs, int N);
extern "C" PROJECTOR_API int get_GPU();
extern "C" PROJECTOR_API bool set_axisOfSymmetry(float axisOfSymmetry);
extern "C" PROJECTOR_API float get_axisOfSymmetry();
extern "C" PROJECTOR_API bool clear_axisOfSymmetry();
extern "C" PROJECTOR_API bool set_projector(int which);
extern "C" PROJECTOR_API int get_projector();
extern "C" PROJECTOR_API bool set_rFOV(float rFOV_in);
extern "C" PROJECTOR_API float get_rFOV();
extern "C" PROJECTOR_API float get_rFOV_min();
extern "C" PROJECTOR_API float get_rFOV_max();
extern "C" PROJECTOR_API bool set_offsetScan(bool);
extern "C" PROJECTOR_API bool get_offsetScan();
extern "C" PROJECTOR_API bool set_truncatedScan(bool);
extern "C" PROJECTOR_API bool set_numTVneighbors(int);
extern "C" PROJECTOR_API int get_numTVneighbors();
extern "C" PROJECTOR_API bool set_rampID(int whichRampFilter);
extern "C" PROJECTOR_API int get_rampID();
extern "C" PROJECTOR_API bool set_FBPlowpass(float W);
extern "C" PROJECTOR_API float get_FBPlowpass();
extern "C" PROJECTOR_API bool set_tau(float tau);
extern "C" PROJECTOR_API bool set_tiltAngle(float tiltAngle);
extern "C" PROJECTOR_API bool set_helicalPitch(float h);
extern "C" PROJECTOR_API bool set_normalizedHelicalPitch(float h_normalized);
extern "C" PROJECTOR_API bool set_attenuationMap(float*);
extern "C" PROJECTOR_API bool set_cylindircalAttenuationMap(float, float);
extern "C" PROJECTOR_API bool convert_conebeam_to_modularbeam();
extern "C" PROJECTOR_API bool convert_parallelbeam_to_modularbeam();
extern "C" PROJECTOR_API bool clear_attenuationMap();
extern "C" PROJECTOR_API bool muSpecified();
extern "C" PROJECTOR_API bool flipAttenuationMapSign(bool data_on_cpu);

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
extern "C" PROJECTOR_API float get_tau();
extern "C" PROJECTOR_API float get_tiltAngle();
extern "C" PROJECTOR_API float get_helicalPitch();
extern "C" PROJECTOR_API float get_normalizedHelicalPitch();
extern "C" PROJECTOR_API float get_z_source_offset();

extern "C" PROJECTOR_API bool get_sourcePositions(float*);
extern "C" PROJECTOR_API bool get_moduleCenters(float*);
extern "C" PROJECTOR_API bool get_rowVectors(float*);
extern "C" PROJECTOR_API bool get_colVectors(float*);

extern "C" PROJECTOR_API bool set_angles(float* phis_in, int numAngles_in);
extern "C" PROJECTOR_API bool get_angles(float*);
extern "C" PROJECTOR_API float get_angularRange();

extern "C" PROJECTOR_API int get_numX();
extern "C" PROJECTOR_API int get_numY();
extern "C" PROJECTOR_API int get_numZ();
extern "C" PROJECTOR_API float get_voxelWidth();
extern "C" PROJECTOR_API float get_voxelHeight();
extern "C" PROJECTOR_API float get_offsetX();
extern "C" PROJECTOR_API float get_offsetY();
extern "C" PROJECTOR_API float get_offsetZ();
extern "C" PROJECTOR_API float get_z0();

extern "C" PROJECTOR_API float find_centerCol(float* g, int iRow, float* searchBounds, bool data_on_cpu);
extern "C" PROJECTOR_API float find_tau(float* g, int iRow, float* searchBounds, bool data_on_cpu);
extern "C" PROJECTOR_API float consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu);
extern "C" PROJECTOR_API float estimate_tilt(float* g, bool data_on_cpu);
extern "C" PROJECTOR_API float conjugate_difference(float* g, float alpha, float centerCol, float* diff, bool data_on_cpu);

extern "C" PROJECTOR_API bool Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu);
extern "C" PROJECTOR_API bool transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, bool data_on_cpu);

extern "C" PROJECTOR_API bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);
extern "C" PROJECTOR_API bool beam_hardening_heel_effect(float* g, float* anode_normal, float* LUT, float* takeOffAngles, int numSamples, int numAngles, float sampleRate, float firstSample, bool data_on_cpu);
extern "C" PROJECTOR_API bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);
extern "C" PROJECTOR_API bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool data_on_cpu);

// Filters for 3D data
extern "C" PROJECTOR_API bool BlurFilter(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool MedianFilter(float* f, int, int, int, float threshold, int w, float signalThreshold, bool data_on_cpu);
extern "C" PROJECTOR_API bool MeanOrVarianceFilter(float* f, int, int, int, int r, int order, bool data_on_cpu);
extern "C" PROJECTOR_API bool HighPassFilter2D(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool BlurFilter2D(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool HighPassFilter(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool MedianFilter2D(float* f, int, int, int, float threshold, int w, float signalThreshold, bool data_on_cpu);
extern "C" PROJECTOR_API bool badPixelCorrection(float* g, int, int, int, float* badPixelMap, int w, bool data_on_cpu);
extern "C" PROJECTOR_API bool BilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu);
extern "C" PROJECTOR_API bool PriorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* prior, bool data_on_cpu);
extern "C" PROJECTOR_API bool GuidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu);
extern "C" PROJECTOR_API bool dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu);

// Anisotropic Total Variation for 3D data
extern "C" PROJECTOR_API float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);
extern "C" PROJECTOR_API bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu);
extern "C" PROJECTOR_API bool TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu);

extern "C" PROJECTOR_API bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip, int oversampling);
extern "C" PROJECTOR_API bool voxelize(float* f, int oversampling);
extern "C" PROJECTOR_API bool clearPhantom();
extern "C" PROJECTOR_API bool scalePhantom(float, float, float);
extern "C" PROJECTOR_API bool rayTrace(float* g, int oversampling, bool data_on_cpu);

extern "C" PROJECTOR_API bool rebin_curved(float* g, float* fanAngles, int order);
extern "C" PROJECTOR_API bool rebin_parallel(float* g, int order);
extern "C" PROJECTOR_API int rebin_parallel_sinogram(float* g, float* output, int order, int desiredRow);

extern "C" PROJECTOR_API bool sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize);

extern "C" PROJECTOR_API bool down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, bool data_on_cpu);
extern "C" PROJECTOR_API bool up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, bool data_on_cpu);

extern "C" PROJECTOR_API bool scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType);

extern "C" PROJECTOR_API bool synthesize_symmetry(float* f_radial, float* f);

extern "C" PROJECTOR_API bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu);

extern "C" PROJECTOR_API bool saveParamsToFile(const char* param_fn);
extern "C" PROJECTOR_API bool save_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight, float pixelWidth, int dtype, float wmin, float wmax);
extern "C" PROJECTOR_API bool read_tif_header(char* fileName, int* shape, float* size, float* slope_and_offset);

extern "C" PROJECTOR_API bool read_tif(char* fileName, float* data);
extern "C" PROJECTOR_API bool read_tif_rows(char* fileName, int firstRow, int lastRow, float* data);
extern "C" PROJECTOR_API bool read_tif_cols(char* fileName, int firstCol, int lastCol, float* data);
extern "C" PROJECTOR_API bool read_tif_roi(char* fileName, int firstRow, int lastRow, int firstCol, int lastCol, float* data);

extern "C" PROJECTOR_API void test_script();
