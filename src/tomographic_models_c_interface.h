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

#include <cstdint>

/**
 * This header and source file's sole purpose is to provide an ANSI C wrapper around the tomographicModels class.
 * This makes it possible to use with the python ctypes module and other interfaces which do not work with C++ classes
 * such as C#, MATLAB, etc.
 */

extern "C" PROJECTOR_API bool set_model(int);
extern "C" PROJECTOR_API int create_new_model();

extern "C" PROJECTOR_API bool copy_parameters(int, bool);
extern "C" PROJECTOR_API bool copy_volume_parameters(int);

extern "C" PROJECTOR_API float* allocate_3D_array(int, int, int, bool);
extern "C" PROJECTOR_API bool free_3D_array(float*, bool);

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
extern "C" PROJECTOR_API int get_maxSlicesForChunking();

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
extern "C" PROJECTOR_API bool weightedBackproject(float* g, float* f, bool doDBP, bool data_on_cpu);

extern "C" PROJECTOR_API bool fmad(float* g, int N_1, int N_2, int N_3, float* scale, float* shift, int M_1, int M_2, int M_3, float clip_low, float clip_high);
extern "C" PROJECTOR_API bool multiply(float* out, float* y, float* x, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API bool scalar_add(float* out, float* y, float a, float* x, int N_1, int N_2, int N_3, bool do_clip);
extern "C" PROJECTOR_API bool negLog(float* g, int, int, int, float gray_value, float clip_low, float clip_high);
extern "C" PROJECTOR_API bool expNeg(float* g, int, int, int, float gray_value);

extern "C" PROJECTOR_API bool HilbertFilterProjections(float* g, bool data_on_cpu, float scalar, float sampleShift);
extern "C" PROJECTOR_API bool rampFilterProjections(float* g, bool data_on_cpu, float scalar);
extern "C" PROJECTOR_API bool filterProjections(float* g, float* g_out, bool inconsistency, bool data_on_cpu);
extern "C" PROJECTOR_API bool filterProjections_gpu(float* g, bool inconsistency);
extern "C" PROJECTOR_API bool filterProjections_cpu(float* g, float* g_out, bool inconsistency);
extern "C" PROJECTOR_API bool DBP_filter(float* g, bool data_on_cpu);
extern "C" PROJECTOR_API bool DBP_filter_cpu(float* g);

extern "C" PROJECTOR_API int extraColumnsForOffsetScan();
extern "C" PROJECTOR_API bool get_offsetScan_weights(float*);
extern "C" PROJECTOR_API bool apply_projection_weights(float* g, float* w, int expNeg_or_negLog, bool data_on_cpu);

extern "C" PROJECTOR_API bool preRampFiltering(float* g, bool data_on_cpu);
extern "C" PROJECTOR_API bool postRampFiltering(float* g, bool data_on_cpu);

extern "C" PROJECTOR_API bool rampFilterVolume(float* f, bool data_on_cpu);
extern "C" PROJECTOR_API float get_FBPscalar();

extern "C" PROJECTOR_API bool FBP(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool DBP(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool inconsistencyReconstruction(float* g, float* f, bool data_on_cpu);
extern "C" PROJECTOR_API bool lambdaTomography(float* g, float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool sensitivity(float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool windowFOV(float* f, bool data_on_cpu);

extern "C" PROJECTOR_API bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau, float tiltAngle, float pitchAngle, float helicalPitch);
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
extern "C" PROJECTOR_API float default_voxelWidth();

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
extern "C" PROJECTOR_API bool sliceRangeNeededForProjection(int* slicesNeeded, bool doClip, bool restrictToVolume);
extern "C" PROJECTOR_API int numRowsRequiredForBackprojectingSlab(int numSlicesPerChunk);

extern "C" PROJECTOR_API bool set_max_cpu_threads(int);
extern "C" PROJECTOR_API bool set_max_gpu_memory(float);
extern "C" PROJECTOR_API bool get_physically_shared_memory();
extern "C" PROJECTOR_API float get_available_system_memory();
extern "C" PROJECTOR_API int number_of_gpus();
extern "C" PROJECTOR_API int get_gpus(int* list_of_gpus);
extern "C" PROJECTOR_API bool set_GPU(int whichGPU);
extern "C" PROJECTOR_API bool set_GPUs(int* whichGPUs, int N);
extern "C" PROJECTOR_API int get_GPU();
extern "C" PROJECTOR_API float get_available_gpu_memory(int);
extern "C" PROJECTOR_API bool set_axisOfSymmetry(float axisOfSymmetry);
extern "C" PROJECTOR_API float get_axisOfSymmetry();
extern "C" PROJECTOR_API bool clear_axisOfSymmetry();
extern "C" PROJECTOR_API bool set_projector(int which);
extern "C" PROJECTOR_API int get_projector();
extern "C" PROJECTOR_API bool set_rFOV(float rFOV_in);
extern "C" PROJECTOR_API float get_rFOV(bool get_default_value);
extern "C" PROJECTOR_API float get_rFOV_min();
extern "C" PROJECTOR_API float get_rFOV_max();
extern "C" PROJECTOR_API float get_zFOV_min();
extern "C" PROJECTOR_API float get_zFOV_max();
extern "C" PROJECTOR_API bool set_offsetScan(bool);
extern "C" PROJECTOR_API bool get_offsetScan();
extern "C" PROJECTOR_API bool set_truncatedScan(bool);
extern "C" PROJECTOR_API bool get_truncatedScan();
extern "C" PROJECTOR_API bool set_cornerPatching(bool);
extern "C" PROJECTOR_API bool set_clipWeightedBackprojection(bool);
extern "C" PROJECTOR_API bool get_clipWeightedBackprojection();

extern "C" PROJECTOR_API bool set_numRowsExtrapolate(int N);

extern "C" PROJECTOR_API bool set_numTVneighbors(int);
extern "C" PROJECTOR_API int get_numTVneighbors();
extern "C" PROJECTOR_API bool set_rampID(int whichRampFilter);
extern "C" PROJECTOR_API int get_rampID();
extern "C" PROJECTOR_API bool set_helicalFilterParameter(float);
extern "C" PROJECTOR_API bool set_FBPlowpass(float W);
extern "C" PROJECTOR_API float get_FBPlowpass();
extern "C" PROJECTOR_API bool set_tau(float tau);
extern "C" PROJECTOR_API bool set_tiltAngle(float tiltAngle);
extern "C" PROJECTOR_API bool set_pitchAngle(float pitchAngle);
extern "C" PROJECTOR_API bool set_helicalPitch(float h);
extern "C" PROJECTOR_API bool set_normalizedHelicalPitch(float h_normalized);
extern "C" PROJECTOR_API bool set_source_size(float height, float width);
extern "C" PROJECTOR_API bool get_source_size(float* height_and_width);
extern "C" PROJECTOR_API bool set_helicalFBPWeight(float q);
extern "C" PROJECTOR_API bool set_DBPparameter(float);
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
extern "C" PROJECTOR_API float get_pitchAngle();
extern "C" PROJECTOR_API float get_helicalPitch();
extern "C" PROJECTOR_API float get_normalizedHelicalPitch();
extern "C" PROJECTOR_API float get_helicalFBPWeight();
extern "C" PROJECTOR_API float get_z_source_offset();
extern "C" PROJECTOR_API bool set_z_source_offset(float z_offs);

extern "C" PROJECTOR_API bool get_sourcePositions(float*);
extern "C" PROJECTOR_API bool get_moduleCenters(float*);
extern "C" PROJECTOR_API bool get_rowVectors(float*);
extern "C" PROJECTOR_API bool get_colVectors(float*);

extern "C" PROJECTOR_API bool set_angles(float* phis_in, int numAngles_in);
extern "C" PROJECTOR_API bool get_angles(float*);
extern "C" PROJECTOR_API float get_angularRange(bool get_sign);

extern "C" PROJECTOR_API int get_numX();
extern "C" PROJECTOR_API int get_numY();
extern "C" PROJECTOR_API int get_numZ();
extern "C" PROJECTOR_API float get_voxelWidth();
extern "C" PROJECTOR_API float get_voxelHeight();
extern "C" PROJECTOR_API float get_offsetX();
extern "C" PROJECTOR_API float get_offsetY();
extern "C" PROJECTOR_API float get_offsetZ();
extern "C" PROJECTOR_API float get_z0();
extern "C" PROJECTOR_API float get_y0();
extern "C" PROJECTOR_API float get_x0();

extern "C" PROJECTOR_API float find_centerCol(float* g, int iRow, float* searchBounds, bool data_on_cpu);
extern "C" PROJECTOR_API float find_tau(float* g, int iRow, float* searchBounds, bool data_on_cpu);
extern "C" PROJECTOR_API float consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu);
extern "C" PROJECTOR_API float estimate_tilt(float* g, bool data_on_cpu);
extern "C" PROJECTOR_API float conjugate_difference(float* g, float alpha, float centerCol, float* diff, bool data_on_cpu);
extern "C" PROJECTOR_API bool inconsistency_sweep(float* g, float* shifts, int numShifts, float* tilts, int numTilts, int which_param, float* costValues, bool data_on_cpu);
extern "C" PROJECTOR_API bool Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu);
extern "C" PROJECTOR_API bool ring_removal(float* g, float delta, float beta, int numIter, float maxChange, int angle_downsampling_factor);
extern "C" PROJECTOR_API bool transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, float FWHM, bool data_on_cpu);

extern "C" PROJECTOR_API bool apply_polynomial_bhc(float* g, int N_1, int N_2, int N_3, float* coeff, int N_coeff, bool data_on_cpu);
extern "C" PROJECTOR_API bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);
extern "C" PROJECTOR_API bool beam_hardening_heel_effect(float* g, float* anode_normal, float* LUT, float* takeOffAngles, int numSamples, int numAngles, float sampleRate, float firstSample, bool data_on_cpu);
extern "C" PROJECTOR_API bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool scalar_LUT, bool data_on_cpu);
extern "C" PROJECTOR_API bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool constrain, bool data_on_cpu);

extern "C" PROJECTOR_API bool applyThreeMaterialBHC(float* sum, float* w_1, float* w_2, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);

// Filters for 3D data
extern "C" PROJECTOR_API bool BlurFilter(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool MedianFilter(float* f, int, int, int, float threshold, int w, float signalThreshold, bool data_on_cpu);
extern "C" PROJECTOR_API bool MeanOrVarianceFilter(float* f, int, int, int, int r, int order, bool data_on_cpu);
extern "C" PROJECTOR_API bool HighPassFilter2D(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool BlurFilter2D(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool HighPassFilter(float* f, int, int, int, float FWHM, bool data_on_cpu);
extern "C" PROJECTOR_API bool MedianFilter2D(float* f, int, int, int, float threshold, int w, float signalThreshold, bool data_on_cpu);
extern "C" PROJECTOR_API bool BlurFilter1D(float* f, int, int, int, float FWHM, int axis, bool isPeriodic, bool data_on_cpu);
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
extern "C" PROJECTOR_API bool TV_fast(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool data_on_cpu);

extern "C" PROJECTOR_API bool addMesh(float* triangles, int numTriangles, float val, const char* chemForm);
extern "C" PROJECTOR_API bool addObject(float* f, int type, float* c, float* r, float val, float* A, float* clip, const char* chemForm, int oversampling);
extern "C" PROJECTOR_API bool voxelize(float* f, int oversampling);
extern "C" PROJECTOR_API bool voxelizeMesh(float* f, float val, int oversampling);
extern "C" PROJECTOR_API bool clearPhantom();
extern "C" PROJECTOR_API bool scalePhantom(float, float, float);
extern "C" PROJECTOR_API bool shiftPhantom(float, float, float);
extern "C" PROJECTOR_API bool rayTrace(float* g, int oversampling, bool data_on_cpu);
extern "C" PROJECTOR_API bool rayTrace_polychromatic(float* g, float* spectralResponse, float* energies, int N_energies, int oversampling, bool data_on_cpu);
extern "C" PROJECTOR_API bool rayTraceMesh(float* g, int oversampling, bool data_on_cpu);
extern "C" PROJECTOR_API bool rayTraceMesh_polychromatic(float* g, float* spectralResponse, float* energies, int N_energies, int oversampling, bool data_on_cpu);
extern "C" PROJECTOR_API bool double_cone(float* f, int N_1, int N_2, int N_3, float beta, float minimum_radius);
extern "C" PROJECTOR_API bool patch_corners(float* f, float* f_top, int numZ_top, float* f_bot, int numZ_bot, int window_width);

extern "C" PROJECTOR_API bool rebin_curved(float* g, float* fanAngles, int order);
extern "C" PROJECTOR_API bool rebin_parallel(float* g, int order);
extern "C" PROJECTOR_API int rebin_parallel_sinogram(float* g, float* output, int order, int desiredRow);

extern "C" PROJECTOR_API bool sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize, int padSide);
extern "C" PROJECTOR_API bool sinogram_replacement_interp(float* g, int* windowSize, int padSide);

extern "C" PROJECTOR_API bool down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, int order, float* offset, float maxWidth, bool data_on_cpu);
extern "C" PROJECTOR_API bool up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, int order, int set_type, bool data_on_cpu);

extern "C" PROJECTOR_API bool resample_projection_angles(float* g, float* g_new, float* phis_new, int N_phis_new);

extern "C" PROJECTOR_API bool antialiasFilter(float* volume, int* N, float L, int order, float* h, int N_h, bool* axis, bool data_on_cpu);
extern "C" PROJECTOR_API bool finiteDifference(float* volume, int* N, int order, int shift, bool* axis, float scalar, bool data_on_cpu);

extern "C" PROJECTOR_API bool scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType);
extern "C" PROJECTOR_API bool scatter_simulation(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float reference_energy, const char** chemForms, int num_materials, float* densities, float* b_L, float* b_H, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters);
extern "C" PROJECTOR_API bool detector_scatter_simulation(
	float* events, float thickness, float mass_density, float* source, float* energies, int N_energies, const char* chemForm, int num_photons, int max_scatters, float* direction);

extern "C" PROJECTOR_API bool polychromatic_attenuation(float* spectralResponse, float* gammas, float referenceEnergy, float* g_1, float* sigma_1, float* g_2, float* sigma_2, float* g_3, float* sigma_3, float* g_poly, int N_gamma);

extern "C" PROJECTOR_API bool synthesize_symmetry(float* f_radial, float* f);

extern "C" PROJECTOR_API bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu);

extern "C" PROJECTOR_API bool divide(float* I, float* J, int N_1, int N_2, int N_3, float divide_by_zero_value, bool skip_zero_denominator, bool data_on_cpu);
extern "C" PROJECTOR_API bool reciprocal(float* I, int N_1, int N_2, int N_3, float divide_by_zero_value, bool data_on_cpu);
extern "C" PROJECTOR_API bool has_nan_or_inf(float* I, int N_1, int N_2, int N_3);
extern "C" PROJECTOR_API bool replaceNAN(float* I, int N_1, int N_2, int N_3, float newValue);
extern "C" PROJECTOR_API bool boundingBox(float* I, int N_1, int N_2, int N_3, int boundary_type, int* AABB);
extern "C" PROJECTOR_API bool heaviside(float* I, int N_1, int N_2, int N_3, float scale, float shift);
extern "C" PROJECTOR_API bool dirac(float* I, int N_1, int N_2, int N_3, float scale, float shift);
extern "C" PROJECTOR_API bool quantize(float* I, int, int, int, float* targets, int numTargets);
extern "C" PROJECTOR_API bool inpaint(float* I, int, int, int);
extern "C" PROJECTOR_API bool threshold(float* I, int, int, int, float value, bool greater_than, int fill_type, int num_pixel_dilate);
extern "C" PROJECTOR_API bool region_growing(float* I, int, int, int, float startThreshold, float endThreshold, int fill_type, int num_pixel_dilate);
extern "C" PROJECTOR_API bool dilate(float* I, int, int, int, int pixelRadius, int fill_type);
extern "C" PROJECTOR_API bool k_means(float* I, int, int, int, float*, int);
extern "C" PROJECTOR_API bool Otsu_thresholds(float* I, int, int, int, float*, int);
extern "C" PROJECTOR_API bool extrema(float* I, int, int, int, float*);
extern "C" PROJECTOR_API bool basic_stats(float* I, int, int, int, float*);
extern "C" PROJECTOR_API bool percentile2D(float* I, int N_images, int N, float q, float* percentiles);

extern "C" PROJECTOR_API float histogram3D(float* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);
extern "C" PROJECTOR_API float histogram3D_uint8(uint8_t* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);
extern "C" PROJECTOR_API float histogram3D_int16(int16_t* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);
extern "C" PROJECTOR_API float histogram3D_uint16(uint16_t* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);
extern "C" PROJECTOR_API float histogram3D_int32(int32_t* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);
extern "C" PROJECTOR_API float histogram3D_uint32(uint32_t* I, int, int, int, float*, float*, int, float rangeMin, float rangeMax);

extern "C" PROJECTOR_API bool covariance(float* I, int, int, int, float, float*, float*);
extern "C" PROJECTOR_API bool center_of_mass(float* I, int, int, int, float, float*);
extern "C" PROJECTOR_API bool valueMask(float* I, int N_1, int N_2, int N_3, float a, float b, float c, float d, bool deriv);
extern "C" PROJECTOR_API bool sum_first_axis(float* I, int N_1, int N_2, int N_3, float* sums);
extern "C" PROJECTOR_API bool sum_axis(float* I, int N_1, int N_2, int N_3, float* sums, int axis);

extern "C" PROJECTOR_API bool saveParamsToFile(const char* param_fn);
extern "C" PROJECTOR_API bool save_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight, float pixelWidth, int dtype, float wmin, float wmax);
extern "C" PROJECTOR_API bool read_tif_header(char* fileName, int* shape, float* size, float* slope_and_offset);

extern "C" PROJECTOR_API bool read_tif(char* fileName, float* data);
extern "C" PROJECTOR_API bool read_tif_rows(char* fileName, int firstRow, int lastRow, float* data);
extern "C" PROJECTOR_API bool read_tif_cols(char* fileName, int firstCol, int lastCol, float* data);
extern "C" PROJECTOR_API bool read_tif_roi(char* fileName, int firstRow, int lastRow, int firstCol, int lastCol, float* data);


extern "C" PROJECTOR_API void test_script();
