////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// header for LEAP API
////////////////////////////////////////////////////////////////////////////////
#ifndef __TOMOGRAPHIC_MODELS_H
#define __TOMOGRAPHIC_MODELS_H

#ifdef WIN32
#pragma once
#endif

#define LEAP_VERSION "1.0"

#include <stdlib.h>
#include "parameters.h"
#include "projectors.h"
#include "filtered_backprojection.h"
#include "cpu_utils.h"

/**
 *  tomographicModels class
 * This is the main interface for LEAP.  All calls to forward project, backproject, FBP, filtering, noise filters come through this class.
 * There are no dependencies on torch from here and all layers beneath.  These torch dependencies are stripped in "main_projectors.cpp".
 * The main job of this class is to set/get parameters, do error checks, and dispath jobs.  It contains almost no algorithm logic.
 * In addition to the jobs listed above, this class is also responsible for divide jobs across multiple GPUs or dividing up GPU jobs so that
 * they fit into the available GPU memory.  Functions called from this class are either CPU based or single GPU based.
 */

class tomographicModels
{
public:
	tomographicModels();
	~tomographicModels();

	/**
	 * \fn          reset
	 * \brief       resets (clears) all CT geometry and CT volume parameter values
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool reset();

	/**
	 * \fn          project_gpu
	 * \brief       performs a forward projection on GPU
 	 * \param[in]   g pointer to the projection data (output) on the GPU
	 * \param[in]   f pointer to the volume data (input) on the GPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool project_gpu(float* g, float* f);

	/**
	 * \fn          backproject_gpu
	 * \brief       performs a backprojection on GPU
	 * \param[in]   g pointer to the projection data (input) on the GPU
	 * \param[in]   f pointer to the volume data (output) on the GPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool backproject_gpu(float* g, float* f);

	/**
	 * \fn          project_cpu
	 * \brief       performs a forward projection on CPU
	 * \param[in]   g pointer to the projection data (output) on the CPU
	 * \param[in]   f pointer to the volume data (input) on the CPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool project_cpu(float* g, float* f);

	/**
	 * \fn          backproject_cpu
	 * \brief       performs a backprojection on CPU
	 * \param[in]   g pointer to the projection data (input) on the CPU
	 * \param[in]   f pointer to the volume data (output) on the CPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool backproject_cpu(float* g, float* f);

	/**
	 * \fn          project
	 * \brief       performs a forward projection
	 * \param[in]   g pointer to the projection data (output)
	 * \param[in]   f pointer to the volume data (input)
	 * \param[in]   data_on_cpu true if data (f and g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool project(float* g, float* f, bool data_on_cpu);

	/**
	 * \fn          backproject
	 * \brief       performs a backprojection
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \param[in]   data_on_cpu true if data (f and g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool backproject(float* g, float* f, bool data_on_cpu);

	/**
	 * \fn          weightedBackproject
	 * \brief       performs a weighted backprojection (for an FBP algorithm)
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \param[in]   data_on_cpu true if data (f and g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool weightedBackproject(float* g, float* f, bool data_on_cpu);

	/**
	 * \fn          FBP_cpu
	 * \brief       performs an FBP reconstruction on CPU
	 * \param[in]   g pointer to the projection data (input) on the CPU
	 * \param[in]   f pointer to the volume data (output) on the CPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool FBP_cpu(float* g, float* f);

	/**
	 * \fn          FBP_gpu
	 * \brief       performs an FBP reconstruction on GPU
	 * \param[in]   g pointer to the projection data (input) on the GPU
	 * \param[in]   f pointer to the volume data (output) on the GPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool FBP_gpu(float* g, float* f);

	/**
	 * \fn          doFBP
	 * \brief       performs an FBP reconstruction
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \param[in]   data_on_cpu true if data (f and g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool doFBP(float* g, float* f, bool data_on_cpu);

	/**
	 * \fn          sensitivity
	 * \brief       calculates the sensitivity, i.e., backprojection of ones
	 * \param[in]   f pointer to the volume data (output)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool sensitivity(float* f, bool data_on_cpu);

	/**
	 * \fn          HilbertFilterProjections
	 * \brief       applies a Hilbert filter to each row and projection angle
	 * \param[in]   g pointer to the projection data (input and output)
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \param[in]   scalar optional scalar to multiply the result by
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool HilbertFilterProjections(float* g, bool data_on_cpu, float scalar = 1.0);

	/**
	 * \fn          rampFilterProjections
	 * \brief       applies a ramp filter to each row and projection angle
	 * \param[in]   g pointer to the projection data (input and output)
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \param[in]   scalar optional scalar to multiply the result by
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool rampFilterProjections(float* g, bool data_on_cpu, float scalar = 1.0);

	/**
	 * \fn          filterProjections
	 * \brief       applies the necessary filters and ray/view weights necessary for FBP reconstruction
	 * \param[in]   g pointer to the projection data (input and output)
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool filterProjections(float* g, bool data_on_cpu);

	/**
	 * \fn          rampFilterVolume
	 * \brief       applies a 2D ramp filter to each z-slice
	 * \param[in]   f pointer to the volume data (input and output)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool rampFilterVolume(float* f, bool data_on_cpu);

	/**
	 * \fn          windowFOV
	 * \brief       sets the array to zero for those values outside the field of view
	 * \param[in]   f pointer to the volume data (input and output)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool windowFOV(float* f, bool data_on_cpu);

	/**
	 * \fn          print_parameters
	 * \brief       prints the CT geometry and CT volume parameters to the screen
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool print_parameters();

	/**
	 * \fn          about
	 * \brief       prints info about LEAP, including the version number
	 */
	const char* about();

	/**
	 * \fn          set_conebeam
	 * \brief       sets the cone-beam parameters
	 * \param[in]   numAngles number of projection angles
	 * \param[in]   numRows number of rows in the x-ray detector
	 * \param[in]   numCols number of columns in the x-ray detector
	 * \param[in]   pixelHeight the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
	 * \param[in]   pixelWidth the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
	 * \param[in]   centerRow the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   centerCol the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   phis pointer to an array for specifying the angles of each projection, measured in degrees
	 * \param[in]   sod source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
	 * \param[in]   sdd source to detector distance, measured in mm
	 * \param[in]   tau the center of rotation horizontal translation (mm)
	 * \param[in]   helicalPitch the helical pitch (mm/radians)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau = 0.0, float helicalPitch = 0.0);

	/**
	 * \fn          set_fanbeam
	 * \brief       sets the fan-beam parameters
	 * \param[in]   numAngles number of projection angles
	 * \param[in]   numRows number of rows in the x-ray detector
	 * \param[in]   numCols number of columns in the x-ray detector
	 * \param[in]   pixelHeight the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
	 * \param[in]   pixelWidth the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
	 * \param[in]   centerRow the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   centerCol the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   phis pointer to an array for specifying the angles of each projection, measured in degrees
	 * \param[in]   sod source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
	 * \param[in]   sdd source to detector distance, measured in mm
	 * \param[in]   tau the center of rotation horizontal translation (mm)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_fanbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau = 0.0);

	/**
	 * \fn          set_parallelbeam
	 * \brief       sets the parallel-beam parameters
	 * \param[in]   numAngles number of projection angles
	 * \param[in]   numRows number of rows in the x-ray detector
	 * \param[in]   numCols number of columns in the x-ray detector
	 * \param[in]   pixelHeight the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
	 * \param[in]   pixelWidth the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
	 * \param[in]   centerRow the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   centerCol the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   phis pointer to an array for specifying the angles of each projection, measured in degrees
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_parallelbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis);

	/**
	 * \fn          set_modularbeam
	 * \brief       sets the modular-beam parameters
	 * \param[in]   numAngles number of projection angles
	 * \param[in]   numRows number of rows in the x-ray detector
	 * \param[in]   numCols number of columns in the x-ray detector
	 * \param[in]   pixelHeight the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
	 * \param[in]   pixelWidth the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
	 * \param[in]   centerRow the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   centerCol the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
	 * \param[in]   sourcePositions (numAngles X 3) array of (x,y,z) coordinates of each source position
	 * \param[in]   moduleCenters (numAngles X 3) array of (x,y,z) coordinates of the center of each detector module
	 * \param[in]   rowVectors (numAngles X 3) array of vectors pointing in the positive detector row direction
	 * \param[in]   colVectors (numAngles X 3) array of vectors pointing in the positive detector column direction
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_modularbeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors);

	/**
	 * \fn          set_flatDetector
	 * \brief       sets the detectorType to parameters::FLAT
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_flatDetector();

	/**
	 * \fn          set_flatDetector
	 * \brief       sets the detectorType to parameters::CURVED
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_curvedDetector();

	/**
	 * \fn          set_centerCol
	 * \brief       sets the centerCol parameter
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_centerCol(float);

	/**
	 * \fn          set_centerCol
	 * \brief       sets the centerCol parameter
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_centerRow(float);

	/**
	 * \fn          set_volume
	 * \brief       sets the CT volume parameters
	 * \param[in]   numX number of voxels in the x-dimension
	 * \param[in]   numY number of voxels in the y-dimension
	 * \param[in]   numZ number of voxels in the z-dimension
	 * \param[in]   voxelWidth voxel pitch (size) in the x and y dimensions (mm)
	 * \param[in]   voxelHeight voxel pitch (size) in the z dimension (mm)
	 * \param[in]   offsetX shift the volume in the x-dimension (mm)
	 * \param[in]   offsetY shift the volume in the y-dimension (mm)
	 * \param[in]   offsetZ shift the volume in the z-dimension (mm)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_volume(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	/**
	 * \fn          set_defaultVolume
	 * \brief       sets the default CT volume parameters
	 * \param[in]   scale the default voxel size is divided by this number
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_default_volume(float scale = 1.0);

	/**
	 * \fn          set_volumeDimensionOrder
	 * \brief       sets the volumeDimensionOrder
	 * \param[in]   which 1 for the ZYX order, 0 for the XYZ order
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_volumeDimensionOrder(int which);

	/**
	 * \fn          get_volumeDimensionOrder
	 * \brief       gets the volumeDimensionOrder
	 * \return      params.volumeDimensionOrder
	 */
	int get_volumeDimensionOrder();

	/**
	 * \fn          set_GPU
	 * \brief       sets the primary GPU index
	 * \param[in]   whichGPU the primary GPU index one wishes to use
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_GPU(int whichGPU);

	/**
	 * \fn          set_GPUs
	 * \brief       sets a list of GPU indices to use
	 * \param[in]   whichGPUs array of GPU indices one wishes to use
	 * \param[in]   N the number of elements in the array
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_GPUs(int* whichGPUs, int N);

	/**
	 * \fn          get_GPU
	 * \brief       gets the primary GPU index
	 * \return      the primary GPU index
	 */
	int get_GPU();

	/**
	 * \fn          set_axisOfSymmetry
	 * \brief       sets axisOfSymmetry
	 * \param[in]   axisOfSymmetry the axis of symmetry angle (degrees)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_axisOfSymmetry(float axisOfSymmetry);

	/**
	 * \fn          clear_axisOfSymmetry
	 * \brief       clears the axisOfSymmetry parameter (turns the cylindrical symmetry model off)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool clear_axisOfSymmetry();

	/**
	 * \fn          set_projector (depreciated)
	 * \brief       sets the projector model (Separable Footprint, Siddon, Joseph)
	 * \param[in]   which the projector type (SIDDON=0,JOSEPH=1,SEPARABLE_FOOTPRINT=2)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_projector(int which);

	/**
	 * \fn          set_rFOV
	 * \brief       sets the radius of the cylindrical field of view in the x-y plane
	 * \param[in]   rFOV the radius of the field of view (mm)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_rFOV(float rFOV);

	/**
	 * \fn          set_rampID
	 * \brief       sets the rampID parameter which controls the sharpness of the filter
	 * \param[in]   rampID the order of the finite difference equation used
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_rampID(int);

	/**
	 * \fn          get_numAngles
	 * \brief       gets the numAngles parameter
	 * \return      numAngles
	 */
	int get_numAngles();

	/**
	 * \fn          get_numRows
	 * \brief       gets the numRows parameter
	 * \return      numRows
	 */
	int get_numRows();

	/**
	 * \fn          get_numCols
	 * \brief       gets the numCols parameter
	 * \return      numCols
	 */
	int get_numCols();

	/**
	 * \fn          get_pixelWidth
	 * \brief       gets the pixelWidth parameter
	 * \return      pixelWidth
	 */
	float get_pixelWidth();

	/**
	 * \fn          get_pixelHeight
	 * \brief       gets the pixelHeight parameter
	 * \return      pixelHeight
	 */
	float get_pixelHeight();

	/**
	 * \fn          set_tau
	 * \brief       sets tau
	 * \param[in]   tau the value for tau (mm)
	 * \return      true if successful, false otherwise
	 */
	bool set_tau(float tau_in);

	/**
	 * \fn          set_helicalPitch
	 * \brief       sets the helicalPitch parameter (mm/radian)
	 * \return      true if successful, false otherwise
	 */
	bool set_helicalPitch(float);

	/**
	 * \fn          set_normalizedHelicalPitch
	 * \brief       sets the helicalPitch and z_source_offset parameters
	 * \param[in]   h_normalized, the normalized helical pitch
	 * \return      true if successful, false otherwise
	 */
	bool set_normalizedHelicalPitch(float h_normalized);

	/**
	 * \fn          get_helicalPitch
	 * \brief       gets the helicalPitch
	 * \return      returns the helicalPitch parameter
	 */
	float get_helicalPitch();

	/**
	 * \fn          get_z_source_offset
	 * \brief       gets the z_source_offset
	 * \return      returns the z_source_offset parameter
	 */
	float get_z_source_offset();

	bool get_sourcePositions(float*);
	bool get_moduleCenters(float*);
	bool get_rowVectors(float*);
	bool get_colVectors(float*);

	/**
	 * \fn          get_numX
	 * \brief       gets the numX parameter
	 * \return      numX
	 */
	int get_numX();

	/**
	 * \fn          get_numY
	 * \brief       gets the numY parameter
	 * \return      numY
	 */
	int get_numY();

	/**
	 * \fn          get_numZ
	 * \brief       gets the numZ parameter
	 * \return      numZ
	 */
	int get_numZ();

	/**
	 * \fn          get_voxelWidth
	 * \brief       gets the voxelWidth parameter
	 * \return      voxelWidth
	 */
	float get_voxelWidth();

	/**
	 * \fn          get_voxelHeight
	 * \brief       gets the voxelHeight parameter
	 * \return      voxelHeight
	 */
	float get_voxelHeight();

	/**
	 * \fn          get_FBPscalar
	 * \brief       gets the scaling coefficient necessary for quantitatively-accurate FBP reconstruction
	 * \return      scaling factor
	 */
	float get_FBPscalar();

	/**
	 * \fn          set_attenuationMap
	 * \brief       sets the floating point array of the attenuation map (used in the Attenuated Radon Transform)
	 * \param[in]   mu the floating point array of attenuation values (mm^-1)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_attenuationMap(float* mu);

	/**
	 * \fn          set_attenuationMap
	 * \brief       sets the cylindrical attenuation parameters (used in the Attenuated Radon Transform)
	 * \param[in]   muCoeff attenuation coefficient (mm^-1)
	 * \param[in]   muRadius radius of the cylindrical attenuation map (mm)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_attenuationMap(float muCoeff, float muRadius);

	/**
	 * \fn          clear_attenuationMap
	 * \brief       clears all parameters associated with the Attenuated Radon Transform
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool clear_attenuationMap();

	/**
	 * \fn          flipAttenuationMapSign
	 * \brief       flips the sign of the attenuation map of the Attenuated Radon Transform
	 * \param[in]   data_on_cpu true if data (mu) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool flipAttenuationMapSign(bool data_on_cpu);

	/**
	 * \fn          Laplacian
	 * \brief       Applies the 2D Laplacian to each projection
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool Laplacian(float* g, bool data_on_cpu);

	/**
	 * \fn          AzimuthalBlur
	 * \brief       applies a low pass filter in the azimuthal angle
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   FWHM full width at half maximum of the filter (measured in degrees)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool AzimuthalBlur(float* f, float FWHM, bool data_on_cpu);

	/**
	 * \fn          applyTransferFunction
	 * \brief       applies a transfer function to arbitrary 3D data, i.e., x = LUT(x)
	 * \param[in]   x pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   LUT pointer to lookup table with transfer function values
	 * \param[in]   firstSample the value of the first sample in the lookup table
	 * \param[in]   sampleRate the step size between samples
	 * \param[in]   numSamples the number of elements in LUT
	 * \param[in]   data_on_cpu true if data (x and LUT) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);

	// Filters for 3D data
	/**
	 * \fn          BlurFilter
	 * \brief       applies a 3D low pass filter
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   FWHM full width at half maximum of the filter (measured in number of voxels)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool BlurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu);

	/**
	 * \fn          BlurFilter2D
	 * \brief       applies a 2D low pass filter to the second two dimensions of a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   FWHM full width at half maximum of the filter (measured in number of voxels)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool BlurFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu);

	/**
	 * \fn          MedianFilter
	 * \brief       applies a thresholded 3D median filter
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   threshold original value is only replaced by the median value
	 *              if the relative difference is greater than this value
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool MedianFilter(float* f, int, int, int, float threshold, bool data_on_cpu);

	/**
	 * \fn          MedianFilter2D
	 * \brief       applies a thresholded 2D median filter to the second two dimensions of a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   threshold original value is only replaced by the median value
	 *              if the relative difference is greater than this value
	 * \param[in]   w the window size in each dimension (must 3, 5, or 7)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool MedianFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, int w, bool data_on_cpu);

	// Anisotropic Total Variation for 3D data
	/**
	 * \fn          TVcost
	 * \brief       calculates the cost of the anisotropic Total Variation (aTV) functional
	 * \param[in]   f pointer to the 3D data (input)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]   beta the strength of the functional
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      value of the aTV functional
	 */
	float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu);

	/**
	 * \fn          TVgradient
	 * \brief       calculates the gradient of the anisotropic Total Variation (aTV) functional
	 * \param[in]   f pointer to the input 3D data
	 * \param[in]   Df pointer to the output 3D data
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]   beta the strength of the functional
	 * \param[in]   data_on_cpu true if data (f and Df) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise 
	 */
	bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu);

	/**
	 * \fn          TVquadForm
	 * \brief       calculates the quadratic form of the anisotropic Total Variation (aTV) functional
	 * \param[in]   f pointer to the 3D data (input)
	 * \param[in]   d the step direction
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]   beta the strength of the functional
	 * \param[in]   data_on_cpu true if data (f and d) are on the cpu, false if they are on the gpu
	 * \return      value of the aTV quadratic form, i.e., <d, dR''(d)>, where R'' is the second derivative of the aTV functional
	 */
	float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, bool data_on_cpu);

	/**
	 * \fn          Diffuse
	 * \brief       anisotropic Total Variation diffusion
	 * \param[in]   f pointer to the input/output 3D data
	 * \param[in]   Df pointer to the output 3D data
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]   numIter the number of iterations
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, int numIter, bool data_on_cpu);
	
	// Set all parameters and Project/Backproject
	bool projectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	parameters params;
private:

	/**
	 * \fn          project_multiGPU
	 * \brief       performs a forward projection on multiple GPUs
	 * \param[in]   g pointer to the projection data (output)
	 * \param[in]   f pointer to the volume data (input)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool project_multiGPU(float* g, float* f);

	/**
	 * \fn          project_multiGPU_splitViews
	 * \brief       performs a forward projection on multiple GPUs, by splitting up the projections (for modular and helical only)
	 * \param[in]   g pointer to the projection data (output)
	 * \param[in]   f pointer to the volume data (input)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool project_multiGPU_splitViews(float* g, float* f);

	/**
	 * \fn          backproject_multiGPU
	 * \brief       performs a backprojection on multiple GPUs
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool backproject_multiGPU(float* g, float* f);

	/**
	 * \fn          backproject_FBP_multiGPU_splitViews
	 * \brief       performs a backprojection or FBP on multiple GPUs, by splitting up the projections (for modular and helical only)
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool backproject_FBP_multiGPU_splitViews(float* g, float* f, bool doFBP);

	/**
	 * \fn          FBP_multiGPU
	 * \brief       performs an FBP reconstruction on multiple GPUs
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool FBP_multiGPU(float* g, float* f);
	
	/**
	 * \fn          backproject_FBP_multiGPU
	 * \brief       performs a backprojection or FBP reconstruction on multiple GPUs
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   f pointer to the volume data (output)
	 * \return      true if operation was executed on multiple GPUs, false if multi-GPU processing is not possible or not necessary
	 */
	bool backproject_FBP_multiGPU(float* g, float* f, bool doFBP);

	/**
	 * \fn          copyRows
	 * \brief       copies a specified set of rows from the projection data
	 * \param[in]   g pointer to the projection data (input)
	 * \param[in]   rowStart the first index to copy
	 * \param[in]   rowEnd the last index to copy
	 * \return      pointer to the copy of the data for the specified indices of rows
	 */
	float* copyRows(float* g, int rowStart, int rowEnd);

	/**
	 * \fn          combineRows
	 * \brief       sets a specified range of rows of a large data set from a smaller one
	 * \param[in]   g pointer to the full projection data
	 * \param[in]   g_chunk the cropped data
	 * \param[in]   rowStart the first index to copy
	 * \param[in]   rowEnd the last index to copy
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool combineRows(float* g, float* g_chunk, int rowStart, int rowEnd);

	/**
	 * \fn          project_memoryRequired
	 * \brief       calculates the memory required to run a forward projection using a subset of detector rows
	 * \param[in]   numRowsPerChunk the number of rows to calculate the projection
	 * \return      the number of GB of memory required
	 */
	float project_memoryRequired(int numRowsPerChunk);

	/**
	 * \fn          project_memoryRequired_splitViews
	 * \brief       calculates the memory required to run a forward projection using a subset of views
	 * \param[in]   numViewsPerChunk the number of views to calculate the projection
	 * \return      the number of GB of memory required
	 */
	float project_memoryRequired_splitViews(int numViewsPerChunk);

	/**
	 * \fn          backproject_memoryRequired
	 * \brief       calculates the memory required to run a backprojection using a subset of z-slices
	 * \param[in]   numSlicesPerChunk the number of z-slice to calculate the backprojection
	 * \param[in]   extraCols the number of extra columns that will be needed to be added to the data for processing
	 * \return      the number of GB of memory required
	 */
	float backproject_memoryRequired(int numSlicesPerChunk, int extraCols = 0);

	/**
	 * \fn          backproject_memoryRequired
	 * \brief       calculates the memory required to run a backprojection using a subset of z-slices and splitting the projections across views
	 * \param[in]   numSlicesPerChunk the number of z-slice to calculate the backprojection
	 * \return      the number of GB of memory required
	 */
	float backproject_memoryRequired_splitViews(int numSlicesPerChunk);

	int maxSlicesForChunking;

	filteredBackprojection FBP;
	projectors proj;
};

#endif
