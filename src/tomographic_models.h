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

#define LEAP_VERSION "1.26"

/*
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
//*/

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include "parameters.h"
#include "projectors.h"
#include "filtered_backprojection.h"
#include "cpu_utils.h"
#include "phantom.h"

/**
 *  tomographicModels class
 * This is the main interface for LEAP.  All calls to forward project, backproject, FBP, filtering, noise filters come through this class.
 * The main job of this class is to set/get parameters, do error checks, and dispatch jobs.  It contains almost no algorithm logic.
 * In addition to the jobs listed above, this class is also responsible for divide jobs across multiple GPUs or dividing up GPU jobs so that
 * they fit into the available GPU memory.  Functions called from this class are either CPU based or single GPU based.
 */

class tomographicModels
{
public:
	tomographicModels();
	~tomographicModels();

	void set_log_error();
	void set_log_warning();
	void set_log_status();
	void set_log_debug();

	/**
	 * \fn          reset
	 * \brief       resets (clears) all CT geometry and CT volume parameter values
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool reset();

	/**
	 * \fn          allocate_volume
	 * \brief       allocates volume data; user is responsible for freeing memory
	 * \return      pointer to the data, NULL if CT volume parameters are not set
	 */
	float* allocate_volume();

	/**
	 * \fn          allocate_projections
	 * \brief       allocates projection data; user is responsible for freeing memory
	 * \return      pointer to the data, NULL if CT geometry parameters are not set
	 */
	float* allocate_projections();

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
	 * \fn          project_with_mask
	 * \brief       performs a forward projection
	 * \param[in]   g pointer to the projection data (output)
	 * \param[in]   f pointer to the volume data (input)
	 * \param[in]   mask: pointer to the mask data
	 * \param[in]   data_on_cpu true if data (f and g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool project_with_mask(float* g, float* f, float* mask, bool data_on_cpu);
	bool project_with_mask_cpu(float* g, float* f, float* mask);
	bool project_with_mask_gpu(float* g, float* f, float* mask);

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
	 * \param[in]   g pointer to the input projection data
	 * \param[in]	g_out pointer to the output projection data
	 * \param[in]   data_on_cpu true if data (g and g_out) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool filterProjections(float* g, float* g_out, bool data_on_cpu);

	/**
	 * \fn          filterProjections_cpu
	 * \brief       applies the necessary filters and ray/view weights necessary for FBP reconstruction
	 * \param[in]   g pointer to the projection data (input and output) on the cpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool filterProjections_cpu(float* g);

	/**
	 * \fn          filterProjections_gpu
	 * \brief       applies the necessary filters and ray/view weights necessary for FBP reconstruction
	 * \param[in]   g pointer to the projection data (input and output) on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool filterProjections_gpu(float* g);

	/**
	 * \fn          preRampFiltering
	 * \brief       Applies all pre ramp filter weighting
	 * \param[in]   g pointer to the projection data (input and output)
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool preRampFiltering(float* g, bool data_on_cpu);

	/**
	 * \fn          postRampFiltering
	 * \brief       Applies all post ramp filter weighting
	 * \param[in]   g pointer to the projection data (input and output)
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool postRampFiltering(float* g, bool data_on_cpu);

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
	 * \param[in]	tiltAngle the rotation of the detector around the optical axis (degrees)
	 * \param[in]   helicalPitch the helical pitch (mm/radians)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_conebeam(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau = 0.0, float tiltAngle = 0.0, float helicalPitch = 0.0);

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
	 * \fn          set_coneparallel
	 * \brief       sets the cone-parallel parameters
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
	bool set_coneparallel(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, float tau = 0.0, float helicalPitch = 0.0);

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
	 * \fn          number_of_gpus
	 * \return      number of GPUs on the system
	 */
	int number_of_gpus();

	/**
	 * \fn          get_gpus
	 * \brief       gets a list of all gpus being used
	 * \return      number of gpus being used
	 */
	int get_gpus(int* list_of_gpus);

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
	 * \fn          set_tiltAngle
	 * \brief       sets tiltAngle
	 * \param[in]   tiltAngle the value for tiltAngle (degrees)
	 * \return      true if the value is valid, false otherwise
	 */
	bool set_tiltAngle(float tiltAngle_in);

	/**
	 * \fn          get_tiltAngle
	 * \brief       gets tiltAngle
	 * \return      tiltAngle
	 */
	float get_tiltAngle();

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
	 * \fn          find_centerCol
	 * \brief       finds centerCol of parallel-, fan-, or cone-beam data using conjugate rays
	 * \param[in]   g pointer to the projection data
	 * \param[in]   iRow the detector row index to use the estimate centerCol
	 * \param[in]	searchBounds 2-element array specifying the search bounds
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	float find_centerCol(float* g, int iRow, float* searchBounds, bool data_on_cpu);

	/**
	 * \fn          find_tau
	 * \brief       finds tau of fan- or cone-beam data using conjugate rays
	 * \param[in]   g pointer to the projection data
	 * \param[in]   iRow the detector row index to use the estimate centerCol
	 * \param[in]	searchBounds 2-element array specifying the search bounds
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	float find_tau(float* g, int iRow, float* searchBounds, bool data_on_cpu);

	/**
	 * \fn          estimate_tilt
	 * \brief       finds the tilt angle (around the optical axis) of parallel-, fan-, or cone-beam data using conjugate rays
	 * \param[in]   g pointer to the projection data
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if they are on the gpu
	 * \return      the estimated detector tilt (degrees)
	 */
	float estimate_tilt(float* g, bool data_on_cpu);

	/**
	 * \fn          conjugate_difference
	 * \brief       calculates the difference of two conjugate projections with optional rotation
	 * \param[in]   g pointer to the projection data
	 * \param[in]   alpha: detector rotation in degrees
	 * \param[in]	centerCol: detector center column index
	 * \param[in]   diff: pointer to data numRows*numCols where the difference is stored
	 * \param[in]   data_on_cpu: true if data (g) is on the cpu, false if they are on the gpu
	 * \return      the estimated detector tilt (degrees)
	 */
	bool conjugate_difference(float* g, float alpha, float centerCol, float* diff, bool data_on_cpu);

	/**
	 * \fn          consistency_cost
	 * \brief       calculates of geometric calibration cost with the given perturbations
	 * \param[in]   g: pointer to the projection data
	 * \param[in]	Delta_centerRow: detector shift (detector row pixel index) in the row direction
	 * \param[in]	Delta_centerCol: detector shift (detector column pixel index) in the column direction
	 * \param[in]	Delta_tau: horizonal shift (mm) of the detector; can also be used to model detector rotations along the vector pointing across the detector rows
	 * \param[in]	Delta_tilt: rotation (degrees) of the detector around the optical axis
	 * \param[in]   data_on_cpu: true if data (g) is on the cpu, false if they are on the gpu
	 * \return      cost of the calibration metric
	 */
	float consistency_cost(float* g, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt, bool data_on_cpu);

	/**
	 * \fn          Laplacian
	 * \brief       Applies the 2D Laplacian to each projection
	 * \param[in]   g pointer to the projection data
	 * \param[in]   numDims the number of dimensions of the Laplacian
	 * \param[in]   smooth if true uses a smooth finite difference
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool Laplacian(float* g, int numDims, bool smooth, bool data_on_cpu);

	/**
	 * \fn          transmissionFilter
	 * \brief       Applies a 2D Filter to each transmission projection
	 * \param[in]   g pointer to the projection data
	 * \param[in]   H pointer to the magnitude of the frequency response of the filter
	 * \param[in]   N_H1 number of samples of H in the first dimension
	 * \param[in]   N_H1 number of samples of H in the second dimension
	 * \param[in]   isAttenuationData true if data (g) is attenuation (post-log), false otherwise
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool transmissionFilter(float* g, float* H, int N_H1, int N_H2, bool isAttenuationData, bool data_on_cpu);

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

	/**
	 * \fn          beam_hardening_heel_effect
	 * \brief       applies a transfer function to projection 3D data, i.e., x = LUT(x) to apply/correct for beam hardening with the heel effect
	 * \param[in]   g pointer to the 3D data (input and output)
	 * \param[in]	anode_normal, unit vector normal to the anode
	 * \param[in]   LUT pointer to lookup table with transfer function values
	 * \param[in]	takeOffAngles the takeoff angles modeled in the lookup table
	 * \param[in]   numSamples the number of elements in LUT
	 * \param[in]   numAngles the number of angles
	 * \param[in]   sampleRate the step size between samples
	 * \param[in]   firstSample the value of the first sample in the lookup table
	 * \param[in]   data_on_cpu true if data (x and LUT) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool beam_hardening_heel_effect(float* g, float* anode_normal, float* LUT, float* takeOffAngles, int numSamples, int numAngles, float sampleRate, float firstSample, bool data_on_cpu);

	/**
	 * \fn          applyDualTransferFunction
	 * \brief       applies a 2D transfer function to arbitrary 3D data pair, i.e., x,y = LUT(x,y)
	 * \param[in]   x pointer to the 3D data of first component (input and output)
	 * \param[in]   y pointer to the 3D data of second component (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   LUT pointer to lookup table with transfer function values
	 * \param[in]   firstSample the value of the first sample in the lookup table
	 * \param[in]   sampleRate the step size between samples
	 * \param[in]   numSamples the number of elements in LUT
	 * \param[in]   data_on_cpu true if data (x, y, and LUT) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu);

	/**
	 * \fn          convertToRhoeZe
	 * \brief       transforms a low and high energy pair to electron density and effective atomic number
	 * \param[in]   f_L pointer to the 3D low energy channel (input and output)
	 * \param[in]   f_H pointer to the 3D high energy channel (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   sigma_L pointer to the cross section values of the elements and the low energy
	 * \param[in]   sigma_H pointer to the cross section values of the elements and the high energy
	 * \param[in]   data_on_cpu true if data (x, y, and LUT) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool data_on_cpu);

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
	 * \fn          HighPassFilter
	 * \brief       applies a 3D high pass filter
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   FWHM full width at half maximum of the filter (measured in number of voxels)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool HighPassFilter(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu);

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
	 * \fn          HighPassFilter2D
	 * \brief       applies a 2D high pass filter to the second two dimensions of a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   FWHM full width at half maximum of the filter (measured in number of voxels)
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool HighPassFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, bool data_on_cpu);

	/**
	 * \fn          MedianFilter
	 * \brief       applies a thresholded 3D median filter
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   threshold original value is only replaced by the median value
	 *              if the relative difference is greater than this value
	 * \param[in]   w the window size in each dimension (must be 3 or 5)
	 * \param[in]	signalThreshold: if greater than zero, only values less than this parameter are filtered
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool MedianFilter(float* f, int, int, int, float threshold, int w, float signalThreshold, bool data_on_cpu);

	/**
	 * \fn          MeanOrVarianceFilter
	 * \brief       applies a 3D mean or variance filter
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   r the radius (in number of pixels) of the window
	 * \param[in]	order 1 for mean, 2 for variance filter
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool MeanOrVarianceFilter(float* f, int, int, int, int r, int order, bool data_on_cpu);

	/**
	 * \fn          MedianFilter2D
	 * \brief       applies a thresholded 2D median filter to the second two dimensions of a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   threshold original value is only replaced by the median value
	 *              if the relative difference is greater than this value
	 * \param[in]   w the window size in each dimension (must be 3, 5, or 7)
	 * \param[in]	signalThreshold: if greater than zero, only values less than this parameter are filtered
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool MedianFilter2D(float* f, int N_1, int N_2, int N_3, float FWHM, int w, float signalThreshold, bool data_on_cpu);

	/**
	 * \fn          badPixelCorrection
	 * \brief       applies a 2D median filter to the second two dimensions of a 3D array to a specified list of pixels only
	 * \param[in]   g: pointer to the 3D projection data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   badPixelMap: pointer to a 2D array of the bad pixels which are label as 1.0
	 * \param[in]   w: the window size in each dimension (must be 3, 5, or 7)
	 * \param[in]   data_on_cpu: true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool badPixelCorrection(float* g, int N_1, int N_2, int N_3, float* badPixelMap, int w, bool data_on_cpu);

	/**
	 * \fn          BilateralFilter
	 * \brief       applies a (scaled) 3D bilateral filter (BLF) to a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   spatialFWHM the FWHM (in number of pixels) of the spatial closeness term of the BLF
	 * \param[in]   intensityFWHM the FWHM of the intensity closeness terms of the BLF
	 * \param[in]   scale an optional argument to used a blurred to calculate the intensity closeness term
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool BilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float scale, bool data_on_cpu);

	/**
	 * \fn          PriorBilateralFilter
	 * \brief       applies a 3D bilateral filter (BLF) to a 3D array where the intensity distance is measured against a prior image
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   spatialFWHM the FWHM (in number of pixels) of the spatial closeness term of the BLF
	 * \param[in]   intensityFWHM the FWHM of the intensity closeness terms of the BLF
	 * \param[in]   prior pointer to 3D data prior
	 * \param[in]   data_on_cpu true if data (f and prior) is on the cpu, false if they are both on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool PriorBilateralFilter(float* f, int N_1, int N_2, int N_3, float spatialFWHM, float intensityFWHM, float* prior, bool data_on_cpu);

	/**
	 * \fn          GuidedFilter
	 * \brief       applies a (scaled) 3D guided filter to a 3D array
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   r the window radius (in number of pixels)
	 * \param[in]   epsilon the degree of smoothing
	 * \param[in]	numIter the number of iteration of the algorithm to run
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool GuidedFilter(float* f, int N_1, int N_2, int N_3, int r, float epsilon, int numIter, bool data_on_cpu);

	/**
	 * \fn          dictionaryDenoising
	 * \brief       represents 3D data by a sparse representation of an overcomplete dictionary
	 * \param[in]   f pointer to the 3D data (input and output)
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   dictionary pointer to the overcomplete dictionary
	 * \param[in]   N_d1 number of pixels in the first dimension of a dictionary elements
	 * \param[in]   N_d2 number of pixels in the second dimension of a dictionary elements
	 * \param[in]   N_d3 number of pixels in the third dimension of a dictionary elements
	 * \param[in]   epsilon the fitting metric
	 * \param[in]   sparsityThreshold the maximum number of dictionary elements to use
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu);

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
	 * \param[in]	p the exponent on the Huber-like loss function
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      value of the aTV functional
	 */
	float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);

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
	 * \param[in]	p the exponent on the Huber-like loss function
	 * \param[in]   data_on_cpu true if data (f and Df) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise 
	 */
	bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, float p, bool doMean, bool data_on_cpu);

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
	 * \param[in]	p the exponent on the Huber-like loss function
	 * \param[in]   data_on_cpu true if data (f and d) are on the cpu, false if they are on the gpu
	 * \return      value of the aTV quadratic form, i.e., <d, dR''(d)>, where R'' is the second derivative of the aTV functional
	 */
	float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, float p, bool data_on_cpu);

	/**
	 * \fn          Diffuse
	 * \brief       anisotropic Total Variation diffusion
	 * \param[in]   f pointer to the input/output 3D data
	 * \param[in]   Df pointer to the output 3D data
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]	p the exponent on the Huber-like loss function
	 * \param[in]   numIter the number of iterations
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool Diffuse(float* f, int N_1, int N_2, int N_3, float delta, float p, int numIter, bool data_on_cpu);

	/**
	 * \fn          TV_denoise
	 * \brief       anisotropic Total Variation denoising
	 * \param[in]   f pointer to the input/output 3D data
	 * \param[in]   Df pointer to the output 3D data
	 * \param[in]   N_1 number of samples in the first dimension
	 * \param[in]   N_2 number of samples in the second dimension
	 * \param[in]   N_3 number of samples in the third dimension
	 * \param[in]   delta transition value of the Huber-like loss function
	 * \param[in]	beta: the regularization strength
	 * \param[in]	p the exponent on the Huber-like loss function
	 * \param[in]   numIter the number of iterations
	 * \param[in]   data_on_cpu true if data (f) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool TV_denoise(float* f, int N_1, int N_2, int N_3, float delta, float beta, float p, int numIter, bool doMean, bool data_on_cpu);

	/**
	 * \fn          rayTrace
	 * \brief       analytic ray tracing simulation
	 * \param[in]   g pointer to the projection data
	 * \param[in]   oversampling the detector oversampling factor
	 * \param[in]   data_on_cpu true if data (g) is on the cpu, false if it is on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool rayTrace(float* g, int oversampling, bool data_on_cpu);

	/**
	 * \fn          rebin_curved
	 * \brief       rebin a collection of flat detector modules to a curved detector
	 * \param[in]   g pointer to the projection data
	 * \param[in]   fanAngles the measured fan angles
	 * \param[in]	order the order of the interpolation polynomial
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool rebin_curved(float* g, float* fanAngles, int order = 6);

	/**
	 * \fn          rebin_parallel
	 * \brief       rebin fan-beam to parallel-beam or cone-beam to cone-parallel
	 * \param[in]   g pointer to the projection data
	 * \param[in]	order the order of the interpolation polynomial
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool rebin_parallel(float* g, int order = 6);

	/**
	 * \fn          sinogram_replacement
	 * \brief       replaces specified region in projection data with other projection data
	 * \param[in]   g, pointer to the projection data to alter
	 * \param[in]	priorSinogram, poiner to the projection data to use for patching
	 * \param[in]	metalTrace, pointer to projection mask showing where to do the patching
	 * \param[in]   windowSize, 3-element int array of the window size in each of the three dimensions
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool sinogram_replacement(float* g, float* priorSinogram, float* metalTrace, int* windowSize);
	
	/**
	 * \fn          down_sample
	 * \brief       down-samples 3D array
	 * \param[in]   I, pointer to the original data
	 * \param[in]	N, 3-element array of the size of each dimension of the original data
	 * \param[in]	I_dn, pointer to the output (down-sampled) 3D array
	 * \param[in]	N_dn, 3-element array of the size of each dimension of the down-sampled data
	 * \param[in]	factors, 3-element array of the down-sampling factors in each dimension
	 * \param[in]	data_on_cpu true if data (I and I_dn) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool down_sample(float* I, int* N, float* I_dn, int* N_dn, float* factors, bool data_on_cpu);

	/**
	 * \fn          up_sample
	 * \brief       up-samples 3D array
	 * \param[in]   I, pointer to the original data
	 * \param[in]	N, 3-element array of the size of each dimension of the original data
	 * \param[in]	I_up, pointer to the output (up-sampled) 3D array
	 * \param[in]	N_up, 3-element array of the size of each dimension of the up-sampled data
	 * \param[in]	factors, 3-element array of the up-sampling factors in each dimension
	 * \param[in]	data_on_cpu true if data (I and I_up) is on the cpu, false if they are on the gpu
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool up_sample(float* I, int* N, float* I_up, int* N_up, float* factors, bool data_on_cpu);

	/**
	 * \fn          scatter_model
	 * \brief       simulates first order scatter through an object composed of a single material type
	 * \param[in]   g, pointer to store the simulated scatter data
	 * \param[in]	f, pointer to the mass density volume (g/mm^3)
	 * \param[in]	source, pointer to the source spectra
	 * \param[in]	energies, pointer to the energy bins in the source spectra model
	 * \param[in]	N_energies: number of energy bins
	 * \param[in]	detector, pointer to the detector response in 1 keV bins
	 * \param[in]	sigma, pointer to the PE, CS, and RS cross sections in 1 keV bins
	 * \param[in]	scatterDist, pointer to the normalized CS and RS distributions sampled in 1 keV bins and 0.1 degree angular bins
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool scatter_model(float* g, float* f, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType);

	/**
	 * \fn          synthesize_symmetry
	 * \brief       converts a symmetric object into a 3D volume
	 * \param[in]   f_radial: pointer to the radial volume
	 * \param[in]	f: pointer to the 3D volume
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool synthesize_symmetry(float* f_radial, float* f);

	/**
	 * \fn          set_maxSlicesForChunking
	 * \brief       sets the maximum number of slices per chunk to process on GPU
	 * \param[in]   N: maximum number of slices per chunk to process on GPU
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool set_maxSlicesForChunking(int N);

	// Set all parameters and Project/Backproject
	bool projectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectFanBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectConeBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	bool projectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);
	bool backprojectParallelBeam(float* g, float* f, bool data_on_cpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ);

	/**
	 * \fn          numRowsRequiredForBackprojectingSlab
	 * \brief       determines the maximum number of detector rows required to backproject a sub-volume with a specified number of z-slices
	 * \param[in]   numSlicesPerChunk: number of z-slices in a subvolume
	 * \return      number of slices required
	 */
	int numRowsRequiredForBackprojectingSlab(int numSlicesPerChunk);
	int extraColumnsForOffsetScan();

	parameters params;
	phantom geometricPhantom;
private:

	/**
	 * \fn          filterProjections_multiGPU
	 * \brief       applies the necessary filters and ray/view weights necessary for FBP reconstruction
	 * \param[in]   g pointer to the projection data (on CPU)
	 * \param[in]	g_out pointer to the output projection data (can be the same as the input)
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool filterProjections_multiGPU(float* g, float* g_out);

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
	 * \param[in]	firstView the first view to copy
	 * \param[in]	lastView the last view to copy
	 * \return      pointer to the copy of the data for the specified indices of rows
	 */
	float* copyRows(float* g, int rowStart, int rowEnd, int firstView = -1, int lastView = -1);

	/**
	 * \fn          combineRows
	 * \brief       sets a specified range of rows of a large data set from a smaller one
	 * \param[in]   g pointer to the full projection data
	 * \param[in]   g_chunk the cropped data
	 * \param[in]   rowStart the first index to copy
	 * \param[in]   rowEnd the last index to copy
	 * \param[in]	firstView the first view to copy
	 * \param[in]	lastView the last view to copy
	 * \return      true if operation  was sucessful, false otherwise
	 */
	bool combineRows(float* g, float* g_chunk, int rowStart, int rowEnd, int firstView = -1, int lastView = -1);

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
	float backproject_memoryRequired(int numSlicesPerChunk, int extraCols = 0, bool doFBP = true, int numViews = -1);

	/**
	 * \fn          backproject_memoryRequired
	 * \brief       calculates the memory required to run a backprojection using a subset of z-slices and splitting the projections across views
	 * \param[in]   numSlicesPerChunk the number of z-slice to calculate the backprojection
	 * \return      the number of GB of memory required
	 */
	float backproject_memoryRequired_splitViews(int numSlicesPerChunk, bool doFBP = true);

	bool copy_volume_data_to_mask(float* f, float* mask, bool data_on_cpu, bool do_forward);

	int maxSlicesForChunking;
	int minSlicesForChunking;

	filteredBackprojection FBP;
	projectors proj;

	std::string className;
	//std::ofstream pfile;
};

#endif
