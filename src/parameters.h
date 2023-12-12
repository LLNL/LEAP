////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// parameters c++ header
////////////////////////////////////////////////////////////////////////////////

#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#ifdef WIN32
#pragma once
#endif

#include "leap_defines.h"
#include <vector>

/**
 *  parameters class
 * This class tracks all the LEAP parameters including: the CT geometry parameters, CT volume parameters, which GPUs to use, etc.
 * A pointer to an instance of this class is usually passed with the input/output arrays so that the algorithms are aware of the
 * CT parameters.  For dividing parameters into chunks, usually copies of the original object are made and certain parameters
 * updated to hande the sub-job.
 */

class parameters
{
public:
	parameters();
    parameters(const parameters& other);
	~parameters();

	/**
	 * \fn          operator =
	 * \brief       makes a deep copy of the given parameter object
	 */
    parameters& operator = (const parameters& other);

	/**
	 * \fn          initialize
	 * \brief       initialize all CT geometry and CT volume parameter values
	 */
	void initialize();

	/**
	 * \fn          printAll
	 * \brief       prints all CT geometry and CT volume parameter values
	 */
	void printAll();

	/**
	 * \fn          clearAll
	 * \brief       clears all (including memory) CT geometry and CT volume parameter values
	 */
	void clearAll();

	/**
	 * \fn          allDefined
	 * \brief       returns whether all CT geometry and CT volume parameter values are defined and valid
	 * \return      returns true if all CT geometry and CT volume parameter values are defined and valid, false otherwise
	 */
	bool allDefined();

	/**
	 * \fn          allDefined
	 * \brief       returns whether all CT geometry parameter values are defined and valid
	 * \return      returns true if all CT geometry parameter values are defined and valid, false otherwise
	 */
	bool geometryDefined();

	/**
	 * \fn          allDefined
	 * \brief       returns whether all CT volume parameter values are defined and valid
	 * \return      returns true if all CT volume parameter values are defined and valid, false otherwise
	 */
	bool volumeDefined();

	/**
	 * \fn          default_voxelWidth
	 * \brief       calculates the default voxelWidth value
	 * \return      returns the default voxelWidth value
	 */
	float default_voxelWidth();

	/**
	 * \fn          default_voxelHeight
	 * \brief       calculates the default voxelHeight value
	 * \return      returns the default voxelHeight value
	 */
	float default_voxelHeight();

	/**
	 * \fn          set_default_volume
	 * \brief       sets the default CT volume parameters
	 * \param[in]   scale the default voxel size is divided by this number
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool set_default_volume(float scale = 1.0);

	/**
	 * \fn          set_angles
	 * \brief       sets phis, an array of the projection angles
	 * \param[in]   phis_in an array (degrees) of the projection angles
	 * \param[in]   numPhis the number of elements in the input array
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool set_angles(float* phis_in, int numPhis);

	/**
	 * \fn          set_angles
	 * \brief       sets phis, an array of the projection angles based on angularRange and numAngles
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool set_angles();

	/**
	 * \fn          get_angles
	 * \brief       populates the input array with phis, an array of the projection angles
	 * \param[in]   phis_in an array (degrees) of the projection angles
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool get_angles(float* phis_in);

	/**
	 * \fn          phaseShift
	 * \brief       shifts the values in the projection angle array
	 * \param[in]   radians the phase shift (in radians)
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool phaseShift(float radians);

	/**
	 * \fn          anglesAreEquispaced
	 * \brief       returns whether or not the projection angle array values are equi-spaced
	 * \return      true if the projection angle array values are equi-spaced, false otherwise
	 */
	bool anglesAreEquispaced();

	/**
	 * \fn          u_0
	 * \brief       returns the location (in mm) of the first detector pixel column
	 * \return      returns the location (in mm) of the first detector pixel column
	 */
	float u_0();

	/**
	 * \fn          v_0
	 * \brief       returns the location (in mm) of the first detector pixel row
	 * \return      returns the location (in mm) of the first detector pixel row
	 */
	float v_0();

	/**
	 * \fn          v_0
	 * \brief       returns the angle of the first projection (radians)
	 * \return      returns the angle of the first projection (radians)
	 */
	float phi_0();

	/**
	 * \fn          x_0
	 * \brief       returns the location (in mm) of the first x-coordinate value
	 * \return      returns the location (in mm) of the first x-coordinate value
	 */
	float x_0();

	/**
	 * \fn          y_0
	 * \brief       returns the location (in mm) of the first y-coordinate value
	 * \return      returns the location (in mm) of the first y-coordinate value
	 */
	float y_0();

	/**
	 * \fn          z_0
	 * \brief       returns the location (in mm) of the first z-coordinate value
	 * \return      returns the location (in mm) of the first z-coordinate value
	 */
	float z_0();

	/**
	 * \fn          furthestFromCenter
	 * \brief       returns the largest distance from the z-axis of all the voxel positions
	 * \return      returns the largest distance from the z-axis of all the voxel positions
	 */
	float furthestFromCenter();

	/**
	 * \fn          pixelWidth_normalized
	 * \brief       returns pixelWidth/sdd for fan- and cone-beam, pixelWidth otherwise
	 * \return      returns pixelWidth/sdd for fan- and cone-beam, pixelWidth otherwise
	 */
	float pixelWidth_normalized();

	/**
	 * \fn          u
	 * \brief       returns the position of the i-th detector column (mm)
	 * \param       i the index of the i-th detector column
	 * \return      returns the position of the i-th detector column (mm)
	 */
	float u(int i);

	/**
	 * \fn          u_inv
	 * \brief       returns the index of the u samples at the given position
	 * \param       val the u position
	 * \return      returns the index of the u samples at the given position
	 */
	float u_inv(float val);

	/**
	 * \fn          v
	 * \brief       returns the position of the i-th detector row (mm)
	 * \param       i the index of the i-th detector row
	 * \return      returns the position of the i-th detector row (mm)
	 */
	float v(int i);

	/**
	 * \fn          z_samples
	 * \brief       returns the position of the k-th z-coordinate value (mm)
	 * \param[in]   k the index of the k-th z-coordinate
	 * \return      returns the position of the k-th z-coordinate value (mm)
	 */
	float z_samples(int k);

	/**
	 * \fn          z_source
	 * \brief       returns the z-coordinate of the source position at the i-th rotation angle
	 * \param[in]   i the index of the i-th rotation angle
	 * \return      returns the z-coordinate of the source position at the i-th rotation angle
	 */
	float z_source(int i);

	/**
	 * \fn          set_tau
	 * \brief       sets tau
	 * \param[in]   tau the value for tau (mm)
	 * \return      true is successful, false otherwise
	 */
	bool set_tau(float tau_in);

	/**
	 * \fn          normalizedHelicalPitch
	 * \brief       calculates the normalized helical pitch
	 * \return      returns the normalized helical pitch
	 */
	float normalizedHelicalPitch();

	/**
	 * \fn          set_helicalPitch
	 * \brief       sets the helicalPitch and z_source_offset parameters
	 * \param[in]   h the helicalPitch (mm/radian)
	 * \return      true is successful, false otherwise
	 */
	bool set_helicalPitch(float h);

	/**
	 * \fn          set_normalizedHelicalPitch
	 * \brief       sets the helicalPitch and z_source_offset parameters
	 * \param[in]   h_normalized, the normalized helical pitch
	 * \return      true is successful, false otherwise
	 */
	bool set_normalizedHelicalPitch(float h_normalized);

	bool normalizeConeAndFanCoordinateFunctions;

	int whichGPU;
	std::vector<int> whichGPUs;

	// Scanner Parameters
	int geometry;
	int detectorType;
	float sod, sdd;
	float pixelWidth, pixelHeight, angularRange;
	int numCols, numRows, numAngles;
	float centerCol, centerRow;
	float* phis;
	float tau;
	float helicalPitch;
	float z_source_offset;

	// Volume Parameters
	int volumeDimensionOrder;
	int numX, numY, numZ;
	float voxelWidth, voxelHeight;
	float offsetX, offsetY, offsetZ;

	// Modular-Beam Parameters
	float* sourcePositions;
	float* moduleCenters;
	float* rowVectors;
	float* colVectors;

	/**
	 * \fn          set_sourcesAndModules
	 * \brief       sets the modular-beam source and detector positions and orientations
	 * \param[in]   sourcePositions_in (numPairs x 3) array of the source positions
	 * \param[in]   moduleCenters_in (numPairs x 3) array of the detector module center positions
	 * \param[in]   rowVectors_in (numPairs x 3) array of the vectors pointing along the row dimension
	 * \param[in]   colVectors_in (numPairs x 3) array of the vectors pointing along the column dimension
	 * \param[in]   numPairs the number of source-detector pairs
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool set_sourcesAndModules(float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in, int numPairs);

	bool set_offsetScan(bool aFlag);
	bool set_truncatedScan(bool aFlag);

	// Attenuated Radon Transform
	float* mu;
	float muCoeff;
	float muRadius;

	/**
	 * \fn          muSpecified
	 * \brief       returns whether or not an attenuation map (for the Attenuated Radon Transform) has been specified
	 * \return      true if mu != NULL or (muCoeff != 0.0 and muRadius > 0.0), false otherwise
	 */
	bool muSpecified();

	// Reconstruction Parameters
	int whichProjector;
	bool doWeightedBackprojection;
	bool doExtrapolation;
	float rFOVspecified;
	int rampID;
	float colShiftFromFilter;
	float rowShiftFromFilter;
	float axisOfSymmetry;
	float chunkingMemorySizeThreshold;
	bool offsetScan;
	bool truncatedScan;
    
	/**
	 * \fn          T_phi
	 * \brief       returns the mean distance between projection angles (radians)
	 * \return      returns the mean distance between projection angles (radians)
	 */
    float T_phi();

	/**
	 * \fn          phi_inv
	 * \brief       returns the real-valued view index for the given view angle
	 * \param[in]   angle the angle (in radians) of a projection/view
	 * \return      returns the real-valued view index for the given view angle
	 */
	float phi_inv(float angle);

	/**
	 * \fn          rFOV
	 * \brief       returns the radius of the field of view of the CT system
	 * \return      returns the radius of the field of view of the CT system
	 */
    float rFOV();
	
	/**
	 * \fn          isSymmetric
	 * \brief       returns whether or not the cylindrically symmetric projectors are enabled
	 * \return      returns true if the cylindrically symmetric projectors are enabled, false otherwise
	 */
	bool isSymmetric();

	/**
	 * \fn          useSF
	 * \brief       returns whether or not the Separable Footprint projectors are to be used
	 * \return      returns true if the Separable Footprint projectors are to be used, false otherwise
	 */
    bool useSF();

	/**
	 * \fn          setToConstant
	 * \brief       sets the array to value
	 * \return      returns pointer to the array if successful
	 */
	float* setToConstant(float* data, uint64 N, float val = 0.0);

	/**
	 * \fn          setToZero
	 * \brief       sets the array to zero
	 * \return      returns pointer to the array if successful
	 */
	float* setToZero(float* data, uint64 N);

	/**
	 * \fn          windowFOV
	 * \brief       sets the array to zero for those values outside the field of view
	 * \return      true is operation  was sucessful, false otherwise
	 */
	bool windowFOV(float*);

	//float smallestVoxelForFastSF();
	//float largestVoxelForFastSF();

	/**
	 * \fn          voxelSizeWorksForFastSF
	 * \brief       returns whether or not the voxel size is appropriate for the fast SF projectors
	 * \return      returns true if the voxel size is appropriate for the fast SF projectors, false otherwise
	 */
	bool voxelSizeWorksForFastSF();

	/**
	 * \fn          projectionData_numberOfElements
	 * \brief       returns the total number of elements in the projection data
	 * \return      returns the total number of elements in the projection data
	 */
	uint64 projectionData_numberOfElements();

	/**
	 * \fn          volumeData_numberOfElements
	 * \brief       returns the total number of voxels in the volume data
	 * \return      returns the total number of voxels in the volume data
	 */
	uint64 volumeData_numberOfElements();

	/**
	 * \fn          projectionDataSize
	 * \brief       returns the number of GB of memory required for the projection data
	 * \param[in]   extraCols the number of extra columns that will be needed to be added to the data for processing
	 * \return      returns the number of GB of memory required for the projection data
	 */
	float projectionDataSize(int extraCols = 0);

	/**
	 * \fn          volumeDataSize
	 * \brief       returns the number of GB of memory required for the volume data
	 * \return      returns the number of GB of memory required for the volume data
	 */
	float volumeDataSize();

	/**
	 * \fn          requiredGPUmemory
	 * \brief       returns projectionDataSize() + volumeDataSize()
	 * \param[in]   extraCols the number of extra columns that will be needed to be added to the data for processing
	 * \return      returns projectionDataSize() + volumeDataSize()
	 */
	float requiredGPUmemory(int extraCols = 0);

	/**
	 * \fn          hasSufficientGPUmemory
	 * \brief       returns whether the amount of free GPU memory > projectionDataSize() + volumeDataSize()
	 * \param[in]   useLeastGPUmemory whether or not to use the GPU with the least amount of memory
	 * \param[in]   extraCols the number of extra columns that will be needed to be added to the data for processing
	 * \return      returns whether the amount of free GPU memory > projectionDataSize() + volumeDataSize()
	 */
	bool hasSufficientGPUmemory(bool useLeastGPUmemory=false, int extraCols = 0);

	/**
	 * \fn          rowRangeNeededForBackprojection
	 * \brief       calculates the necessary detector rows needed to backproject a selection of volume z-slices
	 * \param[in]   firstSlice the index of the first z-slice of a slab
	 * \param[in]   lastSlice the index of the last z-slice of a slab
	 * \param[in]   rowsNeeded 2-element array where the first and last row indices are saved
	 * \return      returns true if successful, false otherwise
	 */
	bool rowRangeNeededForBackprojection(int firstSlice, int lastSlice, int* rowsNeeded);

	/**
	 * \fn          sliceRangeNeededForProjection
	 * \brief       calculates the necessary z-slices needed to project a selection of detector rows
	 * \param[in]   firstRow the index of the first detector row
	 * \param[in]   lastRow the index of the last detector row
	 * \param[in]   slicesNeeded 2-element array where the first and last z-slice indices are saved
	 * \param[in]   doClip clamps the slicesNeeded values to within 0 and numZ-1 if true
	 * \return      returns true if successful, false otherwise
	 */
	bool sliceRangeNeededForProjection(int firstRow, int lastRow, int* slicesNeeded, bool doClip = true);

	/**
	 * \fn          sliceRangeNeededForProjectionRange
	 * \brief       calculates the necessary z-slices needed to project a selection of projections
	 * \param[in]   firstView the index of the first projection
	 * \param[in]   lastView the index of the last projection
	 * \param[in]   slicesNeeded 2-element array where the first and last z-slice indices are saved
	 * \param[in]   doClip clamps the slicesNeeded values to within 0 and numZ-1 if true
	 * \return      returns true if successful, false otherwise
	 */
	bool sliceRangeNeededForProjectionRange(int firstView, int lastView, int* slicesNeeded, bool doClip = true);

	/**
	 * \fn          viewRangeNeededForBackprojection
	 * \brief       calculates the necessary projection range needed to backproject a selection of volume z-slices
	 * \param[in]   firstSlice the index of the first z-slice
	 * \param[in]   lastSlice the index of the last z-slice
	 * \param[in]   viewsNeeded 2-element array where the first and last view indices are saved
	 * \return      returns true if successful, false otherwise
	 */
	bool viewRangeNeededForBackprojection(int firstSlice, int lastSlice, int* viewsNeeded);

	/**
	 * \fn          removeProjections
	 * \brief       modifies the parameters (phis, sourcePositions, etc) when only a segment of the projections are kept
	 * \param[in]   firstProj the index of the first projection to keep
	 * \param[in]   lastProj the index of the last projection to keep
	 * \return      returns true if successful, false otherwise
	 */
	bool removeProjections(int firstProj, int lastProj);

	// Enums
	enum geometry_list { CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3 };
	enum volumeDimensionOrder_list { XYZ = 0, ZYX = 1 };
	enum detectorType_list { FLAT = 0, CURVED = 1 };
    enum whichProjector_list {SIDDON=0,JOSEPH=1,SEPARABLE_FOOTPRINT=2};

	float get_extraMemoryReserved();

	float get_phi_start();
	float get_phi_end();

private:

	/**
	 * \fn          assign
	 * \brief       makes a deep copy of the given parameter object
	 */
	void assign(const parameters& other);

	/**
	 * \fn          clearModularBeamParameters
	 * \brief       frees all memory of arrays used to specify a modular-beam geometry
	 * \return      returns true if successful, false otherwise
	 */
	bool clearModularBeamParameters();

	float extraMemoryReserved;
	float phi_start;
	float phi_end;
};

#endif
