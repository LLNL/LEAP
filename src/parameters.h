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

#ifndef PI
    #define PI 3.1415926535897932385
#endif

class parameters
{
public:
	parameters();
	parameters(int N);
    parameters(const parameters& other);
	~parameters();
    parameters& operator = (const parameters& other);

    void assign(const parameters& other);
    void setDefaults(int N);
	void printAll();
	void clearAll();
	bool allDefined();
	bool geometryDefined();
	bool volumeDefined();
	bool setDefaultVolumeParameters(float scale = 1.0);
	bool setAngles(float*, int);
	bool setAngles();

	bool anglesAreEquispaced();

	float u_0();
	float v_0();
	float x_0();
	float y_0();
	float z_0();

	float furthestFromCenter();

	float pixelWidth_normalized();
	float u(int);
	float v(int);
	bool normalizeConeAndFanCoordinateFunctions;

	int whichGPU;
    int whichProjector;

	// Scanner Parameters
	int geometry;
	int detectorType;
	float sod, sdd;
	float pixelWidth, pixelHeight, angularRange;
	int numCols, numRows, numAngles;
	float centerCol, centerRow;
	float* phis;
	float tau;
	float rFOVspecified;
    
    float T_phi();
    float rFOV();

	float axisOfSymmetry;
	bool isSymmetric();
    bool useSF();
	bool setToZero(float*, int);
	bool windowFOV(float*);

	//float smallestVoxelForFastSF();
	//float largestVoxelForFastSF();
	bool voxelSizeWorksForFastSF();

	float projectionDataSize();
	float volumeDataSize();

	// Volume Parameters
	int volumeDimensionOrder;
	int numX, numY, numZ;
	float voxelWidth, voxelHeight;
	float offsetX, offsetY, offsetZ;

	// Modular-Beam Parameters
	bool clearModularBeamParameters();
	bool setSourcesAndModules(float*, float*, float*, float*, int);
	float* sourcePositions;
	float* moduleCenters;
	float* rowVectors;
	float* colVectors;

	enum geometry_list { CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3 };
	enum volumeDimensionOrder_list { XYZ = 0, ZYX = 1 };
	enum detectorType_list { FLAT = 0, CURVED = 1 };
    enum whichProjector_list {SIDDON=0,JOSEPH=1,SEPARABLE_FOOTPRINT=2};
};

#endif
