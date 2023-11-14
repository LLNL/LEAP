////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// parameters c++ class
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "parameters.h"
#include "cuda_utils.h"

using namespace std;

parameters::parameters()
{
	sourcePositions = NULL;
	moduleCenters = NULL;
	rowVectors = NULL;
	colVectors = NULL;
	phis = NULL;
	initialize();
	//setDefaults(1);
}

parameters::parameters(int N)
{
	sourcePositions = NULL;
	moduleCenters = NULL;
	rowVectors = NULL;
	colVectors = NULL;
	phis = NULL;
	initialize();
	//setDefaults(N);
}

void parameters::initialize()
{
	whichGPUs.clear();
	int numGPUs = numberOfGPUs();
	if (numGPUs > 0)
	{
		whichGPU = 0;
		for (int i = 0; i < numGPUs; i++)
			whichGPUs.push_back(i);
	}
	else
		whichGPU = -1;
	whichProjector = SEPARABLE_FOOTPRINT;
	volumeDimensionOrder = ZYX;
	rampID = 2;

	geometry = CONE;
	detectorType = FLAT;
	sod = 0.0;
	sdd = 0.0;
	numCols = 0;
	numRows = 0;
	numAngles = 0;
	pixelWidth = 0.0;
	pixelHeight = 0.0;
	angularRange = 0.0;
	centerCol = 0.0;
	centerRow = 0.0;
	tau = 0.0;
	rFOVspecified = 0.0;

	normalizeConeAndFanCoordinateFunctions = false;

	axisOfSymmetry = 90.0; // must be less than 30 to be activated

	numX = 0;
	numY = 0;
	numZ = 0;
	voxelHeight = 0.0;
	voxelWidth = 0.0;
	offsetX = 0.0;
	offsetY = 0.0;
	offsetZ = 0.0;
}

parameters::parameters(const parameters& other)
{
    sourcePositions = NULL;
    moduleCenters = NULL;
    rowVectors = NULL;
    colVectors = NULL;
    phis = NULL;
    //setDefaults(1);
	initialize();
    assign(other);
}

parameters::~parameters()
{
    clearAll();
}

parameters& parameters::operator = (const parameters& other)
{
    if (this != &other)
        this->assign(other);
    return *this;
}

void parameters::assign(const parameters& other)
{
    this->clearAll();
    
    this->whichGPU = other.whichGPU;
	for (int i = 0; i < int(other.whichGPUs.size()); i++)
		this->whichGPUs.push_back(other.whichGPUs[i]);
    this->whichProjector = other.whichProjector;
	this->rampID = other.rampID;
    this->geometry = other.geometry;
    this->detectorType = other.detectorType;
    this->sod = other.sod;
    this->sdd = other.sdd;
    this->pixelWidth = other.pixelWidth;
    this->pixelHeight = other.pixelHeight;
    this->angularRange = other.angularRange;
    this->numCols = other.numCols;
    this->numRows = other.numRows;
    this->numAngles = other.numAngles;
    this->centerCol = other.centerCol;
    this->centerRow = other.centerRow;
    this->tau = other.tau;
    this->rFOVspecified = other.rFOVspecified;
    this->axisOfSymmetry = other.axisOfSymmetry;
    this->volumeDimensionOrder = other.volumeDimensionOrder;
    this->numX = other.numX;
    this->numY = other.numY;
    this->numZ = other.numZ;
    this->voxelWidth = other.voxelWidth;
    this->voxelHeight = other.voxelHeight;
    this->offsetX = other.offsetX;
    this->offsetY = other.offsetY;
    this->offsetZ = other.offsetZ;

    if (this->phis != NULL)
        delete [] this->phis;
    this->phis = new float[numAngles];
    for (int i = 0; i < numAngles; i++)
        this->phis[i] = other.phis[i];
    
    this->setSourcesAndModules(other.sourcePositions, other.moduleCenters, \
        other.rowVectors, other.colVectors, other.numAngles);
        
}

void parameters::setDefaults(int N)
{
	whichGPUs.clear();
	int numGPUs = numberOfGPUs();
	if (numGPUs > 0)
	{
		whichGPU = 0;
		for (int i = 0; i < numGPUs; i++)
			whichGPUs.push_back(i);
	}
	else
		whichGPU = -1;
    whichProjector = SEPARABLE_FOOTPRINT;
	volumeDimensionOrder = ZYX;
	rampID = 2;

	geometry = CONE;
	detectorType = FLAT;
	sod = 1100.0;
	sdd = 1400.0;
	numCols = 2048 / N;
	numRows = numCols;
	numAngles = int(ceil(1440.0*float(numCols) / 2048.0));
	pixelWidth = 0.2*2048.0 / float(numCols);
	pixelHeight = pixelWidth;
	angularRange = 360.0;
	centerCol = float(numCols - 1) / 2.0;
	centerRow = float(numCols - 1) / 2.0;
	tau = 0.0;
	rFOVspecified = 0.0;

	normalizeConeAndFanCoordinateFunctions = false;

	axisOfSymmetry = 90.0; // must be less than 30 to be activated

    setAngles(); // added by Hyojin
	setDefaultVolumeParameters();
}

float parameters::T_phi()
{
    if (numAngles <= 1 || phis == NULL)
        return 2.0*PI;
	else
		return (phis[numAngles-1] - phis[0]) / float(numAngles-1);
}

float parameters::rFOV()
{
	if (rFOVspecified > 0.0)
		return rFOVspecified;
    else if (geometry == MODULAR)
        return 1.0e16;
    else if (geometry == PARALLEL)
        return min(fabs(u_0()), fabs(pixelWidth*float(numCols-1) + u_0()));
    else if (geometry == FAN || geometry == CONE)
	{
        /*
        double alpha_right = lateral(0);
        double alpha_left = lateral(N_lateral-1);
        if (isFlatPanel == true)
        {
            alpha_right = atan(alpha_right);
            alpha_left = atan(alpha_left);
        }
        //return R_tau*sin(min(fabs(alpha_right+atan(tau/R)), fabs(alpha_left+atan(tau/R))));
        double retVal = R_tau*sin(min(fabs(alpha_right-atan(tau/R)), fabs(alpha_left-atan(tau/R))));

        //sid/sdd * c / sqrt(1+(c/sdd)*(c/sdd))
        //return R*u_max()/sqrt(1.0+u_max()*u_max());
        if (theSCT->dxfov.unknown == false && theSCT->dxfov.value > 0.0)
            retVal = min(retVal, 0.5*theSCT->dxfov.value);
        return retVal;
        //*/
        
        float alpha_right = u_0();
        float alpha_left = pixelWidth*float(numCols-1) + u_0();
        alpha_right = atan(alpha_right/sdd);
        alpha_left = atan(alpha_left/sdd);
        return sod*sin(min(fabs(alpha_right),fabs(alpha_left)));
    }
}

float parameters::furthestFromCenter()
{
	if (numX <= 0 || numY <= 0 || voxelWidth <= 0.0)
		return rFOV();
	float x_max = (numX - 1) * voxelWidth + x_0();
	float y_max = (numY - 1) * voxelWidth + y_0();

	//*
	float temp;
	float retVal = x_0() * x_0() + y_0() * y_0();
	temp = x_max * x_max + y_0() * y_0();
	if (temp > retVal)
		retVal = temp;
	temp = y_max * y_max + x_0() * x_0();
	if (temp > retVal)
		retVal = temp;
	temp = x_max * x_max + y_max * y_max;
	if (temp > retVal)
		retVal = temp;

	return sqrt(retVal);
	//*/
}

bool parameters::voxelSizeWorksForFastSF()
{
	float r = min(furthestFromCenter(), rFOV());
	if (geometry == CONE) // || geometry == FAN)
	{
		//f->T_x / (g->R - (rFOV - 0.25 * f->T_x)) < detectorPixelMultiplier * g->T_lateral
		//voxelWidth < 2.0 * pixelWidth * (sod - rFOV) / sdd

		float largestDetectorWidth = (sod + r) / sdd * pixelWidth;
		float smallestDetectorWidth = (sod - r) / sdd * pixelWidth;

		float largestDetectorHeight = (sod + r) / sdd * pixelHeight;
		float smallestDetectorHeight = (sod - r) / sdd * pixelHeight;
		//printf("%f to %f\n", 0.5*largestDetectorWidth, 2.0*smallestDetectorWidth);
		if (0.5 * largestDetectorWidth <= voxelWidth && voxelWidth <= 2.0 * smallestDetectorWidth && 0.5 * largestDetectorHeight <= voxelHeight && voxelHeight <= 2.0 * smallestDetectorHeight)
		{
			//printf("using SF projector\n");
			return true;
		}
		else
		{
			//printf("using Siddon projector\n");
			return false;
		}
	}
	else if (geometry == FAN)
	{
		float largestDetectorWidth = (sod + r) / sdd * pixelWidth;
		float smallestDetectorWidth = (sod - r) / sdd * pixelWidth;

		if (0.5 * largestDetectorWidth <= voxelWidth && voxelWidth <= 2.0 * smallestDetectorWidth)
		{
			//printf("using SF projector\n");
			return true;
		}
		else
		{
			//printf("using Siddon projector\n");
			return false;
		}
	}
	else //if (geometry == PARALLEL)
	{
		if (0.5 * pixelWidth <= voxelWidth && voxelWidth <= 2.0 * pixelWidth)
			return true;
		else
			return false;
	}
}

bool parameters::useSF()
{
    if (whichProjector == SIDDON || geometry == MODULAR || isSymmetric() == true)
        return false;
    else
    {
		//return true;
		return voxelSizeWorksForFastSF();
    }
}

bool parameters::isSymmetric()
{
	if (numAngles == 1 && fabs(axisOfSymmetry) <= 30.0 && (geometry == CONE || geometry == PARALLEL))
		return true;
	else
		return false;
}

bool parameters::allDefined()
{
	return geometryDefined() & volumeDefined();
}

bool parameters::geometryDefined()
{
	if (geometry != CONE && geometry != PARALLEL && geometry != FAN && geometry != MODULAR)
		return false;
	if (numCols <= 0 || numRows <= 0 || numAngles <= 0 || pixelWidth <= 0.0 || pixelHeight <= 0.0)
		return false;
	if (geometry == MODULAR)
	{
		if (sourcePositions == NULL || moduleCenters == NULL || rowVectors == NULL || colVectors == NULL)
			return false;
	}
	else if (angularRange == 0.0 && phis == NULL)
		return false;
	if (geometry == CONE || geometry == FAN)
	{
		if (sod <= 0.0 || sdd <= sod)
			return false;
	}

	return true;
}

bool parameters::volumeDefined()
{
	if (numX <= 0 || numY <= 0 || numZ <= 0 || voxelWidth <= 0.0 || voxelHeight <= 0.0 || volumeDimensionOrder < 0 || volumeDimensionOrder > 1)
		return false;
	else
	{
		if (geometry == PARALLEL || geometry == FAN)
		{
			if (voxelHeight != pixelHeight)
			{
				//voxelHeight = pixelHeight;
				//printf("Warning: for parallel and fan-beam data volume voxel height must equal detector pixel height, so forcing voxel height to match pixel height!\n");
				printf("Error: for parallel and fan-beam data volume voxel height must equal detector pixel height!\n");
				printf("Please modify either the detector pixel height or the voxel height so they match!\n");
				return false;
			}
			if (numRows != numZ)
			{
				//voxelHeight = pixelHeight;
				//printf("Warning: for parallel and fan-beam data volume voxel height must equal detector pixel height, so forcing voxel height to match pixel height!\n");
				printf("Error: for parallel and fan-beam data numZ == numRows!\n");
				return false;
			}
			offsetZ = 0.0;
			//offsetZ = floor(0.5 + offsetZ / voxelHeight) * voxelHeight;
		}
		if (geometry == MODULAR && voxelWidth != voxelHeight)
		{
			voxelHeight = voxelWidth;
			printf("Warning: for modular-beam data volume voxel height must equal voxel width (voxels must be cubic), so forcing voxel height to match voxel width!\n");
		}
		if (isSymmetric())
		{
			if (numX > 1)
			{
				printf("Error: symmetric objects must specify numX = 1!\n");
				return false;
			}
			if (numY % 2 == 1)
			{
				printf("Error: symmetric objects must specify numY as even !\n");
				return false;
			}
			offsetX = 0.0;
			offsetY = 0.0;
		}
		return true;
	}
}

bool parameters::setDefaultVolumeParameters(float scale)
{
	if (geometryDefined() == false)
		return false;

	// Volume Parameters
	//volumeDimensionOrder = ZYX;
	numX = int(ceil(float(numCols) / scale));
	numY = numX;
	numZ = numRows;
	if (geometry == PARALLEL)
	{
		voxelWidth = pixelWidth * scale;
		voxelHeight = pixelHeight;
	}
	else if (geometry == FAN)
	{
		voxelWidth = sod / sdd * pixelWidth * scale;
		voxelHeight = pixelHeight;
	}
	else
	{
		voxelWidth = sod / sdd * pixelWidth * scale;
		voxelHeight = sod / sdd * pixelHeight * scale;

		numZ = int(ceil(float(numRows) / scale));
	}
	offsetX = 0.0;
	offsetY = 0.0;
	offsetZ = 0.0;

	if (isSymmetric())
	{
		numX = 1;
		offsetX = 0.0;

		offsetY = 0.0;
		if (numY % 2 == 1)
			numY += 1;
	}

	return true;
}

void parameters::clearAll()
{
	whichGPUs.clear();
	if (phis != NULL)
		delete[] phis;
	phis = NULL;

	clearModularBeamParameters();
}

bool parameters::clearModularBeamParameters()
{
	if (sourcePositions != NULL)
		delete[] sourcePositions;
	sourcePositions = NULL;
	if (moduleCenters != NULL)
		delete[] moduleCenters;
	moduleCenters = NULL;
	if (rowVectors != NULL)
		delete[] rowVectors;
	rowVectors = NULL;
	if (colVectors != NULL)
		delete[] colVectors;
	colVectors = NULL;
	return true;
}

void parameters::printAll()
{
	printf("\n");

	if (geometry == CONE)
		printf("======== CT Cone-Beam Geometry ========\n");
	else if (geometry == PARALLEL)
		printf("======== CT Parallel-Beam Geometry ========\n");
	else if (geometry == FAN)
		printf("======== CT Fan-Beam Geometry ========\n");
	else if (geometry == MODULAR)
		printf("======== CT Modular-Beam Geometry ========\n");
	printf("number of angles: %d\n", numAngles);
	printf("number of detector elements: %d x %d\n", numRows, numCols);
	if (phis != NULL && numAngles >= 2)
		printf("angular range: %f degrees\n", angularRange);
		//printf("angular range: %f degrees\n", 180.0 / PI * ((phis[numAngles - 1] - phis[0]) + 0.5 * (phis[numAngles - 1] - phis[numAngles - 2]) + 0.5 * (phis[1] - phis[0])));
	printf("detector pixel size: %f mm x %f mm\n", pixelHeight, pixelWidth);
	printf("center detector pixel: %f, %f\n", centerRow, centerCol);
	if (geometry == CONE || geometry == FAN)
	{
		printf("sod = %f mm\n", sod);
		printf("sdd = %f mm\n", sdd);
	}
	printf("\n");

	printf("======== CT Volume ========\n");
	printf("number of voxels: %d x %d x %d\n", numX, numY, numZ);
	printf("voxel size: %f mm x %f mm x %f mm\n", voxelWidth, voxelWidth, voxelHeight);
	if (offsetX != 0.0 || offsetY != 0.0 || offsetZ != 0.0)
		printf("volume offset: %f mm, %f mm, %f mm\n", offsetX, offsetY, offsetZ);
	if (isSymmetric())
		printf("axis of symmetry = %f degrees\n", axisOfSymmetry);
	//printf("x_0 = %f, y_0 = %f, z_0 = %f\n", x_0(), y_0(), z_0());

	printf("\n");
}

bool parameters::setToZero(float* data, int N)
{
	if (data != NULL && N > 0)
	{
		for (int i = 0; i < N; i++)
			data[i] = 0.0;
		return true;
	}
	else
		return false;
}

bool parameters::windowFOV(float* f)
{
	if (f == NULL)
		return false;
	else
	{
		float rFOVsq = rFOV()*rFOV();
		for (int ix = 0; ix < numX; ix++)
		{
			float x = ix * voxelWidth + x_0();
			for (int iy = 0; iy < numY; iy++)
			{
				float y = iy * voxelWidth + y_0();
				if (x*x + y * y > rFOVsq)
				{
					float* zLine = &f[ix*numY*numZ + iy*numZ];
					for (int iz = 0; iz < numZ; iz++)
						zLine[iz] = 0.0;
				}
			}
		}
		return true;
	}
}

bool parameters::setAngles()
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;
	if (numAngles <= 0 || angularRange == 0.0)
		return false;
	else
	{
		phis = new float[numAngles];
		for (int i = 0; i < numAngles; i++)
			phis[i] = float(i)*angularRange*(PI / 180.0) / float(numAngles) - 0.5*PI;
		return true;
	}
}

bool parameters::setAngles(float* phis_new, int numAngles_new)
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;
	if (phis_new == NULL || numAngles_new <= 0)
		return false;
	else
	{
		numAngles = numAngles_new;
		phis = new float[numAngles];
		for (int i = 0; i < numAngles; i++)
			phis[i] = phis_new[i] * PI / 180.0 - 0.5*PI;

		if (numAngles >= 2)
			angularRange = (fabs(phis_new[numAngles - 1] - phis_new[0]) + 0.5 * fabs(phis_new[numAngles - 1] - phis_new[numAngles - 2]) + 0.5 * fabs(phis_new[1] - phis_new[0]));
		else
			angularRange = 0.0;

		return true;
	}
}

bool parameters::setSourcesAndModules(float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in, int numPairs)
{
	clearModularBeamParameters();
	if (sourcePositions_in == NULL || moduleCenters_in == NULL || rowVectors_in == NULL || colVectors_in == NULL || numPairs <= 0)
		return false;
	else
	{
		numAngles = numPairs;
		sourcePositions = new float[3 * numPairs];
		moduleCenters = new float[3 * numPairs];
		rowVectors = new float[3 * numPairs];
		colVectors = new float[3 * numPairs];
		for (int i = 0; i < 3 * numPairs; i++)
		{
			sourcePositions[i] = sourcePositions_in[i];
			moduleCenters[i] = moduleCenters_in[i];
			rowVectors[i] = rowVectors_in[i];
			colVectors[i] = colVectors_in[i];
		}

		return true;
	}
}

float parameters::u_0()
{
	return -centerCol * pixelWidth;
}

float parameters::v_0()
{
	return -centerRow * pixelHeight;
}

float parameters::u(int i)
{
	if (normalizeConeAndFanCoordinateFunctions == true && (geometry == CONE || geometry == FAN))
		return (i * pixelWidth + u_0()) / sdd;
	else
		return i * pixelWidth + u_0();
}

float parameters::v(int i)
{
	if (normalizeConeAndFanCoordinateFunctions == true && (geometry == CONE || geometry == FAN))
		return (i * pixelHeight + v_0()) / sdd;
	else
		return i * pixelHeight + v_0();
}

float parameters::pixelWidth_normalized()
{
	if (geometry == PARALLEL)
		return pixelWidth;
	else
		return pixelWidth / sdd;
}

float parameters::x_0()
{
	return offsetX - 0.5*float(numX - 1)*voxelWidth;
}

float parameters::y_0()
{
	return offsetY - 0.5*float(numY - 1)*voxelWidth;
}

float parameters::z_0()
{
	//return offsetZ - 0.5*float(numZ - 1)*voxelHeight;
	if (geometry == PARALLEL || geometry == FAN)
		return offsetZ - centerRow * (pixelHeight / voxelHeight) * voxelHeight;
	else if (geometry == MODULAR)
		return offsetZ - 0.5*float(numZ-1) * voxelHeight;
	else
		return offsetZ - centerRow * ((sod / sdd * pixelHeight) / voxelHeight) * voxelHeight;
}

float parameters::z_samples(int iz)
{
	return iz * voxelHeight + z_0();
}

bool parameters::anglesAreEquispaced()
{
	if (phis == NULL || numAngles < 2)
		return true;
	else
	{
		float firstSpacing = phis[1] - phis[0];
		for (int i = 1; i < numAngles-1; i++)
		{
			float curSpacing = phis[i + 1] - phis[i];
			if (fabs(curSpacing - firstSpacing) > 1.0e-8)
				return false;
		}
		return true;
	}
}

float parameters::projectionDataSize()
{
	return float(4.0 * double(numAngles) * double(numRows) * double(numCols) / pow(2.0, 30.0));
}

float parameters::volumeDataSize()
{
	return float(4.0 * double(numX) * double(numY) * double(numZ) / pow(2.0, 30.0));
}

bool parameters::rowRangeNeededForReconstruction(int firstSlice, int lastSlice , int* rowsNeeded)
{
	if (rowsNeeded == NULL || firstSlice > lastSlice)
		return false;

	if (geometry == PARALLEL || geometry == FAN)
	{
		rowsNeeded[0] = max(0, firstSlice);
		rowsNeeded[1] = min(numRows-1, lastSlice);
	}
	else if (geometry == MODULAR)
	{
		rowsNeeded[0] = 0;
		rowsNeeded[1] = numRows - 1;
	}
	else
	{
		float z_lo = float(firstSlice) * voxelHeight + z_0() - 0.5 * voxelHeight;
		float z_hi = float(lastSlice) * voxelHeight + z_0() + 0.5 * voxelHeight;

		// v = z / (R - <x, theta>)
		// v = z / (R - rFOV())
		float T_v = pixelHeight / sdd;
		float v_denom_min = sod - rFOV() - voxelWidth;
		float v_denom_max = sod + rFOV() + voxelWidth;
		float v_lo = min(z_lo / v_denom_min, z_lo / v_denom_max) - 0.5*pixelHeight/sdd;
		float v_hi = max(z_hi / v_denom_min, z_hi / v_denom_max) + 0.5*pixelHeight/sdd;

		rowsNeeded[0] = max(0, int(floor((v_lo - v_0() / sdd) / T_v)));
		rowsNeeded[1] = min(numRows-1, int(ceil((v_hi - v_0() / sdd) / T_v)));
	}
	return true;
}

bool parameters::sliceRangeNeededForProjection(int firstRow, int lastRow, int* slicesNeeded)
{
	if (slicesNeeded == NULL || firstRow > lastRow)
		return false;

	if (geometry == PARALLEL || geometry == FAN)
	{
		slicesNeeded[0] = max(0, firstRow);
		slicesNeeded[1] = min(numZ - 1, lastRow);
	}
	else if (geometry == MODULAR)
	{
		slicesNeeded[0] = 0;
		slicesNeeded[1] = numZ - 1;
	}
	else
	{
		float v_lo = (float(firstRow) * pixelHeight + v_0() - 0.5 * pixelHeight) / sdd;
		float v_hi = (float(lastRow) * pixelHeight + v_0() + 0.5 * pixelHeight) / sdd;

		// v = z / (R - <x, theta>)
		// v = z / (R - rFOV())
		float T_z = voxelHeight;
		float v_denom_min = sod - rFOV() - voxelWidth;
		float v_denom_max = sod + rFOV() + voxelWidth;
		float z_lo = min(v_lo * v_denom_min, v_lo * v_denom_max) - 0.5 * voxelHeight;
		float z_hi = max(v_hi * v_denom_min, v_hi * v_denom_max) + 0.5 * voxelHeight;

		slicesNeeded[0] = max(0, int(floor((z_lo - z_0()) / T_z)));
		slicesNeeded[1] = min(numZ - 1, int(ceil((z_hi - z_0()) / T_z)));
	}
	return true;
}
