////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// parameters c++ class which defines all the CT geometry and CT volume parameters
// it also has functions that help enable splitting data into chunks
// for sequential or parallel processing
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include "parameters.h"
#include "cpu_utils.h"
#ifndef __USE_CPU
#include "cuda_utils.h"
#endif

using namespace std;

parameters::parameters()
{
	sourcePositions = NULL;
	moduleCenters = NULL;
	rowVectors = NULL;
	colVectors = NULL;
	phis = NULL;
	mu = NULL;
	initialize();
}

void parameters::initialize()
{
	whichGPUs.clear();
#ifndef __USE_CPU
	int numGPUs = numberOfGPUs();
#else
	int numGPUs = 0;
#endif
	if (numGPUs > 0)
	{
		whichGPU = 0;
		for (int i = 0; i < numGPUs; i++)
			whichGPUs.push_back(i);
	}
	else
		whichGPU = -1;
	whichProjector = SEPARABLE_FOOTPRINT;
	doWeightedBackprojection = false;
	doExtrapolation = false;
	volumeDimensionOrder = ZYX;
	rampID = 2;
	FBPlowpass = 1.0;
	chunkingMemorySizeThreshold = float(0.1);
	colShiftFromFilter = 0.0;
	rowShiftFromFilter = 0.0;
	offsetScan = false;
	truncatedScan = false;
	inconsistencyReconstruction = false;
	lambdaTomography = false;
	numTVneighbors = 26;

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
	helicalPitch = 0.0;
	z_source_offset = 0.0;
	rFOVspecified = 0.0;

	muCoeff = 0.0;
	muRadius = 0.0;

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

	extraMemoryReserved = 0.25;
	phi_start = 0.0;
	phi_end = 0.0;
}

float parameters::get_extraMemoryReserved()
{
	return extraMemoryReserved;
}

float parameters::get_phi_start()
{
	return phi_start;
}

float parameters::get_phi_end()
{
	return phi_end;
}

parameters::parameters(const parameters& other)
{
    sourcePositions = NULL;
    moduleCenters = NULL;
    rowVectors = NULL;
    colVectors = NULL;
    phis = NULL;
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
	this->doWeightedBackprojection = other.doWeightedBackprojection;
	this->doExtrapolation = other.doExtrapolation;
	this->rampID = other.rampID;
	this->FBPlowpass = other.FBPlowpass;
	this->chunkingMemorySizeThreshold = other.chunkingMemorySizeThreshold;
	this->colShiftFromFilter = other.colShiftFromFilter;
	this->rowShiftFromFilter = other.rowShiftFromFilter;
	this->offsetScan = other.offsetScan;
	this->truncatedScan = other.truncatedScan;
	this->inconsistencyReconstruction = other.inconsistencyReconstruction;
	this->lambdaTomography = other.lambdaTomography;
	this->numTVneighbors = other.numTVneighbors;
	this->mu = other.mu;
	this->muCoeff = other.muCoeff;
	this->muRadius = other.muRadius;
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
	this->helicalPitch = other.helicalPitch;
	this->z_source_offset = other.z_source_offset;
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

	this->phi_start = other.phi_start;
	this->phi_end = other.phi_end;

    if (this->phis != NULL)
        delete [] this->phis;
	if (other.phis != NULL)
	{
		this->phis = new float[numAngles];
		for (int i = 0; i < numAngles; i++)
			this->phis[i] = other.phis[i];
	}
    
	if (other.sourcePositions != NULL)
	{
		this->set_sourcesAndModules(other.sourcePositions, other.moduleCenters, \
			other.rowVectors, other.colVectors, other.numAngles);
	}
}

float parameters::T_phi()
{
    if (numAngles <= 1 || phis == NULL)
        return float(2.0*PI);
	else
		return (phis[numAngles-1] - phis[0]) / float(numAngles-1);
}

float parameters::min_T_phi()
{
	if (numAngles <= 1 || phis == NULL)
		return float(2.0 * PI);
	else
	{
		double retVal = fabs(phis[1] - phis[0]);
		for (int i = 1; i < numAngles - 1; i++)
			retVal = std::min(retVal, double(fabs(phis[i + 1] - phis[i])));
		return float(retVal);
	}
}

float parameters::phi_inv(float angle)
{
	if (phis == NULL)
		return -1.0;
	else if (numAngles == 1)
		return 0.0;
	else if (angle <= min(phis[numAngles - 1], phis[0]))
	{
		if (phis[numAngles - 1] < phis[0])
			return float(numAngles - 1);
		else
			return 0;
	}
	else if (angle >= max(phis[numAngles - 1], phis[0]))
	{
		if (phis[numAngles - 1] > phis[0])
			return float(numAngles - 1);
		else
			return 0;
	}
	else
	{
		if (T_phi() > 0.0)
		{
			//phis[0] < angle
			for (int i = 1; i < numAngles; i++)
			{
				if (angle <= phis[i])
				{
					// phis[i-1] < angle <= phis[i]
					float d = (angle - phis[i - 1]) / (phis[i] - phis[i - 1]);
					return float(i - 1) + d;
				}
			}
			return float(numAngles);
		}
		else
		{
			//phis[numAngles-1] < angle
			for (int i = numAngles-2; i >= 0; i--)
			{
				if (angle <= phis[i])
				{
					// phis[i+1] < angle <= phis[i]
					float d = (angle - phis[i + 1]) / (phis[i] - phis[i + 1]);
					return float(i+1) - d;
				}
			}
			return 0.0;
		}
	}
}

float parameters::rFOV()
{
	if (rFOVspecified > 0.0)
	{
		float r_max = furthestFromCenter();
		if (r_max > 0.0)
			return min(r_max, rFOVspecified); // this just helps with chunking calculations
		else
			return rFOVspecified;
	}
	else if (geometry == MODULAR)
	{
		//return 1.0e16;
		return float(furthestFromCenter()+0.5*voxelWidth);
	}
	else if (geometry == PARALLEL || geometry == CONE_PARALLEL)
	{
		if (offsetScan)
			return float(max(fabs(u_0()), fabs(pixelWidth * float(numCols - 1) + u_0())));
		else
			return float(min(fabs(u_0()), fabs(pixelWidth * float(numCols - 1) + u_0())));
	}
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

		//float R_tau = sqrt(sod * sod + tau * tau);
		float alpha_right = u_0();
		float alpha_left = pixelWidth * float(numCols - 1) + u_0();
		//float alpha_left = u(numCols - 1);
		if (detectorType == FLAT)
		{
			alpha_right = atan(alpha_right / sdd);
			alpha_left = atan(alpha_left / sdd);
		}
		if (offsetScan)
			return sod * sin(max(fabs(alpha_right - float(atan(tau / sod))), fabs(alpha_left - float(atan(tau / sod)))));
		else
			return sod * sin(min(fabs(alpha_right - float(atan(tau / sod))), fabs(alpha_left - float(atan(tau / sod)))));
	}
	else
		return float(1.0e16);
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

bool parameters::voxelSizeWorksForFastSF(int whichDirection)
{
	float r = min(furthestFromCenter(), rFOV());
	if (geometry == CONE || geometry == MODULAR) // || geometry == FAN)
	{
		//f->T_x / (g->R - (rFOV - 0.25 * f->T_x)) < detectorPixelMultiplier * g->T_lateral
		//voxelWidth < 2.0 * pixelWidth * (sod - rFOV) / sdd

		float largestDetectorWidth = (sod + r) / sdd * pixelWidth;
		float smallestDetectorWidth = (sod - r) / sdd * pixelWidth;

		float largestDetectorHeight = (sod + r) / sdd * pixelHeight;
		float smallestDetectorHeight = (sod - r) / sdd * pixelHeight;
		//printf("%f to %f\n", 0.5*largestDetectorWidth, 2.0*smallestDetectorWidth);
		if (whichDirection == -1) // backprojection
		{
			if (voxelWidth > 2.0 * smallestDetectorWidth || voxelHeight > 2.0 * smallestDetectorHeight)
				return false;
			else
				return true;
		}
		else if (whichDirection == 1) // forward projection
		{
			if (0.5 * largestDetectorWidth > voxelWidth || 0.5 * largestDetectorHeight > voxelHeight)
				return false;
			else
				return true;
		}
		else //if (whichDirection == 0)
		{
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
	}
	else if (geometry == CONE_PARALLEL)
	{
		float largestDetectorWidth = pixelWidth;
		float smallestDetectorWidth = pixelWidth;

		float largestDetectorHeight = (sod + r) / sdd * pixelHeight;
		float smallestDetectorHeight = (sod - r) / sdd * pixelHeight;
		//printf("%f to %f\n", 0.5*largestDetectorWidth, 2.0*smallestDetectorWidth);
		if (whichDirection == -1) // backprojection
		{
			if (voxelWidth > 2.0 * smallestDetectorWidth || voxelHeight > 2.0 * smallestDetectorHeight)
				return false;
			else
				return true;
		}
		else if (whichDirection == 1) // forward projection
		{
			if (0.5 * sqrt(2.0) * largestDetectorWidth > voxelWidth || 0.5 * largestDetectorHeight > voxelHeight)
				return false;
			else
				return true;
		}
		else //if (whichDirection == 0)
		{
			if (0.5 * sqrt(2.0) * largestDetectorWidth <= voxelWidth && voxelWidth <= 2.0 * smallestDetectorWidth && 0.5 * largestDetectorHeight <= voxelHeight && voxelHeight <= 2.0 * smallestDetectorHeight)
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
	}
	else if (geometry == FAN)
	{
		float largestDetectorWidth = (sod + r) / sdd * pixelWidth;
		float smallestDetectorWidth = (sod - r) / sdd * pixelWidth;

		if (whichDirection == -1) // backprojection
		{
			if (voxelWidth > 2.0 * smallestDetectorWidth)
				return false;
			else
				return true;
		}
		else if (whichDirection == 1) // projection
		{
			if (0.5 * largestDetectorWidth > voxelWidth)
				return false;
			else
				return true;
		}
		else
		{
			if (0.5 * largestDetectorWidth <= voxelWidth && voxelWidth <= 2.0 * smallestDetectorWidth)
				return true;
			else
				return false;
		}
	}
	else //if (geometry == PARALLEL)
	{
		if (whichDirection == -1) // backprojection
		{
			if (voxelWidth > 2.0 * pixelWidth)
				return false;
			else
				return true;
		}
		else if (whichDirection == 1) // projection
		{
			if (0.5*sqrt(2.0) * pixelWidth > voxelWidth)
				return false;
			else
				return true;
		}
		else
		{
			if (0.5 * sqrt(2.0) * pixelWidth <= voxelWidth && voxelWidth <= 2.0 * pixelWidth)
				return true;
			else
				return false;
		}
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

bool parameters::muSpecified()
{
	if (mu != NULL || (muCoeff != 0.0 && muRadius > 0.0))
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
	if (geometry != CONE && geometry != PARALLEL && geometry != FAN && geometry != MODULAR && geometry != CONE_PARALLEL)
	{
		printf("Error: CT geometry type not defined!\n");
		return false;
	}
	if (numCols <= 0 || numRows <= 0 || numAngles <= 0 || pixelWidth <= 0.0 || pixelHeight <= 0.0)
	{
		printf("Error: detector pixel sizes and number of data elements must be positive\n");
		return false;
	}
	if (geometry == MODULAR)
	{
		if (sourcePositions == NULL || moduleCenters == NULL || rowVectors == NULL || colVectors == NULL)
		{
			printf("Error: sourcePositions, moduleCenters, rowVectors, and colVectors must be defined for modular-beam geometries\n");
			return false;
		}
	}
	else if (angularRange == 0.0 && phis == NULL)
	{
		printf("Error: projection angles not defined\n");
		return false;
	}
	if (geometry == CONE || geometry == FAN || geometry == CONE_PARALLEL)
	{
		if (sod <= 0.0 || sdd <= sod)
		{
			printf("Error: sod and sdd must be positive for fan- and cone-beam geometries\n");
			return false;
		}
	}
	if (detectorType == CURVED && geometry != CONE)
	{
		printf("Error: curved detector only defined for cone-beam geometries\n");
		return false;
	}

	return true;
}

bool parameters::volumeDefined()
{
	if (numX <= 0 || numY <= 0 || numZ <= 0 || voxelWidth <= 0.0 || voxelHeight <= 0.0 || volumeDimensionOrder < 0 || volumeDimensionOrder > 1)
	{
		printf("numZ = %d voxelHeight = %f\n", numZ, voxelHeight);
		printf("Error: volume voxel sizes and number of data elements must be positive\n");
		return false;
	}
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
			if (modularbeamIsAxiallyAligned() == false || voxelSizeWorksForFastSF() == false)
			{
				voxelHeight = voxelWidth;
				printf("Warning: for (non axially-aligned) modular-beam data volume voxel height must equal voxel width (voxels must be cubic), so forcing voxel height to match voxel width!\n");
			}
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

			// Shift offsetY so that y=0 occurs at the boundary of two pixels
			int n = 2 * int(floor(0.5 + y_0() / voxelWidth - 0.5)) + 1;
			offsetY = 0.5 * voxelWidth * float(numY + n - 1);
		}
		return true;
	}
}

float parameters::default_voxelWidth()
{
	if (geometry == PARALLEL || geometry == CONE_PARALLEL)
		return pixelWidth;
	else
		return sod / sdd * pixelWidth;
}

float parameters::default_voxelHeight()
{
	if (geometry == PARALLEL || geometry == FAN)
		return pixelHeight;
	else
		return sod / sdd * pixelHeight;
}

bool parameters::set_default_volume(float scale)
{
	if (geometryDefined() == false)
		return false;

	// Volume Parameters
	//volumeDimensionOrder = ZYX;
	numX = int(ceil(float(numCols) / scale));
	numY = numX;

	voxelWidth = default_voxelWidth() * scale;

	if ((geometry == CONE && numRows > 1) || geometry == MODULAR || (geometry == CONE_PARALLEL && numRows > 1))
	{
		voxelHeight = default_voxelHeight() * scale;
		numZ = int(ceil(float(numRows) / scale));
	}
	else
	{
		voxelHeight = default_voxelHeight();
		numZ = numRows;
	}
	offsetX = 0.0;
	offsetY = 0.0;
	offsetZ = 0.0;

	if ((geometry == CONE || geometry == CONE_PARALLEL) && helicalPitch != 0.0)
	{
		int minSlices = int(sod / sdd * float(numRows) * pixelHeight / voxelHeight + 0.5);
		if (fabs(angularRange) <= 180.0)
			numZ = minSlices;
		else if (fabs(angularRange) <= 360.0)
		{
			numZ = int(ceil((sod / sdd * (numRows - 1) * pixelHeight + fabs(helicalPitch) * (fabs(angularRange) * PI / 180.0 - PI)) / voxelHeight));
			numZ = max(minSlices, numZ);
		}
		else
		{
			numZ = int(ceil(fabs(helicalPitch) * (fabs(angularRange) * PI / 180.0 - PI) / voxelHeight));
			numZ = max(minSlices, numZ);
		}
	}

	if (geometry == CONE || geometry == CONE_PARALLEL)
	{
		if (helicalPitch != 0.0)
		{
			offsetZ = (0.5 * float(numRows - 1) - centerRow) * (sod / sdd * pixelHeight);
		}
		else
		{
			// want: z_0 = -centerRow * (sod / sdd * pixelHeight)
			// have: z_0 = offsetZ - 0.5 * float(numZ - 1) * voxelHeight
			//offsetZ = 0.5 * float(numZ - 1) * voxelHeight - centerRow * (sod / sdd * pixelHeight);
			if (scale == 1.0)
				offsetZ = (0.5 * float(numZ - 1) - centerRow) * voxelHeight;
			else
				offsetZ = (0.5 * float(numRows - 1) - centerRow) * (sod / sdd * pixelHeight);
		}
		/* old specification of z_0
		float rzref = -centerRow * (sod / sdd * pixelHeight);
		if (helicalPitch != 0.0)
		{
			rzref = (0.5 * float(numRows - 1) - centerRow) * (sod / sdd * pixelHeight) / voxelHeight;
			rzref -= 0.5 * float(numZ - 1);
			rzref *= voxelHeight;
		}
		return offsetZ + rzref;
		//*/
		//return offsetZ - 0.5 * float(numZ - 1) * voxelHeight; // current specification of z_0
	}

	if (offsetScan)
	{
		numX = 2 * int(ceil(rFOV() / voxelWidth));
		numY = numX;
	}

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
	{
		if (detectorType == CURVED)
			printf("======== CT Cone-Beam Geometry (curved detector) ========\n");
		else
			printf("======== CT Cone-Beam Geometry ========\n");
	}
	if (geometry == CONE_PARALLEL)
		printf("======== CT Cone-Parallel Geometry ========\n");
	else if (geometry == PARALLEL)
		printf("======== CT Parallel-Beam Geometry ========\n");
	else if (geometry == FAN)
		printf("======== CT Fan-Beam Geometry ========\n");
	else if (geometry == MODULAR)
	{
		if (modularbeamIsAxiallyAligned())
			printf("======== CT Modular-Beam Geometry (axially aligned) ========\n");
		else
			printf("======== CT Modular-Beam Geometry ========\n");
	}
	printf("number of angles: %d\n", numAngles);
	printf("number of detector elements (rows, cols): %d x %d\n", numRows, numCols);
	if (phis != NULL && numAngles >= 2)
	{
		if (T_phi() < 0.0)
			printf("angular range: -%f degrees\n", angularRange);
		else
			printf("angular range: %f degrees\n", angularRange);
		//printf("angular range: %f degrees\n", 180.0 / PI * ((phis[numAngles - 1] - phis[0]) + 0.5 * (phis[numAngles - 1] - phis[numAngles - 2]) + 0.5 * (phis[1] - phis[0])));
	}
	printf("detector pixel size: %f mm x %f mm\n", pixelHeight, pixelWidth);
	printf("center detector pixel: %f, %f\n", centerRow, centerCol);
	if (geometry == CONE || geometry == FAN || geometry == CONE_PARALLEL)
	{
		printf("sod = %f mm\n", sod);
		printf("sdd = %f mm\n", sdd);
		if (tau != 0.0)
			printf("tau = %f mm\n", tau);
	}
	else if (geometry == MODULAR)
	{
		printf("mean sod = %f mm\n", sod);
		printf("mean sdd = %f mm\n", sdd);
	}
	if ((geometry == CONE || geometry == CONE_PARALLEL) && helicalPitch != 0.0)
	{
		printf("helicalPitch = %f (mm/radian)\n", helicalPitch);
		printf("normalized helicalPitch = %f\n", normalizedHelicalPitch());
	}
	printf("\n");

	printf("======== CT Volume ========\n");
	printf("number of voxels (x, y, z): %d x %d x %d\n", numX, numY, numZ);
	printf("voxel size: %f mm x %f mm x %f mm\n", voxelWidth, voxelWidth, voxelHeight);
	if (offsetX != 0.0 || offsetY != 0.0 || offsetZ != 0.0)
		printf("volume offset: %f mm, %f mm, %f mm\n", offsetX, offsetY, offsetZ);
	printf("FOV: [%f, %f] x [%f, %f] x [%f, %f]\n", x_0()-0.5*voxelWidth, (numX-1)*voxelWidth+x_0()+0.5*voxelWidth, y_0()- 0.5 * voxelWidth, (numY - 1) * voxelWidth + y_0()+ 0.5 * voxelWidth, z_0()- 0.5 * voxelHeight, (numZ - 1) * voxelHeight + z_0()+ 0.5 * voxelHeight);
	if (isSymmetric())
		printf("axis of symmetry = %f degrees\n", axisOfSymmetry);
	//printf("x_0 = %f, y_0 = %f, z_0 = %f\n", x_0(), y_0(), z_0());
	printf("\n");

	printf("======== Processing Settings ========\n");
	if (whichGPU < 0)
		printf("%d-core CPU processing\n", int(omp_get_num_procs()));
	else
	{
#ifndef __USE_CPU
		if (whichGPUs.size() == 1)
			printf("GPU processing on device %d\n", whichGPU);
		else
		{
			printf("GPU processing on devices ");
			for (int i = 0; i < int(whichGPUs.size())-1; i++)
				printf("%d, ", whichGPUs[i]);
			printf("%d\n", whichGPUs[whichGPUs.size() - 1]);
		}
		printf("GPU with least amount of memory: %f GB\n", getAvailableGPUmemory(whichGPUs));
#endif
	}

	/*
	if (geometry == MODULAR)
	{
		for (int i = 0; i < numAngles; i++)
		{
			printf("projection %d\n", i);
			printf("  sourcePosition = (%f, %f, %f)\n", sourcePositions[i*3+0], sourcePositions[i * 3 + 1], sourcePositions[i * 3 + 2]);
			printf("  moduleCenter = (%f, %f, %f)\n", moduleCenters[i * 3 + 0], moduleCenters[i * 3 + 1], moduleCenters[i * 3 + 2]);
			printf("  rowVector = (%f, %f, %f)\n", rowVectors[i * 3 + 0], rowVectors[i * 3 + 1], rowVectors[i * 3 + 2]);
			printf("  colVector = (%f, %f, %f)\n", colVectors[i * 3 + 0], colVectors[i * 3 + 1], colVectors[i * 3 + 2]);
		}
	}
	//*/
	/*
	if (phis != NULL)
	{
		for (int i = 0; i < numAngles; i++)
			printf("%f ", phis[i] * 180.0 / PI);
		printf("\n");
	}
	//*/

	printf("\n");
}

float* parameters::setToConstant(float* data, uint64 N, float val)
{
	if (N <= 0)
		return NULL;
	if (data == NULL)
		data = (float*)malloc(sizeof(float) * N);

	for (uint64 i = 0; i < N; i++)
		data[i] = val;

	return data;
}

float* parameters::setToZero(float* data, uint64 N)
{
	return setToConstant(data, N, 0.0);
}

bool parameters::set_angles()
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
		//phi_start = phis[0];
		//phi_end = phis[numAngles - 1];
		phi_start = min(phis[0], phis[numAngles - 1]);
		phi_end = max(phis[0], phis[numAngles - 1]);
		return true;
	}
}

bool parameters::phaseShift(float radians)
{
	if (phis == NULL || numAngles <= 0)
		return false;
	else
	{
		if (fabs(radians) > 2.0 * PI)
			printf("Warning: phaseShift argument is given in radians\n");
		for (int i = 0; i < numAngles; i++)
			phis[i] += radians;
		phi_start = min(phis[0], phis[numAngles - 1]);
		phi_end = max(phis[0], phis[numAngles - 1]);
		return true;
	}
}

bool parameters::set_angles(float* phis_new, int numAngles_new)
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
		phi_start = min(phis[0], phis[numAngles - 1]);
		phi_end = max(phis[0], phis[numAngles - 1]);

		if (numAngles >= 2)
			angularRange = (fabs(phis_new[numAngles - 1] - phis_new[0]) + 0.5 * fabs(phis_new[numAngles - 1] - phis_new[numAngles - 2]) + 0.5 * fabs(phis_new[1] - phis_new[0]));
		else
			angularRange = 0.0;

		return true;
	}
}

bool parameters::get_angles(float* phis_out)
{
	if (phis_out == NULL || phis == NULL || numAngles <= 0)
		return false;
	for (int i = 0; i < numAngles; i++)
		phis_out[i] = (phis[i]+0.5*PI)*180.0/PI;
	return true;
}

bool parameters::set_sod(float sod_in)
{
	if (sod_in >= 0.0)// && (sod_in < sdd || sdd == 0.0))
	{
		sod = sod_in;
		return true;
	}
	else
	{
		printf("Error: invalid value!\n");
		return false;
	}
}

bool parameters::set_sdd(float sdd_in)
{
	if (sdd_in >= 0.0)// && sdd_in > sod)
	{
		sdd = sdd_in;
		return true;
	}
	else
	{
		printf("Error: invalid value!\n");
		return false;
	}
}

bool parameters::set_sourcesAndModules(float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in, int numPairs)
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;
	clearModularBeamParameters();
	if (sourcePositions_in == NULL || moduleCenters_in == NULL || rowVectors_in == NULL || colVectors_in == NULL || numPairs <= 0)
	{
		printf("Error: no source/detector geometry parameters specified\n");
		return false;
	}
	else
	{
		bool retVal = true;

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

		float* temp_phis = new float[numAngles];
		bool isAxial = true;
		float minSourceZ = sourcePositions[3 * 0 + 2];
		float maxSourceZ = sourcePositions[3 * 0 + 2];

		sod = 0.0;
		sdd = 0.0;
		for (int i = 0; i < numPairs; i++)
		{
			// Normalized rowVectors and colVectors
			float mag;
			mag = sqrt(rowVectors[3 * i + 0] * rowVectors[3 * i + 0] + rowVectors[3 * i + 1] * rowVectors[3 * i + 1] + rowVectors[3 * i + 2] * rowVectors[3 * i + 2]);
			if (mag > 0.0)
			{
				rowVectors[3 * i + 0] /= mag;
				rowVectors[3 * i + 1] /= mag;
				rowVectors[3 * i + 2] /= mag;
			}
			else
			{
				printf("Error: rowVectors must be non-zero!\n");
				retVal = false;
				break;
			}
			mag = sqrt(colVectors[3 * i + 0] * colVectors[3 * i + 0] + colVectors[3 * i + 1] * colVectors[3 * i + 1] + colVectors[3 * i + 2] * colVectors[3 * i + 2]);
			if (mag > 0.0)
			{
				colVectors[3 * i + 0] /= mag;
				colVectors[3 * i + 1] /= mag;
				colVectors[3 * i + 2] /= mag;
			}
			else
			{
				printf("Error: colVectors must be non-zero!\n");
				retVal = false;
				break;
			}

			// Orthogonalize row and col vectors
			float u_dot_v = rowVectors[3 * i + 0] * colVectors[3 * i + 0] + rowVectors[3 * i + 1] * colVectors[3 * i + 1] + rowVectors[3 * i + 2] * colVectors[3 * i + 2];
			if (fabs(u_dot_v) > 0.01)
			{
				printf("Error: colVectors and rowVectors must be orthogonal!\n");
				retVal = false;
				break;
			}
			//printf("u_dot_v = %f\n", u_dot_v);
			rowVectors[3 * i + 0] -= u_dot_v * colVectors[3 * i + 0];
			rowVectors[3 * i + 1] -= u_dot_v * colVectors[3 * i + 1];
			rowVectors[3 * i + 2] -= u_dot_v * colVectors[3 * i + 2];

			// Calculate sod and sdd
			float s_x = sourcePositions[3 * i + 0];
			float s_y = sourcePositions[3 * i + 1];
			float s_z = sourcePositions[3 * i + 2];

			float d_x = moduleCenters[3 * i + 0];
			float d_y = moduleCenters[3 * i + 1];
			float d_z = moduleCenters[3 * i + 2];

			float sod_cur = sqrt(s_x * s_x + s_y * s_y + s_z * s_z);
			//float odd_cur = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
			float sdd_cur = sqrt((d_x-s_x) * (d_x-s_x) + (d_y-s_y) * (d_y-s_y) + (d_z-s_z) * (d_z-s_z));

			if (sod_cur <= 0.0 || sdd_cur <= 0.0)
			{
				printf("Error: invalid source/detector position!\n");
				retVal = false;
				break;
			}
			sod += sod_cur;
			sdd += sdd_cur;

			// Set phis
			//temp_phis[i] = atan2(sourcePositions[3 * i + 1], sourcePositions[3 * i + 0]);
			temp_phis[i] = atan2(s_y - d_y, s_x - d_x); //atan2(s_minus_c.y, s_minus_c.x);
			if (i > 0)
			{
				//printf("%f ==> ", temp_phis[i] * 180.0 / PI);
				temp_phis[i] -= 2.0 * PI * floor(0.5 + (temp_phis[i] - temp_phis[i - 1]) / (2.0 * PI));
				//printf("%f\n", temp_phis[i] * 180.0 / PI);
			}

			// Check if geometry is axial
			minSourceZ = min(minSourceZ, sourcePositions[3 * i + 2]);
			maxSourceZ = max(maxSourceZ, sourcePositions[3 * i + 2]);
			//if (fabs(rowVectors[3 * i + 2]) < 0.995) // max 5.732 degree panel rotation
			if ((rowVectors[3 * i + 2]) < 0.9961) // max 5.0 degree panel rotation
				isAxial = false;
		}
		sod = sod / float(numPairs);
		sdd = sdd / float(numPairs);
		//sdd += sod;

		if (maxSourceZ - minSourceZ > 0.5 * numRows * pixelHeight)
			isAxial = false;

		if (isAxial == true && retVal == true)
		{
			tau = sin(temp_phis[0]) * sourcePositions[0] - cos(temp_phis[0]) * sourcePositions[1];
			for (int i = 0; i < numAngles; i++)
				temp_phis[i] = (temp_phis[i] + 0.5 * PI) * 180.0 / PI;
			set_angles(temp_phis, numAngles);
		}
		else
			tau = 0.0;
		delete[] temp_phis;
		temp_phis = NULL;

		if (retVal == false)
			clearModularBeamParameters();

		return retVal;
	}
}

bool parameters::rotateDetector(float alpha)
{
	if (geometry == MODULAR && colVectors != NULL && rowVectors != NULL)
	{
		for (int i = 0; i < numAngles; i++)
		{
			float* u_vec = &colVectors[3 * i];
			float* v_vec = &rowVectors[3 * i];

			float u_cross_v[3];
			u_cross_v[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
			u_cross_v[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
			u_cross_v[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

			//printf("normal vector: %f, %f, %f\n", u_cross_v[0], u_cross_v[1], u_cross_v[2]);

			rotateAroundAxis(u_cross_v, alpha * PI / 180.0, u_vec);
			rotateAroundAxis(u_cross_v, alpha * PI / 180.0, v_vec);
		}
		return true;
	}
	else
	{
		printf("Error: can only rotate modular-beam detectors\n");
		return false;
	}
}

bool parameters::shiftDetector(float r, float c)
{
	if (geometry == MODULAR)
	{
		if (colVectors != NULL && rowVectors != NULL)
		{
			for (int i = 0; i < numAngles; i++)
			{
				float* u_vec = &colVectors[3 * i];
				float* v_vec = &rowVectors[3 * i];

				float* modCenter = &moduleCenters[3 * i];

				modCenter[0] += u_vec[0] * c + v_vec[0] * r;
				modCenter[1] += u_vec[1] * c + v_vec[1] * r;
				modCenter[2] += u_vec[2] * c + v_vec[2] * r;
			}
			return true;
		}
		else
			return false;
	}
	else
	{
		centerRow -= r/pixelHeight;
		centerCol -= c/pixelWidth;
		return true;
	}
}

bool parameters::offsetScan_has_adequate_angular_range()
{
	if (numAngles <= 1 || angularRange < min(359.0, 360.0 - fabs(T_phi()) * 180.0 / PI))
		return false;
	else
		return true;
}

bool parameters::set_offsetScan(bool aFlag)
{
	if (aFlag == false)
		offsetScan = aFlag;
	else
	{
		if (offsetScan_has_adequate_angular_range() == false)
		{
			printf("Error: offsetScan requires at least 360 degrees of projections!\n");
			//printf("angularRange = %f, T_phi = %f\n", angularRange, T_phi());
			offsetScan = false;
			return false;
		}
		//*
		if (geometry == MODULAR)
		{
			printf("Error: offsetScan only applies to parallel-, fan-, or cone-beam data!\n");
			offsetScan = false;
			return false;
		}
		//*/
		//printf("Warning: offsetScan not working yet!\n");
		offsetScan = aFlag;
		//truncatedScan = false;
	}
	return true;
}

bool parameters::set_truncatedScan(bool aFlag)
{
	if (aFlag == false)
		truncatedScan = aFlag;
	else
	{
		truncatedScan = aFlag;
		//offsetScan = false;
	}
	return true;
}

float parameters::u_0()
{
	if (geometry == CONE && detectorType == CURVED)
		return -(centerCol + colShiftFromFilter) * atan(pixelWidth / sdd);
	else
		return -(centerCol + colShiftFromFilter) * pixelWidth;
}

float parameters::v_0()
{
	return -(centerRow + rowShiftFromFilter) * pixelHeight;
}

float parameters::phi_0()
{
	if (phis == NULL)
		return 0.0;
	else
		return phis[0];
}

float parameters::u(int i, int iphi)
{
	/*
	if (geometry == MODULAR && iphi >= 0 && iphi < numAngles)
	{
		float* s = &sourcePositions[3 * iphi];
		float* c = &moduleCenters[3 * iphi];
		float* u_vec = &colVectors[3 * iphi];
		float* v_vec = &rowVectors[3 * iphi];

		float detNormal[3];
		detNormal[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
		detNormal[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
		detNormal[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

		float t = 1.0 - (c[0] * detNormal[0] + c[1] * detNormal[1] + c[2] * detNormal[2]) / (s[0] * detNormal[0] + s[1] * detNormal[1] + s[2] * detNormal[2]);
		float u_offs = (1 - t) * (s[0] * u_vec[0] + s[1] * u_vec[1] + s[2] * u_vec[2]);

		if (normalizeConeAndFanCoordinateFunctions)
			return u_offs / sdd + u(i);
		else
			return u_offs + u(i);
	}
	//*/
	if (modularbeamIsAxiallyAligned() && iphi >= 0 && iphi < numAngles)
	{
		float* u_vec = &colVectors[3 * iphi];
		float colVec_dot_theta_perp = -u_vec[0] * sin(phis[iphi]) + u_vec[1] * cos(phis[iphi]);
		if (normalizeConeAndFanCoordinateFunctions)
			return colVec_dot_theta_perp*(i * pixelWidth + u_0()) / sdd;
		else
			return colVec_dot_theta_perp * (i * pixelWidth + u_0());
	}
	else if (geometry == CONE && detectorType == CURVED)
		return (i * atan(pixelWidth/sdd) + u_0());
	else if (normalizeConeAndFanCoordinateFunctions == true && (geometry == CONE || geometry == FAN || geometry == MODULAR))
		return (i * pixelWidth + u_0()) / sdd;
	else
		return i * pixelWidth + u_0();
}

float parameters::u_inv(float val)
{
	if (geometry == CONE && detectorType == CURVED)
		return (val - u_0()) / atan(pixelWidth / sdd);
	else if (normalizeConeAndFanCoordinateFunctions == true && (geometry == CONE || geometry == FAN))
	{
		return (sdd*val - u_0()) / pixelWidth;
	}
	else
		return (val - u_0()) / pixelWidth;
}

float parameters::v(int i, int iphi)
{
	/*
	if (geometry == MODULAR && iphi >= 0 && iphi < numAngles)
	{
		float* s = &sourcePositions[3 * iphi];
		float* c = &moduleCenters[3 * iphi];
		float* u_vec = &colVectors[3 * iphi];
		float* v_vec = &rowVectors[3 * iphi];

		float detNormal[3];
		detNormal[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
		detNormal[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
		detNormal[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

		float t = 1.0 - (c[0] * detNormal[0] + c[1] * detNormal[1] + c[2] * detNormal[2]) / (s[0] * detNormal[0] + s[1] * detNormal[1] + s[2] * detNormal[2]);
		float v_offs = (1 - t) * (s[0] * v_vec[0] + s[1] * v_vec[1] + s[2] * v_vec[2]);
		if (normalizeConeAndFanCoordinateFunctions)
			return v_offs / sdd + v(i);
		else
			return v_offs + v(i);
	}
	else */
	if (modularbeamIsAxiallyAligned() && iphi >= 0 && iphi < numAngles)
	{
		float* v_vec = &rowVectors[3 * iphi];
		float rowVec_dot_z = v_vec[2];
		if (normalizeConeAndFanCoordinateFunctions)
			return rowVec_dot_z * (i * pixelHeight + v_0()) / sdd;
		else
			return rowVec_dot_z * (i * pixelHeight + v_0());
	}
	else if (normalizeConeAndFanCoordinateFunctions == true && (geometry == CONE || geometry == FAN || geometry == MODULAR || geometry == CONE_PARALLEL))
		return (i * pixelHeight + v_0()) / sdd;
	else
		return i * pixelHeight + v_0();
}

float parameters::pixelWidth_normalized()
{
	if (geometry == PARALLEL || geometry == CONE_PARALLEL)
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
	{
		//return offsetZ - centerRow * (pixelHeight / voxelHeight) * voxelHeight;
		return offsetZ - centerRow * pixelHeight; // note that offsetZ is always forced to zero
	}
	else if (geometry == MODULAR)
		return offsetZ - 0.5 * float(numZ - 1) * voxelHeight;
	else
	{
		//float rzref = -centerRow * ((sod / sdd * pixelHeight) / voxelHeight) * voxelHeight;
		/*
		float rzref = -centerRow * (sod / sdd * pixelHeight);
		if (helicalPitch != 0.0)
		{
			rzref = (0.5 * float(numRows - 1) - centerRow) * (sod / sdd * pixelHeight) / voxelHeight;
			rzref -= 0.5 * float(numZ - 1);
			rzref *= voxelHeight;
		}
		return offsetZ + rzref;
		//*/
		return offsetZ - 0.5 * float(numZ - 1) * voxelHeight;
	}
}

float parameters::z_samples(int iz)
{
	return iz * voxelHeight + z_0();
}

float parameters::z_source(int i, int k)
{
	if ((geometry != CONE && geometry != CONE_PARALLEL) || phis == NULL || helicalPitch == 0.0)
		return 0.0;
	else
	{
		//float midAngle = 0.5 * (phis[numAngles - 1] + phis[0]);
		//return (phis[i] - midAngle) * helicalPitch;
		if (geometry == CONE)
			return phis[i] * helicalPitch + z_source_offset;
		else //if (geometry == CONE_PARALLEL)
		{
			float alpha = asin(u(k) / sod) + asin(tau / sod);
			return (phis[i] + alpha) * helicalPitch + z_source_offset;
		}
	}
}

bool parameters::set_tau(float tau_in)
{
	tau = tau_in;
	return true;
}

bool parameters::set_normalizedHelicalPitch(float h_normalized)
{
	//h_normalized = 2.0 * PI * helicalPitch / (numRows * pixelHeight * sod / sdd)
	float h = h_normalized * (numRows * pixelHeight * sod / sdd) / (2.0*PI);
	return set_helicalPitch(h);
}

bool parameters::set_helicalPitch(float h)
{
	if (phis == NULL)
	{
		printf("Error: must set CT geometry before setting helicalPitch\n");
		return false;
	}
	else if (geometry != CONE && geometry != CONE_PARALLEL)
	{
		printf("Error: CT geometry must be CONE or CONE_PARALLEL for helical scans\n");
		return false;
	}
	else
	{
		helicalPitch = h;
		if (h == 0.0)
			z_source_offset = 0.0;
		else
		{
			float midAngle = 0.5 * (phis[numAngles - 1] + phis[0]);
			//return (phis[i] - midAngle) * helicalPitch;
			z_source_offset = -midAngle * helicalPitch;
		}
		return true;
	}
}

float parameters::normalizedHelicalPitch()
{
	//helicalPitch = normalizedHelicalPitch*pzsize*nslices*(sod/sdd)/(2pi)
	if (geometry != CONE && geometry != CONE_PARALLEL)
		return 0.0;
	else
		return 2.0 * PI * helicalPitch / (numRows * pixelHeight * sod / sdd);
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
			if (fabs(curSpacing - firstSpacing) > 1.0e-6)
			{
				//printf("dist: %e\n", fabs(curSpacing - firstSpacing));
				return false;
			}
		}
		return true;
	}
}

uint64 parameters::projectionData_numberOfElements()
{
	return uint64(numAngles) * uint64(numRows) * uint64(numCols);
}

uint64 parameters::volumeData_numberOfElements()
{
	return uint64(numX) * uint64(numY) * uint64(numZ);
}

float parameters::projectionDataSize(int extraCols)
{
	return float(4.0 * double(numAngles) * double(numRows) * double(numCols + extraCols) / pow(2.0, 30.0));
}

float parameters::volumeDataSize()
{
	return float(4.0 * double(numX) * double(numY) * double(numZ) / pow(2.0, 30.0));
}

bool parameters::rowRangeNeededForBackprojection(int firstSlice, int lastSlice, int* rowsNeeded, bool doDebug)
{
	if (rowsNeeded == NULL || firstSlice > lastSlice)
		return false;

	if (geometry == PARALLEL || geometry == FAN)
	{
		rowsNeeded[0] = max(0, firstSlice);
		rowsNeeded[1] = min(numRows - 1, lastSlice);
	}
	else if (geometry == MODULAR)
	{
		if (modularbeamIsAxiallyAligned())
		{
			float z_lo = float(firstSlice) * voxelHeight + z_0() - 0.5 * voxelHeight;
			float z_hi = float(lastSlice) * voxelHeight + z_0() + 0.5 * voxelHeight;

			if (doDebug)
				printf("z range: %f to %f\n", z_lo, z_hi);

			vector<int> indices;
			for (int iphi = 0; iphi < numAngles; iphi++)
			{
				float* s = &sourcePositions[3 * iphi];
				float* c = &moduleCenters[3 * iphi];
				float* u_vec = &colVectors[3 * iphi];
				float* v_vec = &rowVectors[3 * iphi];

				float detNormal[3];
				detNormal[0] = u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1];
				detNormal[1] = u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2];
				detNormal[2] = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0];

				//float R = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
				float R_z = sqrt(s[0] * s[0] + s[1] * s[1]);
				//float D = sqrt((s[0] - c[0]) * (s[0] - c[0]) + (s[1] - c[1]) * (s[1] - c[1]) + (s[2] - c[2]) * (s[2] - c[2]));
				float D_z = sqrt((s[0] - c[0]) * (s[0] - c[0]) + (s[1] - c[1]) * (s[1] - c[1]));

				float dist;
				float r[3];
				float t_lo, v_lo, t_hi, v_hi;

				dist = R_z - rFOV() - voxelWidth;
				r[0] = (c[0] - s[0]) * dist / D_z;
				r[1] = (c[1] - s[1]) * dist / D_z;

				r[2] = (z_lo - s[2]);
				t_lo = ((c[0] - s[0]) * detNormal[0] + (c[1] - s[1]) * detNormal[1] + (c[2] - s[2]) * detNormal[2]) / (r[0] * detNormal[0] + r[1] * detNormal[1] + r[2] * detNormal[2]);
				v_lo = (s[0] + t_lo * r[0] - c[0]) * v_vec[0] + (s[1] + t_lo * r[1] - c[1]) * v_vec[1] + (s[2] + t_lo * r[2] - c[2]) * v_vec[2];

				//const float denom = (x - sourcePosition[0]) * detNormal.x + (y - sourcePosition[1]) * detNormal.y + (z - sourcePosition[2]) * detNormal.z;
				//const float t_C = c_minus_s_dot_n / denom;
				//const float v_phi_x = (t_C * ((x - sourcePosition[0]) * v_vec[0] + (y - sourcePosition[1]) * v_vec[1] + (z - sourcePosition[2]) * v_vec[2]) - c_minus_s_dot_v - startVals_g.y) * Tv_inv;

				if (doDebug && iphi == 0)
				{
					//printf("%d: %f to %f\n", iphi, v_lo, v_hi);
					//printf("R_z = %f, dist = %f, D_z = %f, z_lo = %f, z_hi = %f, sourceZ = %f, detector_z = %f\n", R_z, dist, D_z, z_lo, z_hi, s[2], c[2]);
					//printf("%f, %f, %f\n", s[0] + t_lo * r[0], s[1] + t_lo * r[1], s[2] + t_lo * r[2]);
					printf("source = %f, %f, %f  r = %f,%f,%f    t_lo = %f\n", s[0], s[1], s[2], r[0], r[1], r[2], t_lo);
					printf("detector: %f, %f, %f   point = %f, %f, %f\n", c[0], c[1], c[2], s[0] + t_lo * r[0], s[1] + t_lo * r[1], s[2] + t_lo * r[2]);
				}

				r[2] = (z_hi - s[2]);
				t_hi = ((c[0] - s[0]) * detNormal[0] + (c[1] - s[1]) * detNormal[1] + (c[2] - s[2]) * detNormal[2]) / (r[0] * detNormal[0] + r[1] * detNormal[1] + r[2] * detNormal[2]);
				v_hi = (s[0] + t_hi * r[0] - c[0]) * v_vec[0] + (s[1] + t_hi * r[1] - c[1]) * v_vec[1] + (s[2] + t_hi * r[2] - c[2]) * v_vec[2];

				if (doDebug && iphi == 0)
				{
					//printf("%d: %f to %f\n", iphi, v_lo, v_hi);
					//printf("R_z = %f, dist = %f, D_z = %f, z_lo = %f, z_hi = %f, sourceZ = %f, detector_z = %f\n", R_z, dist, D_z, z_lo, z_hi, s[2], c[2]);
					//printf("%f, %f, %f\n", s[0]+ t_hi *r[0], s[1] + t_hi * r[1], s[2] + t_hi * r[2]);
					printf("detector: %f, %f, %f   point = %f, %f, %f\n", c[0], c[1], c[2], s[0] + t_hi * r[0], s[1] + t_hi * r[1], s[2] + t_hi * r[2]);
				}

				float accountForClockingAndPixelSize = fabs(u_vec[2] * 0.5 * pixelWidth * float(numCols - 1)) + 0.5 * pixelHeight;
				//printf("accountForClockingAndPixelSize = %f\n", accountForClockingAndPixelSize);

				//*
				v_lo -= accountForClockingAndPixelSize;
				v_hi += accountForClockingAndPixelSize;
				//*/
				/*
				if (v_lo < 0.0)
					v_lo -= accountForClockingAndPixelSize;
				if (v_hi < 0.0)
					v_hi -= accountForClockingAndPixelSize;
				if (v_lo > 0.0)
					v_lo += accountForClockingAndPixelSize;
				if (v_hi > 0.0)
					v_hi += accountForClockingAndPixelSize;
				//*/

				indices.push_back(max(0, min(numRows - 1, int(floor((min(v_lo, v_hi) - v_0()) / pixelHeight)))));
				indices.push_back(max(0, min(numRows - 1, int(ceil((max(v_lo, v_hi) - v_0()) / pixelHeight)))));

				dist = R_z + rFOV() + voxelWidth;
				r[0] = (c[0] - s[0]) * dist / D_z;
				r[1] = (c[1] - s[1]) * dist / D_z;

				r[2] = (z_lo - s[2]);
				t_lo = ((c[0] - s[0]) * detNormal[0] + (c[1] - s[1]) * detNormal[1] + (c[2] - s[2]) * detNormal[2]) / (r[0] * detNormal[0] + r[1] * detNormal[1] + r[2] * detNormal[2]);
				v_lo = (s[0] + t_lo * r[0] - c[0]) * v_vec[0] + (s[1] + t_lo * r[1] - c[1]) * v_vec[1] + (s[2] + t_lo * r[2] - c[2]) * v_vec[2];

				r[2] = (z_hi - s[2]);
				t_hi = ((c[0] - s[0]) * detNormal[0] + (c[1] - s[1]) * detNormal[1] + (c[2] - s[2]) * detNormal[2]) / (r[0] * detNormal[0] + r[1] * detNormal[1] + r[2] * detNormal[2]);
				v_hi = (s[0] + t_hi * r[0] - c[0]) * v_vec[0] + (s[1] + t_hi * r[1] - c[1]) * v_vec[1] + (s[2] + t_hi * r[2] - c[2]) * v_vec[2];

				//*
				v_lo -= accountForClockingAndPixelSize;
				v_hi += accountForClockingAndPixelSize;
				//*/
				/*
				if (v_lo < 0.0)
					v_lo -= accountForClockingAndPixelSize;
				if (v_hi < 0.0)
					v_hi -= accountForClockingAndPixelSize;
				if (v_lo > 0.0)
					v_lo += accountForClockingAndPixelSize;
				if (v_hi > 0.0)
					v_hi += accountForClockingAndPixelSize;
				//*/

				indices.push_back(max(0, min(numRows - 1, int(floor((min(v_lo, v_hi) - v_0()) / pixelHeight)))));
				indices.push_back(max(0, min(numRows - 1, int(ceil((max(v_lo, v_hi) - v_0()) / pixelHeight)))));
			}
			rowsNeeded[0] = *std::min_element(std::begin(indices), std::end(indices));
			rowsNeeded[1] = *std::max_element(std::begin(indices), std::end(indices));
		}
		else
		{
			rowsNeeded[0] = 0;
			rowsNeeded[1] = numRows - 1;
		}
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
		float v_lo = min(z_lo / v_denom_min, z_lo / v_denom_max) - 0.5 * pixelHeight / sdd;
		float v_hi = max(z_hi / v_denom_min, z_hi / v_denom_max) + 0.5 * pixelHeight / sdd;

		rowsNeeded[0] = min(numRows - 1, max(0, int(floor((v_lo - v_0() / sdd) / T_v))));
		rowsNeeded[1] = max(0, min(numRows - 1, int(ceil((v_hi - v_0() / sdd) / T_v))));

		/*
		if (rowsNeeded[0] > rowsNeeded[1])
		{
			printf("T_v = %f\n", T_v);
			printf("v_denom_min = %f\n", v_denom_min);
			printf("v_denom_max = %f\n", v_denom_max);
			printf("v_lo = %f\n", v_lo);
			printf("v_hi = %f\n", v_hi);
			printf("(v_lo - v_0() / sdd) / T_v = %f\n", (v_lo - v_0() / sdd) / T_v);
			printf("(v_hi - v_0() / sdd) / T_v = %f\n", (v_hi - v_0() / sdd) / T_v);
		}
		//*/
	}
	return true;
}

bool parameters::sliceRangeNeededForProjectionRange(int firstView, int lastView, int* slicesNeeded, bool doClip)
{
	if ((geometry == CONE || geometry == CONE_PARALLEL) && helicalPitch != 0.0)
	{
		float v_lo = (v_0() - 0.5 * pixelHeight) / sdd;
		float v_hi = (float(numRows - 1) * pixelHeight + v_0() + 0.5 * pixelHeight) / sdd;

		// v = z / (R - <x, theta>)
		// v = z / (R - rFOV())
		float T_z = voxelHeight;
		float v_denom_min = sod - rFOV() - voxelWidth;
		float v_denom_max = sod + rFOV() + voxelWidth;
		//float z_lo = min(v_lo * v_denom_min, v_lo * v_denom_max) - 0.5 * voxelHeight;
		//float z_hi = max(v_hi * v_denom_min, v_hi * v_denom_max) + 0.5 * voxelHeight;

		float z_source_firstView = phis[firstView] * helicalPitch + z_source_offset;
		float z_source_lastView = phis[lastView] * helicalPitch + z_source_offset;
		if (geometry == CONE_PARALLEL)
		{
			float alpha_min = asin(u(0) / sod) + asin(tau / sod);
			float alpha_max = asin(u(numCols-1) / sod) + asin(tau / sod);
			if (helicalPitch > 0.0)
			{
				z_source_firstView += alpha_min * helicalPitch;
				z_source_lastView += alpha_max * helicalPitch;
			}
			else
			{
				z_source_firstView += alpha_max * helicalPitch;
				z_source_lastView += alpha_min * helicalPitch;
			}
		}

		vector<float> slices;
		slices.push_back(z_source_firstView + v_denom_min * v_lo);
		slices.push_back(z_source_firstView + v_denom_min * v_hi);
		slices.push_back(z_source_firstView + v_denom_max * v_lo);
		slices.push_back(z_source_firstView + v_denom_max * v_hi);
		slices.push_back(z_source_lastView + v_denom_min * v_lo);
		slices.push_back(z_source_lastView + v_denom_min * v_hi);
		slices.push_back(z_source_lastView + v_denom_max * v_lo);
		slices.push_back(z_source_lastView + v_denom_max * v_hi);

		float z_lo = *std::min_element(std::begin(slices), std::end(slices)) - 0.5 * voxelHeight;
		float z_hi = *std::max_element(std::begin(slices), std::end(slices)) + 0.5 * voxelHeight;

		slicesNeeded[0] = int(floor((z_lo - z_0()) / T_z));
		slicesNeeded[1] = int(ceil((z_hi - z_0()) / T_z));

		if (doClip)
		{
			slicesNeeded[0] = max(0, slicesNeeded[0]);
			slicesNeeded[1] = min(numZ - 1, slicesNeeded[1]);
		}

		return true;
	}
	else
		return sliceRangeNeededForProjection(0, numRows - 1, slicesNeeded, doClip);
}

bool parameters::viewRangeNeededForBackprojection(int firstSlice, int lastSlice, int* viewsNeeded)
{
	viewsNeeded[0] = 0;
	viewsNeeded[1] = numAngles - 1;
	if ((geometry == CONE || geometry == CONE_PARALLEL) && helicalPitch != 0.0)
	{
		float v_lo = (v_0() - 0.5 * pixelHeight) / sdd;
		float v_hi = (float(numRows-1) * pixelHeight + v_0() + 0.5 * pixelHeight) / sdd;

		// v = z / (R - <x, theta>)
		// v = z / (R - rFOV())
		float T_z = voxelHeight;
		float v_denom_min = sod - rFOV() - voxelWidth;
		float v_denom_max = sod + rFOV() + voxelWidth;
		//float z_lo = min(v_lo * v_denom_min, v_lo * v_denom_max) - 0.5 * voxelHeight;
		//float z_hi = max(v_hi * v_denom_min, v_hi * v_denom_max) + 0.5 * voxelHeight;
		float z_lo = firstSlice * voxelHeight + z_0() - 0.5 * voxelHeight;
		float z_hi = lastSlice * voxelHeight + z_0() + 0.5 * voxelHeight;

		vector<float> sourcePositions;
		sourcePositions.push_back(z_lo - v_denom_min * v_lo);
		sourcePositions.push_back(z_lo - v_denom_max * v_lo);
		sourcePositions.push_back(z_lo - v_denom_min * v_hi);
		sourcePositions.push_back(z_lo - v_denom_max * v_hi);
		sourcePositions.push_back(z_hi - v_denom_min * v_lo);
		sourcePositions.push_back(z_hi - v_denom_max * v_lo);
		sourcePositions.push_back(z_hi - v_denom_min * v_hi);
		sourcePositions.push_back(z_hi - v_denom_max * v_hi);
		float sourcePositionFloor = *std::min_element(std::begin(sourcePositions), std::end(sourcePositions));
		float sourcePositionCeil = *std::max_element(std::begin(sourcePositions), std::end(sourcePositions));

		if (geometry == CONE_PARALLEL)
		{
			float alpha_min = asin(u(0) / sod) + asin(tau / sod);
			float alpha_max = asin(u(numCols - 1) / sod) + asin(tau / sod);
			if (helicalPitch > 0.0)
			{
				sourcePositionFloor -= helicalPitch * alpha_max;
				sourcePositionCeil -= helicalPitch * alpha_min;
			}
			else
			{
				sourcePositionFloor -= helicalPitch * alpha_min;
				sourcePositionCeil -= helicalPitch * alpha_max;
			}
		}

		float phi_ind_A = phi_inv((sourcePositionFloor - z_source_offset) / helicalPitch);
		float phi_ind_B = phi_inv((sourcePositionCeil - z_source_offset) / helicalPitch);
		
		if (phi_ind_A <= phi_ind_B)
		{
			viewsNeeded[0] = int(floor(phi_ind_A));
			viewsNeeded[1] = int(ceil(phi_ind_B));
		}
		else
		{
			viewsNeeded[0] = int(floor(phi_ind_B));
			viewsNeeded[1] = int(ceil(phi_ind_A));
		}
	}
	return true;
}

bool parameters::sliceRangeNeededForProjection(int firstRow, int lastRow, int* slicesNeeded, bool doClip)
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
		if (modularbeamIsAxiallyAligned())
		{
			vector<float> zs;
			for (int iphi = 0; iphi < numAngles; iphi++)
			{
				float* s = &sourcePositions[3 * iphi];
				float* c = &moduleCenters[3 * iphi];
				float* u_vec = &colVectors[3 * iphi];
				float* v_vec = &rowVectors[3 * iphi];

				float accountForClockingAndPixelSize = fabs(u_vec[2] * 0.5 * pixelWidth * float(numCols - 1)) + 0.5 * pixelHeight;
				float v_lo = float(firstRow) * pixelHeight + v_0() - accountForClockingAndPixelSize;
				float v_hi = float(lastRow) * pixelHeight + v_0() + accountForClockingAndPixelSize;

				float R_z = sqrt(s[0] * s[0] + s[1] * s[1]);
				float D_z = sqrt((s[0] - c[0]) * (s[0] - c[0]) + (s[1] - c[1]) * (s[1] - c[1]));

				float dist_close = R_z - rFOV() - voxelWidth;
				float dist_far = R_z + rFOV() + voxelWidth;

				zs.push_back(s[2] + (c[2] + v_lo * v_vec[2] - s[2]) * dist_close / D_z);
				zs.push_back(s[2] + (c[2] + v_lo * v_vec[2] - s[2]) * dist_far / D_z);
				zs.push_back(s[2] + (c[2] + v_hi * v_vec[2] - s[2]) * dist_close / D_z);
				zs.push_back(s[2] + (c[2] + v_hi * v_vec[2] - s[2]) * dist_far / D_z);
			}
			float z_min = *std::min_element(std::begin(zs), std::end(zs)) - 0.5 * voxelHeight;
			float z_max = *std::max_element(std::begin(zs), std::end(zs)) + 0.5 * voxelHeight;

			slicesNeeded[0] = int(floor((z_min - z_0()) / voxelHeight));
			slicesNeeded[1] =  int(ceil((z_max - z_0()) / voxelHeight));
			if (doClip)
			{
				slicesNeeded[0] = max(0, min(numZ - 1, slicesNeeded[0]));
				slicesNeeded[1] = max(0, min(numZ - 1, slicesNeeded[1]));
			}
		}
		else
		{
			slicesNeeded[0] = 0;
			slicesNeeded[1] = numZ - 1;
		}
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

		slicesNeeded[0] = int(floor((z_lo - z_0()) / T_z));
		slicesNeeded[1] = int(ceil((z_hi - z_0()) / T_z));

		if (doClip)
		{
			//slicesNeeded[0] = max(0, slicesNeeded[0]);
			//slicesNeeded[1] = min(numZ - 1, slicesNeeded[1]);

			slicesNeeded[0] = max(0, min(numZ - 1, slicesNeeded[0]));
			slicesNeeded[1] = max(0, min(numZ - 1, slicesNeeded[1]));
		}
	}
	return true;
}

float parameters::requiredGPUmemory(int extraCols, int numProjectionData, int numVolumeData)
{
	if (mu != NULL)
		return numProjectionData*projectionDataSize(extraCols) + 2.0* numVolumeData*volumeDataSize() + extraMemoryReserved;
	else
		return numProjectionData*projectionDataSize(extraCols) + numVolumeData*volumeDataSize() + extraMemoryReserved;
}

bool parameters::hasSufficientGPUmemory(bool useLeastGPUmemory, int extraColumns, int numProjectionData, int numVolumeData)
{
#ifndef __USE_CPU
	if (useLeastGPUmemory)
	{
		if (getAvailableGPUmemory(whichGPUs) < requiredGPUmemory(extraColumns, numProjectionData, numVolumeData))
			return false;
		else
			return true;
	}
	else
	{
		if (getAvailableGPUmemory(whichGPU) < requiredGPUmemory(extraColumns, numProjectionData, numVolumeData))
			return false;
		else
			return true;
	}
#else
	return false;
#endif
}

bool parameters::removeProjections(int firstProj, int lastProj)
{
	if (firstProj < 0 || lastProj >= numAngles || firstProj > lastProj)
	{
		printf("Error: invalid range of projections to remove/keep");
		return false;
	}

	int numAngles_new = lastProj - firstProj + 1;
	float* phis_new = NULL;
	if (phis != NULL)
	{
		float* phis_cur = new float[numAngles];
		get_angles(phis_cur);
		phis_new = new float[numAngles_new];
		for (int i = firstProj; i <= lastProj; i++)
			phis_new[i - firstProj] = phis_cur[i];
		delete[] phis_cur;
	}
	if (geometry == MODULAR)
	{
		if (sourcePositions == NULL || moduleCenters == NULL || rowVectors == NULL || colVectors == NULL)
			return false;
		float* sourcePositions_new = new float[3 * numAngles_new];
		float* moduleCenters_new = new float[3 * numAngles_new];
		float* rowVectors_new = new float[3 * numAngles_new];
		float* colVectors_new = new float[3 * numAngles_new];

		for (int i = firstProj; i <= lastProj; i++)
		{
			int ii = i - firstProj;
			for (int j = 0; j < 3; j++)
			{
				sourcePositions_new[3 * ii + j] = sourcePositions[3 * i + j];
				moduleCenters_new[3 * ii + j] = moduleCenters[3 * i + j];
				rowVectors_new[3 * ii + j] = rowVectors[3 * i + j];
				colVectors_new[3 * ii + j] = colVectors[3 * i + j];
			}
		}
		set_sourcesAndModules(sourcePositions_new, moduleCenters_new, rowVectors_new, colVectors_new, numAngles_new);
		delete[] sourcePositions_new;
		delete[] moduleCenters_new;
		delete[] rowVectors_new;
		delete[] colVectors_new;
	}
	if (phis_new != NULL)
	{
		float phi_start_save = phi_start;
		float phi_end_save = phi_end;
		set_angles(phis_new, numAngles_new);
		phi_start = phi_start_save;
		phi_end = phi_end_save;
	}
	numAngles = numAngles_new;
	return true;
}

bool parameters::set_numTVneighbors(int N)
{
	if (N == 6 || N == 26)
	{
		numTVneighbors = N;
		return true;
	}
	else
	{
		printf("Error: set_numTVneighbors must be 6 or 26\n");
		return false;
	}
}

bool parameters::modularbeamIsAxiallyAligned()
{
	if (geometry == MODULAR && phis != NULL)
		return true;
	else
		return false;
}

bool parameters::convert_conebeam_to_modularbeam()
{
	if (geometry != CONE || detectorType == CURVED)
	{
		printf("Error: input geometry must be flat-panel cone-beam");
		return false;
	}
	if (geometryDefined() == false)
	{
		printf("Error: input geometry must be cone-beam");
		return false;
	}

	float horizontalDetectorShift = 0.5 * float(numCols - 1) * pixelWidth + u_0();
	float verticalDetectorShift = 0.5 * float(numRows - 1) * pixelHeight + v_0();
	//printf("horizontalDetectorShift = %f\n", horizontalDetectorShift);

	float* s_pos = new float[3 * numAngles];
	float* d_pos = new float[3 * numAngles];
	float* v_vec = new float[3 * numAngles];
	float* u_vec = new float[3 * numAngles];
	for (int iphi = 0; iphi < numAngles; iphi++)
	{
		float cos_phi = cos(phis[iphi]);
		float sin_phi = sin(phis[iphi]);

		s_pos[3 * iphi + 0] = sod * cos_phi + tau * sin_phi;
		s_pos[3 * iphi + 1] = sod * sin_phi - tau * cos_phi;
		s_pos[3 * iphi + 2] = z_source(iphi);

		d_pos[3 * iphi + 0] = (sod - sdd) * cos_phi;
		d_pos[3 * iphi + 1] = (sod - sdd) * sin_phi;
		d_pos[3 * iphi + 2] = z_source(iphi);

		v_vec[3 * iphi + 0] = 0.0;
		v_vec[3 * iphi + 1] = 0.0;
		v_vec[3 * iphi + 2] = 1.0;

		u_vec[3 * iphi + 0] = -sin_phi;
		u_vec[3 * iphi + 1] = cos_phi;
		u_vec[3 * iphi + 2] = 0.0;

		d_pos[3 * iphi + 0] += horizontalDetectorShift * u_vec[3 * iphi + 0];
		d_pos[3 * iphi + 1] += horizontalDetectorShift * u_vec[3 * iphi + 1];
		d_pos[3 * iphi + 2] += horizontalDetectorShift * u_vec[3 * iphi + 2];

		d_pos[3 * iphi + 0] += verticalDetectorShift * v_vec[3 * iphi + 0];
		d_pos[3 * iphi + 1] += verticalDetectorShift * v_vec[3 * iphi + 1];
		d_pos[3 * iphi + 2] += verticalDetectorShift * v_vec[3 * iphi + 2];
	}
	centerRow = 0.5 * float(numRows - 1);
	centerCol = 0.5 * float(numCols - 1);
	if (set_sourcesAndModules(s_pos, d_pos, v_vec, u_vec, numAngles))
		geometry = MODULAR;
	delete[] s_pos;
	delete[] d_pos;
	delete[] v_vec;
	delete[] u_vec;

	//for (int i = 0; i < numAngles; i++)
	//	printf("%f\n", phis[i]*180.0/PI);

	set_offsetScan(offsetScan);

	return true;
}

bool parameters::convert_parallelbeam_to_modularbeam()
{
	if (geometry != PARALLEL)
	{
		printf("Error: input geometry must be parallel-beam");
		return false;
	}
	if (geometryDefined() == false)
	{
		printf("Error: input geometry must be cone-beam");
		return false;
	}

	float horizontalDetectorShift = 0.5 * float(numCols - 1) * pixelWidth + u_0();
	float verticalDetectorShift = 0.5 * float(numRows - 1) * pixelHeight + v_0();
	//printf("horizontalDetectorShift = %f\n", horizontalDetectorShift);

	double rFOV_max = max(double(furthestFromCenter()), max(double(fabs(z_0())), double(fabs(z_samples(numZ - 1)))));
	float odd = 4.0 * rFOV_max;

	sdd = 10.0 * odd;
	while (sdd / odd < 1.0e20)
	{
		double alpha = max(atan(0.5 * (numCols - 1) * pixelWidth / sdd), atan(0.5 * (numRows - 1) * pixelHeight / sdd));
		double maxDivergence = tan(alpha) * (sdd + rFOV_max) - tan(alpha) * (sdd - rFOV_max);
		double maxTravel = maxDivergence / voxelWidth;

		if (maxTravel < 0.25)
			break;
		sdd *= 10.0;
	}
	sod = sdd - odd;

	float* s_pos = new float[3 * numAngles];
	float* d_pos = new float[3 * numAngles];
	float* v_vec = new float[3 * numAngles];
	float* u_vec = new float[3 * numAngles];
	for (int iphi = 0; iphi < numAngles; iphi++)
	{
		float cos_phi = cos(phis[iphi]);
		float sin_phi = sin(phis[iphi]);

		s_pos[3 * iphi + 0] = sod * cos_phi;// +tau * sin_phi;
		s_pos[3 * iphi + 1] = sod * sin_phi;// -tau * cos_phi;
		s_pos[3 * iphi + 2] = 0.0;

		d_pos[3 * iphi + 0] = (sod - sdd) * cos_phi;
		d_pos[3 * iphi + 1] = (sod - sdd) * sin_phi;
		d_pos[3 * iphi + 2] = 0.0;

		v_vec[3 * iphi + 0] = 0.0;
		v_vec[3 * iphi + 1] = 0.0;
		v_vec[3 * iphi + 2] = 1.0;

		u_vec[3 * iphi + 0] = -sin_phi;
		u_vec[3 * iphi + 1] = cos_phi;
		u_vec[3 * iphi + 2] = 0.0;

		d_pos[3 * iphi + 0] += horizontalDetectorShift * u_vec[3 * iphi + 0];
		d_pos[3 * iphi + 1] += horizontalDetectorShift * u_vec[3 * iphi + 1];
		d_pos[3 * iphi + 2] += horizontalDetectorShift * u_vec[3 * iphi + 2];

		d_pos[3 * iphi + 0] += verticalDetectorShift * v_vec[3 * iphi + 0];
		d_pos[3 * iphi + 1] += verticalDetectorShift * v_vec[3 * iphi + 1];
		d_pos[3 * iphi + 2] += verticalDetectorShift * v_vec[3 * iphi + 2];
	}
	centerRow = 0.5 * float(numRows - 1);
	centerCol = 0.5 * float(numCols - 1);
	if (set_sourcesAndModules(s_pos, d_pos, v_vec, u_vec, numAngles))
		geometry = MODULAR;
	delete[] s_pos;
	delete[] d_pos;
	delete[] v_vec;
	delete[] u_vec;

	set_offsetScan(offsetScan);

	return true;
}
