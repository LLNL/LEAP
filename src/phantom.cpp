////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// sets voxelized phantoms of 3D geometric shapes to assist in algorithm
// development and testing
////////////////////////////////////////////////////////////////////////////////

#include "phantom.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>

using namespace std;

phantom::phantom()
{
}

phantom::~phantom()
{
}

bool phantom::addObject(float* f, parameters* params_in, int type, float* c, float* r, float val, float* A, float* clip)
{
	if (f == NULL || params_in == NULL || c == NULL || r == NULL)
		return false;
	params = params_in;
	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		printf("Error: phantom class not yet defined for XYZ volume order!\n");
		return false;
	}
	x_0 = params->x_0();
	y_0 = params->y_0();
	z_0 = params->z_0();
	numX = params->numX;
	numY = params->numY;
	numZ = params->numZ;
	T_x = params->voxelWidth;
	T_y = params->voxelWidth;
	T_z = params->voxelHeight;

	//*
	if (params->isSymmetric())
	{
		y_0 = params->z_0();
		numY = params->numZ; // 1
		T_y = params->voxelHeight;

		//y_0 = 0.0;

		x_0 = params->y_0();
		numX = params->numY;
		T_x = params->voxelWidth;
		
		z_0 = params->x_0();
		numZ = params->numX;
		T_z = params->voxelWidth;
	}
	//*/

	int minX = int(floor(x_inv(c[0] - r[0] - T_x)));
	int maxX = int(ceil(x_inv(c[0] + r[0] + T_x)));
	int minY = int(floor(y_inv(c[1] - r[1] - T_y)));
	int maxY = int(ceil(y_inv(c[1] + r[1] + T_y)));
	int minZ = int(floor(z_inv(c[2] - r[2] - T_z)));
	int maxZ = int(ceil(z_inv(c[2] + r[2] + T_z)));

	bool isRotated = false;
	if (A != NULL)
	{
		if (A[0] != 1.0 || A[4] != 1.0 || A[8] != 1.0)
			isRotated = true;
	}
	if (isRotated || (CONE_X <= type && type <= CONE_Z))
	{
		float r_max = max(r[0], max(r[1], r[2]));
		minX = int(floor(x_inv(c[0] - r_max - T_x)));
		maxX = int(ceil(x_inv(c[0] + r_max + T_x)));
		minY = int(floor(y_inv(c[1] - r_max - T_y)));
		maxY = int(ceil(y_inv(c[1] + r_max + T_y)));
		minZ = int(floor(z_inv(c[2] - r_max - T_z)));
		maxZ = int(ceil(z_inv(c[2] + r_max + T_z)));
	}

	if (type == CONE_X)
	{
		double l = r[0];
		double r_1 = r[1];
		double r_2 = r[2];

		r[1] = fabs(r_2 - r_1) / (2.0 * l);
		r[2] = r[1];
		r[0] = 1.0;

		c[0] = c[0] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}
	else if (type == CONE_Y)
	{
		double r_1 = r[0];
		double l = r[1];
		double r_2 = r[2];

		r[0] = fabs(r_2 - r_1) / (2.0 * l);
		r[2] = r[0];
		r[1] = 1.0;

		c[1] = c[1] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}
	else if (type == CONE_Z)
	{
		double r_1 = r[0];
		double r_2 = r[1];
		double l = r[2];

		r[0] = fabs(r_2 - r_1) / (2.0 * l);
		r[1] = r[0];
		r[2] = 1.0;

		c[2] = c[2] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}

	minX = max(0, min(numX - 1, minX));
	maxX = max(0, min(numX - 1, maxX));
	minY = max(0, min(numY - 1, minY));
	maxY = max(0, min(numY - 1, maxY));
	minZ = max(0, min(numZ - 1, minZ));
	maxZ = max(0, min(numZ - 1, maxZ));

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iz = minZ; iz <= maxZ; iz++)
	{
		float z = iz * T_z + z_0;
		float z_hat = (z - c[2]) / r[2];
		float* zSlice = &f[uint64(iz)*uint64(numY*numX)];
		for (int iy = minY; iy <= maxY; iy++)
		{
			float y = iy * T_y + y_0;
			float y_hat = (y - c[1]) / r[1];
			float* xLine = &zSlice[iy*numX];
			for (int ix = minX; ix <= maxX; ix++)
			{
				float x = ix * T_x + x_0;
				float x_hat = (x - c[0]) / r[0];

				if (isRotated)
				{
					float x_r = (A[0] * (x - c[0]) + A[1] * (y - c[1]) + A[2] * (z - c[2])) / r[0];
					float y_r = (A[3] * (x - c[0]) + A[4] * (y - c[1]) + A[5] * (z - c[2])) / r[1];
					float z_r = (A[6] * (x - c[0]) + A[7] * (y - c[1]) + A[8] * (z - c[2])) / r[2];
					if (isInside(x_r, y_r, z_r, type, clip))
						xLine[ix] = val;
				}
				else
				{
					if (isInside(x_hat, y_hat, z_hat, type, clip))
						xLine[ix] = val;
				}
			}
		}
	}
	return true;
}

bool phantom::isInside(float x, float y, float z, int type, float* clip)
{
	bool retVal = false;
	switch (type)
	{
	case ELLIPSOID:
		if (x * x + y * y + z * z <= 1.0)
			retVal = true;
		break;
	case PARALLELEPIPED:
		if (fabs(x) <= 1.0 && fabs(y) <= 1.0 && fabs(z) <= 1.0)
			retVal = true;
		break;
	case CYLINDER_X:
		if (fabs(x) <= 1.0 && y * y + z * z <= 1.0)
			retVal = true;
		break;
	case CYLINDER_Y:
		if (fabs(y) <= 1.0 && x * x + z * z <= 1.0)
			retVal = true;
		break;
	case CYLINDER_Z:
		if (fabs(z) <= 1.0 && x * x + y * y <= 1.0)
			retVal = true;
		break;
	case CONE_X:
		if (y * y + z * z <= x * x && clipCone[0] <= x && x <= clipCone[1])
			retVal = true;
		break;
	case CONE_Y:
		if (x * x + z * z <= y * y && clipCone[0] <= y && y <= clipCone[1])
			retVal = true;
		break;
	case CONE_Z:
		if (x * x + y * y <= z * z && clipCone[0] <= z && z <= clipCone[1])
			retVal = true;
		break;
	default:
		retVal = false;
	}
	if (clip != NULL)
	{
		if (clip[0] * x > 0.0)
			retVal = false;
		if (clip[1] * y > 0.0)
			retVal = false;
		if (clip[2] * z > 0.0)
			retVal = false;
	}
	return retVal;
}

float phantom::x_inv(float val)
{
	return (val - x_0) / T_x;
}

float phantom::y_inv(float val)
{
	return (val - y_0) / T_y;
}

float phantom::z_inv(float val)
{
	return (val - z_0) / T_z;
}
