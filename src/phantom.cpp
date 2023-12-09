////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
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

bool phantom::addObject(float* f, parameters* params_in, int type, float* c, float* r, float val)
{
	if (f == NULL || params_in == NULL || c == NULL || r == NULL)
		return false;
	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		printf("Error: phantom class not yet defined for XYZ volume order!\n");
		return false;
	}
	params = params_in;
	int minX = int(floor(x_inv(c[0] - r[0] - params->voxelWidth)));
	int maxX = int(ceil(x_inv(c[0] + r[0] + params->voxelWidth)));
	int minY = int(floor(y_inv(c[1] - r[1] - params->voxelWidth)));
	int maxY = int(ceil(y_inv(c[1] + r[1] + params->voxelWidth)));
	int minZ = int(floor(z_inv(c[2] - r[2] - params->voxelHeight)));
	int maxZ = int(ceil(z_inv(c[2] + r[2] + params->voxelHeight)));

	minX = max(0, min(params->numX - 1, minX));
	maxX = max(0, min(params->numX - 1, maxX));
	minY = max(0, min(params->numY - 1, minY));
	maxY = max(0, min(params->numY - 1, maxY));
	minZ = max(0, min(params->numZ - 1, minZ));
	maxZ = max(0, min(params->numZ - 1, maxZ));

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iz = minZ; iz <= maxZ; iz++)
	{
		float z = iz * params->voxelHeight + params->z_0();
		float z_hat = (z - c[2]) / r[2];
		float* zSlice = &f[uint64(iz)*uint64(params->numY*params->numX)];
		for (int iy = minY; iy <= maxY; iy++)
		{
			float y = iy * params->voxelWidth + params->y_0();
			float y_hat = (y - c[1]) / r[1];
			float* xLine = &zSlice[iy*params->numX];
			for (int ix = minX; ix <= maxX; ix++)
			{
				float x = ix * params->voxelWidth + params->x_0();
				float x_hat = (x - c[0]) / r[0];
				switch (type)
				{
				case ELLIPSOID:
					if (x_hat * x_hat + y_hat * y_hat + z_hat * z_hat <= 1.0)
						xLine[ix] = val;
					break;
				case PARALLELEPIPED:
					if (fabs(x_hat) <= 1.0 && fabs(y_hat) <= 1.0 && fabs(z_hat) <= 1.0)
						xLine[ix] = val;
					break;
				case CYLINDER_X:
					if (fabs(x_hat) <= 1.0 && y_hat * y_hat + z_hat * z_hat <= 1.0)
						xLine[ix] = val;
					break;
				case CYLINDER_Y:
					if (fabs(y_hat) <= 1.0 && x_hat * x_hat + z_hat * z_hat <= 1.0)
						xLine[ix] = val;
					break;
				case CYLINDER_Z:
					if (fabs(z_hat) <= 1.0 && x_hat * x_hat + y_hat * y_hat <= 1.0)
						xLine[ix] = val;
					break;
				default:
					xLine[ix] = xLine[ix];
				}
			}
		}
	}
	return true;
}

float phantom::x_inv(float val)
{
	return (val - params->x_0()) / params->voxelWidth;
}

float phantom::y_inv(float val)
{
	return (val - params->y_0()) / params->voxelWidth;
}

float phantom::z_inv(float val)
{
	return (val - params->z_0()) / params->voxelHeight;
}
