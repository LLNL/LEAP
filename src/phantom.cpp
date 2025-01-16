////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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
#define OUT_OF_BOUNDS 1.0e12;

phantom::phantom()
{
	floatData = NULL;
	intData = NULL;
	params = NULL;
	objects.clear();
}

phantom::phantom(parameters* params_in)
{
	floatData = NULL;
	intData = NULL;
	params = params_in;
	if (params != NULL)
	{
		x_0 = params->x_0();
		y_0 = params->y_0();
		z_0 = params->z_0();
		numX = params->numX;
		numY = params->numY;
		numZ = params->numZ;
		T_x = params->voxelWidth;
		T_y = params->voxelWidth;
		T_z = params->voxelHeight;
	}
	objects.clear();
}

phantom::~phantom()
{
	if (intData != NULL)
		free(intData);
	intData = NULL;
	if (floatData != NULL)
		free(floatData);
	floatData = NULL;
	objects.clear();
}

bool phantom::makeTempData(int num_threads)
{
	if (intData != NULL)
		free(intData);
	intData = NULL;
	if (floatData != NULL)
		free(floatData);
	floatData = NULL;

	if (num_threads > 0 && objects.size() > 0)
	{
		floatData = (double*)malloc(size_t(2 * num_threads * objects.size()) * sizeof(double));
		intData = (int*)malloc(size_t(num_threads * objects.size()) * sizeof(int));
		return true;
	}
	else
		return false;
}

bool phantom::addObject(int type, float* c, float* r, float val, float* A, float* clip)
{
	geometricObject object;
	if (object.init(type, c, r, val, A, clip) == true)
	{
		objects.push_back(object);
		//printf("number of objects: %d\n", int(objects.size()));
		return true;
	}
	else
	{
		//printf("failed!\n");
		return false;
	}
}

void phantom::clearObjects()
{
	objects.clear();
}

bool phantom::voxelize(float* f, parameters* params_in, int oversampling)
{
	if (f == NULL || params_in == NULL)
		return false;
	else
	{
		for (int n = 0; n < int(objects.size()); n++)
		{
			float clip[4];
			float* clip_ptr = clip;
			if (objects[n].numClippingPlanes <= 0)
				clip_ptr = NULL;
			//*
			if (CONE_X <= objects[n].type && objects[n].type <= CONE_Z)
			{
				objects[n].restore_cone_params();
				clip[0] = 0.0;
				clip[1] = 0.0;
				clip[2] = 0.0;
				clip[3] = 0.0;
				clip_ptr = NULL;
			}
			else
			{
				clip[0] = -objects[n].clippingPlanes[0][0];
				clip[1] = -objects[n].clippingPlanes[0][1];
				clip[2] = -objects[n].clippingPlanes[0][2];
				clip[3] = -objects[n].clippingPlanes[0][3];
			}
			//*/
			addObject(f, params_in, objects[n].type, objects[n].centers, objects[n].radii, objects[n].val, objects[n].A, clip_ptr, oversampling);
		}
		return true;
	}
}

bool phantom::addObject(float* f, parameters* params_in, int type, float* c, float* r, float val, float* A, float* clip, int oversampling)
{
	if (c == NULL || r == NULL)
		return false;
	if (f == NULL || params_in == NULL)
		return addObject(type, c, r, val, A, clip);
	params = params_in;
	/*
	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		printf("Error: phantom class not yet defined for XYZ volume order!\n");
		return false;
	}
	//*/
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
		float r_max = sqrt(3.0)*max(r[0], max(r[1], r[2]));
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

	oversampling = max(1, min(oversampling, 5));

	if (params->volumeDimensionOrder == parameters::ZYX)
	{
		if (oversampling == 1)
		{
			omp_set_num_threads(omp_get_num_procs());
			#pragma omp parallel for
			for (int iz = minZ; iz <= maxZ; iz++)
			{
				float z = iz * T_z + z_0;
				float z_hat = (z - c[2]) / r[2];
				float* zSlice = &f[uint64(iz) * uint64(numY * numX)];
				for (int iy = minY; iy <= maxY; iy++)
				{
					float y = iy * T_y + y_0;
					float y_hat = (y - c[1]) / r[1];
					float* xLine = &zSlice[iy * numX];
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
		}
		else
		{
			double frac = 1.0 / float(oversampling * oversampling * oversampling);

			omp_set_num_threads(omp_get_num_procs());
			#pragma omp parallel for
			for (int iz = minZ; iz <= maxZ; iz++)
			{
				float z = iz * T_z + z_0;
				//float z_hat = (z - c[2]) / r[2];
				float* zSlice = &f[uint64(iz) * uint64(numY * numX)];
				for (int iy = minY; iy <= maxY; iy++)
				{
					float y = iy * T_y + y_0;
					//float y_hat = (y - c[1]) / r[1];
					float* xLine = &zSlice[iy * numX];
					for (int ix = minX; ix <= maxX; ix++)
					{
						float x = ix * T_x + x_0;
						//float x_hat = (x - c[0]) / r[0];

						float curVal = xLine[ix];
						float accum = 0.0;
						for (int iz_os = 0; iz_os < oversampling; iz_os++)
						{
							float z_os = z + T_z / float(oversampling + 1) * (float(iz_os) - 0.5 * float(oversampling - 1));
							float z_hat = (z_os - c[2]) / r[2];
							for (int iy_os = 0; iy_os < oversampling; iy_os++)
							{
								float y_os = y + T_y / float(oversampling + 1) * (float(iy_os) - 0.5 * float(oversampling - 1));
								float y_hat = (y_os - c[1]) / r[1];
								for (int ix_os = 0; ix_os < oversampling; ix_os++)
								{
									float x_os = x + T_x / float(oversampling + 1) * (float(ix_os) - 0.5 * float(oversampling - 1));
									float x_hat = (x_os - c[0]) / r[0];
									if (isRotated)
									{
										float x_r = (A[0] * (x_os - c[0]) + A[1] * (y_os - c[1]) + A[2] * (z_os - c[2])) / r[0];
										float y_r = (A[3] * (x_os - c[0]) + A[4] * (y_os - c[1]) + A[5] * (z_os - c[2])) / r[1];
										float z_r = (A[6] * (x_os - c[0]) + A[7] * (y_os - c[1]) + A[8] * (z_os - c[2])) / r[2];
										if (isInside(x_r, y_r, z_r, type, clip))
											accum += (val-curVal)*frac;
									}
									else
									{
										if (isInside(x_hat, y_hat, z_hat, type, clip))
											accum += (val - curVal) * frac;
									}
								}
							}
						}
						xLine[ix] = accum+curVal;
					}
				}
			}
		}
	}
	else
	{
		omp_set_num_threads(omp_get_num_procs());
		#pragma omp parallel for
		for (int ix = minX; ix <= maxX; ix++)
		{
			float x = ix * T_x + x_0;
			float x_hat = (x - c[0]) / r[0];
			
			float* xSlice = &f[uint64(ix) * uint64(numY * numZ)];
			for (int iy = minY; iy <= maxY; iy++)
			{
				float y = iy * T_y + y_0;
				float y_hat = (y - c[1]) / r[1];
				float* zLine = &xSlice[iy * numZ];
				for (int iz = minZ; iz <= maxZ; iz++)
				{
					float z = iz * T_z + z_0;
					float z_hat = (z - c[2]) / r[2];

					if (isRotated)
					{
						float x_r = (A[0] * (x - c[0]) + A[1] * (y - c[1]) + A[2] * (z - c[2])) / r[0];
						float y_r = (A[3] * (x - c[0]) + A[4] * (y - c[1]) + A[5] * (z - c[2])) / r[1];
						float z_r = (A[6] * (x - c[0]) + A[7] * (y - c[1]) + A[8] * (z - c[2])) / r[2];
						if (isInside(x_r, y_r, z_r, type, clip))
							zLine[iz] = val;
					}
					else
					{
						if (isInside(x_hat, y_hat, z_hat, type, clip))
							zLine[iz] = val;
					}
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

double phantom::lineIntegral(double* p, double* r)
{
	if (p == NULL || r == NULL)
		return 0.0;

	/*
	printf("number of objects: %d\n", int(objects.size()));
	printf("p = (%f, %f, %f)\n", p[0], p[1], p[2]);
	printf("r = (%f, %f, %f)\n", r[0], r[1], r[2]);
	//*/

	int count = 0;
	vector<double> endPoints;
	//vector<int> objectIndices;
	int* objectIndices = &intData[omp_get_thread_num() * objects.size()];
	//double* intersection_0 = (double*)malloc(size_t(int(objects.size())) * sizeof(double));
	//double* intersection_1 = (double*)malloc(size_t(int(objects.size())) * sizeof(double));
	double* intersection_0 = &floatData[omp_get_thread_num() * 2 * objects.size()]; // 2 * num_threads * objects.size()
	//double* intersection_0 = (double*)malloc(size_t(2*int(objects.size())) * sizeof(double));
	double* intersection_1 = &intersection_0[int(objects.size())];
	for (int i = 0; i < int(objects.size()); i++)
	{
		double ts[2];
		if (objects[i].intersectionEndPoints(p, r, ts))
		{
			endPoints.push_back(ts[0]);
			endPoints.push_back(ts[1]);
			intersection_0[i] = ts[0];
			intersection_1[i] = ts[1];
			//objectIndices.push_back(i);
			objectIndices[count] = i;
			count += 1;
			//printf("intersection: %f to %f\n", ts[0], ts[1]);
		}
		else
		{
			//printf("no intersection (%f, %f)\n", ts[0], ts[1]);
			intersection_0[i] = OUT_OF_BOUNDS;
			intersection_1[i] = OUT_OF_BOUNDS;
		}
	}
	double retVal = 0.0;
	if (count > 0)
	{
		sort(endPoints.begin(), endPoints.end());

		/*
		double* arealDensities = (double*)malloc(size_t(int(objects.size()))*sizeof(double));
		for (int j = 0; j < int(objects.size()); j++)
			arealDensities[j] = 0.0;
		//*/

		for (int i = 0; i < int(endPoints.size())-1; i++)
		{
			// Consider the interval (allPoints[i], allPoints[i+1])
			double midPoint = (endPoints[i + 1] + endPoints[i]) / 2.0;
			//for (int j = int(objects.size())-1; j >= 0; j--)
			for (int ind = count-1; ind >= 0; ind--)
			{
				int j = objectIndices[ind];
				//if (objects[j].val != 0.0)
				{
					// Find which object this interval belongs to
					if (intersection_0[j] <= midPoint && midPoint <= intersection_1[j])
					{
						//if (isnan(arealDensities[j]))
						//	arealDensities[j] = 0.0;
						//arealDensities[j] += objects[j].val * (endPoints[i + 1] - endPoints[i]);
						retVal += objects[j].val * (endPoints[i + 1] - endPoints[i]);
						break;
					}
				}
			}
		}
	}
	//free(intersection_0);
	//free(intersection_1);
	return retVal;
}

//#####################################################################################################################
// geometricObject
//#####################################################################################################################
geometricObject::geometricObject()
{
	reset();
}

geometricObject::geometricObject(int type_in, float* c_in, float* r_in, float val_in, float* A_in, float* clip_in)
{
	init(type_in, c_in, r_in, val_in, A_in, clip_in);
}

geometricObject::~geometricObject()
{
	//reset();
}

void geometricObject::reset()
{
	type = -1;
	for (int i = 0; i < 3; i++)
	{
		centers[i] = 0.0;
		radii[i] = 0.0;
	}
	for (int i = 0; i < 4; i++)
	{
		clippingPlanes[0][i] = 0.0;
		clippingPlanes[1][i] = 0.0;
		clippingPlanes[2][i] = 0.0;
		clippingPlanes[3][i] = 0.0;
		clippingPlanes[4][i] = 0.0;
		clippingPlanes[5][i] = 0.0;
	}
	val = 0.0;
	for (int i = 0; i < 9; i++)
		A[i] = 0.0;
	isRotated = false;
	numClippingPlanes = 0;
}

void geometricObject::restore_cone_params()
{
	for (int i = 0; i < 3; i++)
	{
		radii[i] = radii_save[i];
		centers[i] = centers_save[i];
	}
}

bool geometricObject::init(int type_in, float* c_in, float* r_in, float val_in, float* A_in, float* clip_in)
{
	reset();
	if (c_in == NULL || r_in == NULL || type_in < 0 || type_in > phantom::CONE_Z)
	{
		/*
		if (c_in == NULL)
			printf("Error: center not given\n");
		if (r_in == NULL)
			printf("Error: radii not given\n");
		if (type < 0 || type > phantom::CONE_Z)
			printf("Error: invalid type (%d)\n", type_in);
		//*/
		return false;
	}
	type = type_in;
	for (int i = 0; i < 3; i++)
	{
		centers[i] = c_in[i];
		radii[i] = r_in[i];

		centers_save[i] = c_in[i];
		radii_save[i] = r_in[i];
	}

	if (clip_in != NULL)
	{
		numClippingPlanes = 0;
		for (int i = 0; i < 3; i++)
		{
			if (clip_in[i] != 0.0)
			{
				clippingPlanes[numClippingPlanes][i] = -1.0;
				clippingPlanes[numClippingPlanes][3] = -clip_in[i];
				//clippingPlanes[numClippingPlanes][3] = 0.0;
				numClippingPlanes += 1;
			}
		}
	}

	val = val_in;
	isRotated = false;
	if (A_in != NULL)
	{
		for (int i = 0; i < 9; i++)
		{
			A[i] = A_in[i];
			if ((i == 0 || i == 4 || i == 8))
			{
				if (fabs(A[i] - 1.0) > 1.0e-8)
					isRotated = true;
			}
			else
			{
				if (fabs(A[i]) > 1.0e-8)
					isRotated = true;
			}
			
		}
	}

	if (type == phantom::CONE_X)
	{
		double l = radii[0];
		double r_1 = radii[1];
		double r_2 = radii[2];

		radii[1] = fabs(r_2 - r_1) / (2.0 * l);
		radii[2] = radii[1];
		radii[0] = 1.0;

		centers[0] = centers[0] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}
	else if (type == phantom::CONE_Y)
	{
		double r_1 = radii[0];
		double l = radii[1];
		double r_2 = radii[2];

		radii[0] = fabs(r_2 - r_1) / (2.0 * l);
		radii[2] = radii[0];
		radii[1] = 1.0;

		centers[1] = centers[1] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}
	else if (type == phantom::CONE_Z)
	{
		double r_1 = radii[0];
		double r_2 = radii[1];
		double l = radii[2];

		radii[0] = fabs(r_2 - r_1) / (2.0 * l);
		radii[1] = radii[0];
		radii[2] = 1.0;

		centers[2] = centers[2] - l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[0] = -l + l * (r_2 + r_1) / (r_2 - r_1);
		clipCone[1] = l + l * (r_2 + r_1) / (r_2 - r_1);
	}

	return true;
}

bool geometricObject::intersectionEndPoints(double* p, double* r, double* ts)
{
	// assume ||r|| == 1 && r != (0,0,1) && axis[i] > 0 for i=0,1,2
	// alpha is rotation around x-y axis, currently there is no rotation for x-z or y-z axes
	double q[3];
	double Minv_r[3];

	if (isRotated == false)
	{
		// Scale; (9,0) ops
		q[0] = (p[0] - centers[0]) / radii[0];
		q[1] = (p[1] - centers[1]) / radii[1];
		q[2] = (p[2] - centers[2]) / radii[2];

		Minv_r[0] = r[0] / radii[0];
		Minv_r[1] = r[1] / radii[1];
		Minv_r[2] = r[2] / radii[2];
	}
	else
	{
		double temp[3];

		// Shift
		q[0] = p[0] - centers[0];
		q[1] = p[1] - centers[1];
		q[2] = p[2] - centers[2];

		// Rotate and Scale; (36, 0) ops
		temp[0] = (q[0] * A[0 * 3 + 0] + q[1] * A[0 * 3 + 1] + q[2] * A[0 * 3 + 2]) / radii[0];
		temp[1] = (q[0] * A[1 * 3 + 0] + q[1] * A[1 * 3 + 1] + q[2] * A[1 * 3 + 2]) / radii[1];
		temp[2] = (q[0] * A[2 * 3 + 0] + q[1] * A[2 * 3 + 1] + q[2] * A[2 * 3 + 2]) / radii[2];

		q[0] = temp[0];
		q[1] = temp[1];
		q[2] = temp[2];

		Minv_r[0] = (r[0] * A[0 * 3 + 0] + r[1] * A[0 * 3 + 1] + r[2] * A[0 * 3 + 2]) / radii[0];
		Minv_r[1] = (r[0] * A[1 * 3 + 0] + r[1] * A[1 * 3 + 1] + r[2] * A[1 * 3 + 2]) / radii[1];
		Minv_r[2] = (r[0] * A[2 * 3 + 0] + r[1] * A[2 * 3 + 1] + r[2] * A[2 * 3 + 2]) / radii[2];
	}

	//printf("before: (%f, %f, %f) + t(%f, %f, %f)\n", p[0], p[1], p[2], r[0], r[1], r[2]);
	//printf("after : (%f, %f, %f) + t(%f, %f, %f)\n", q[0], q[1], q[2], Minv_r[0], Minv_r[1], Minv_r[2]); exit(1);

	//return intersectionEndPoints_centeredAndNormalized(&q[0], &Minv_r[0], ts);

	//printf("q = (%f, %f, %f)\n", q[0], q[1], q[2]);
	//printf("Minv_r = (%f, %f, %f)\n", Minv_r[0], Minv_r[1], Minv_r[2]);

	if (intersectionEndPoints_centeredAndNormalized(q, Minv_r, ts) == false)
		return false;
	if (parametersOfClippingPlaneIntersections(ts, p, r) == false)
	{
		ts[0] = -OUT_OF_BOUNDS;
		ts[1] = ts[0];
		return false;
	}
	if (ts[1] > ts[0])
		return true;
	else
		return false;
}

bool geometricObject::intersectionEndPoints_centeredAndNormalized(double* p, double* r, double* ts)
{
	if (ts == NULL)
		return false;
	ts[0] = -OUT_OF_BOUNDS;
	ts[1] = ts[0];

	// r != (0,0,1)
	double r_dot_r = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
	double p_dot_r = r[0] * p[0] + r[1] * p[1] + r[2] * p[2];
	//double t0 = (r[0]*p[0] + r[1]*p[1] + r[2]*p[2])/r_dot_r;
	/*
	if (type != phantom::CONE_Y && type != phantom::CONE_Z && type != phantom::CONE_X)
	{
		double t0 = -p_dot_r / r_dot_r;
		//if (fabs(p[0]+t0*r[0]) > 1.0 || fabs(p[1]+t0*r[1]) > 1.0 || fabs(p[2]+t0*r[2]) > 1.0)
		//if ((p[0]-t0*r[0])*(p[0]-t0*r[0]) + (p[1]-t0*r[1])*(p[1]-t0*r[1]) + (p[2]-t0*r[2])*(p[2]-t0*r[2]) > 3.0)
		if ((p[0] + t0 * r[0]) * (p[0] + t0 * r[0]) + (p[1] + t0 * r[1]) * (p[1] + t0 * r[1]) + (p[2] + t0 * r[2]) * (p[2] + t0 * r[2]) > 3.0)
			return false;
	}
	//*/

	if (type == phantom::ELLIPSOID)
	{
		//double p_dot_r = dot(p,r);
		//double p_dot_r = r[0] * p[0] + r[1] * p[1] + r[2] * p[2];
		//double p_dot_r = t0*r_dot_r;
		//double disc = p_dot_r * p_dot_r + r_dot_r * (1.0 - dot(p, p));
		double disc = p_dot_r * p_dot_r + r_dot_r * (1.0 - (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]));
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			ts[0] = (-p_dot_r - disc) / r_dot_r;
			ts[1] = (-p_dot_r + disc) / r_dot_r;
		}
		else
			return false;
	}
	else if (type == phantom::PARALLELEPIPED)
	{
		double tx[2];
		double ty[2];
		double tz[2];
		if (parametersOfIntersection_1D(&tx[0], p[0], r[0]) == true)
		{
			if (parametersOfIntersection_1D(&ty[0], p[1], r[1]) == true)
			{
				if (parametersOfIntersection_1D(&tz[0], p[2], r[2]) == true)
				{
					ts[0] = max(max(tx[0], ty[0]), tz[0]);
					ts[1] = min(min(tx[1], ty[1]), tz[1]);
				}
				else
					return false;
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (type == phantom::CYLINDER_Z)
	{
		//double r_dot_r_2D = r[0] * r[0] + r[1] * r[1]; // 3
		double r_dot_r_2D = r_dot_r - r[2] * r[2]; // 2
		double p_dor_r_2D = p_dot_r - p[2] * r[2]; // 2
		//double disc = (p[0] * r[0] + p[1] * r[1]) * (p[0] * r[0] + p[1] * r[1]) - r_dot_r_2D * (p[0] * p[0] + p[1] * p[1] - 1.0); // 13
		double disc = p_dor_r_2D * p_dor_r_2D - r_dot_r_2D * (p[0] * p[0] + p[1] * p[1] - 1.0); // 7
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			//double tmin = (-(p[0] * r[0] + r[1] * p[1]) - disc) / r_dot_r_2D; // 5
			//double tmax = (-(p[0] * r[0] + r[1] * p[1]) + disc) / r_dot_r_2D; // 5
			double tmin = (-p_dor_r_2D - disc) / r_dot_r_2D; // 2
			double tmax = (-p_dor_r_2D + disc) / r_dot_r_2D; // 2

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], p[2], r[2]) == true)
			{
				ts[0] = max(tmin, tz[0]);
				ts[1] = min(tmax, tz[1]);
			}
			else
				return false;
		}
		else if (r[0] == 0.0 && r[1] == 0.0 && p[0] * p[0] + p[1] * p[1] <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p[2], r[2]);
		}
		else
			return false;
	}
	else if (type == phantom::CYLINDER_X) // ellipsoidal cross sections parallel to x-y axis
	{
		//double r_dot_r_2D = r[2] * r[2] + r[1] * r[1];
		double r_dot_r_2D = r_dot_r - r[0] * r[0];
		double disc = (p[2] * r[2] + p[1] * r[1]) * (p[2] * r[2] + p[1] * r[1]) - r_dot_r_2D * (p[2] * p[2] + p[1] * p[1] - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-(p[2] * r[2] + r[1] * p[1]) - disc) / r_dot_r_2D;
			double tmax = (-(p[2] * r[2] + r[1] * p[1]) + disc) / r_dot_r_2D;

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], p[0], r[0]) == true)
			{
				ts[0] = max(tmin, tz[0]);
				ts[1] = min(tmax, tz[1]);
			}
			else
				return false;
		}
		else if (r[1] == 0.0 && r[2] == 0.0 && p[1] * p[1] + p[2] * p[2] <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p[0], r[0]);
		}
		else
			return false;
	}
	else if (type == phantom::CYLINDER_Y) // ellipsoidal cross sections parallel to x-y axis
	{
		//double r_dot_r_2D = r[0] * r[0] + r[2] * r[2];
		double r_dot_r_2D = r_dot_r - r[1] * r[1];
		double disc = (p[0] * r[0] + p[2] * r[2]) * (p[0] * r[0] + p[2] * r[2]) - r_dot_r_2D * (p[0] * p[0] + p[2] * p[2] - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-(p[0] * r[0] + r[2] * p[2]) - disc) / r_dot_r_2D;
			double tmax = (-(p[0] * r[0] + r[2] * p[2]) + disc) / r_dot_r_2D;

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], p[1], r[1]) == true)
			{
				ts[0] = max(tmin, tz[0]);
				ts[1] = min(tmax, tz[1]);
			}
			else
				return false;
		}
		else if (r[0] == 0.0 && r[2] == 0.0 && p[0] * p[0] + p[2] * p[2] <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p[1], r[1]);
		}
		else
			return false;
	}
	else if (type == phantom::CONE_Z)
	{
		double a = r[0] * r[0] + r[1] * r[1] - r[2] * r[2];
		double b_half = p[0] * r[0] + p[1] * r[1] - p[2] * r[2];
		double c = p[0] * p[0] + p[1] * p[1] - p[2] * p[2];
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-b_half - disc) / a;
			double tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			double theShift = 0.5 * (clipCone[1] + clipCone[0]);
			double theScale = 0.5 * (clipCone[1] - clipCone[0]);

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], (p[2] - theShift) / theScale, r[2] / theScale) == true)
			{
				ts[0] = max(tmin, tz[0]);
				ts[1] = min(tmax, tz[1]);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (type == phantom::CONE_X)
	{
		double a = r[2] * r[2] + r[1] * r[1] - r[0] * r[0];
		double b_half = p[2] * r[2] + p[1] * r[1] - p[0] * r[0];
		double c = p[2] * p[2] + p[1] * p[1] - p[0] * p[0];
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-b_half - disc) / a;
			double tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			double theShift = 0.5 * (clipCone[1] + clipCone[0]);
			double theScale = 0.5 * (clipCone[1] - clipCone[0]);

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], (p[0] - theShift) / theScale, r[0] / theScale) == true)
			{
				ts[0] = max(tmin, tz[0]);
				ts[1] = min(tmax, tz[1]);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (type == phantom::CONE_Y)
	{
		double a = r[0] * r[0] + r[2] * r[2] - r[1] * r[1];
		double b_half = p[0] * r[0] + p[2] * r[2] - p[1] * r[1];
		double c = p[0] * p[0] + p[2] * p[2] - p[1] * p[1];
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			double tmin = (-b_half - disc) / a;
			double tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			double theShift = 0.5 * (clipCone[1] + clipCone[0]);
			double theScale = 0.5 * (clipCone[1] - clipCone[0]);

			double tz[2];
			if (parametersOfIntersection_1D(&tz[0], (p[1] - theShift) / theScale, r[1] / theScale) == true)
			{
				if (fabs(r[1]) > 1.0e-12)
				{
					bool isInside_0 = false;
					double x_val, y_val, z_val;
					x_val = p[0] + tz[0] * r[0]; x_val *= x_val;
					y_val = p[1] + tz[0] * r[1]; y_val *= y_val;
					z_val = p[2] + tz[0] * r[2]; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_0 = true;

					bool isInside_1 = false;
					x_val = p[0] + tz[1] * r[0]; x_val *= x_val;
					y_val = p[1] + tz[1] * r[1]; y_val *= y_val;
					z_val = p[2] + tz[1] * r[2]; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_1 = true;
					if (isInside_0 == true)
					{
						if (isInside_1 == true)
						{
							//ts[0] = max(tmin, tz[0]);
							//ts[1] = min(tmax, tz[1]);
							ts[0] = tz[0];
							ts[1] = tz[1];
						}
						else
						{
							//insiders: tz[0], tmin, tmax
							if (tz[0] <= tmin && tmin <= tz[1])
							{
								ts[0] = tz[0];
								ts[1] = tmin;
							}
							else
							{
								ts[0] = tz[0];
								ts[1] = tmax;
							}
						}
					}
					else
					{
						if (isInside_1 == true)
						{
							//insiders: tz[1], tmin, tmax
							if (tz[0] <= tmin && tmin <= tz[1])
							{
								ts[0] = tmin;
								ts[1] = tz[1];
							}
							else
							{
								ts[0] = tmax;
								ts[1] = tz[1];
							}
						}
						else
						{
							// insiders: tmin, tmax
							if (tz[0] <= tmin && tmax <= tz[1])
							{
								ts[0] = tmin;
								ts[1] = tmax;
							}
							else
							{
								ts[0] = -OUT_OF_BOUNDS;
								ts[1] = ts[0];
								return false;
							}
						}
					}
				}
				else
				{
					ts[0] = tmin;
					ts[1] = tmax;
				}
			}
		}
	}
	else
	{
		ts[0] = -OUT_OF_BOUNDS;
		ts[1] = -OUT_OF_BOUNDS;
		return false;
	}

	if (ts[0] >= ts[1])
	{
		ts[0] = -OUT_OF_BOUNDS;
		ts[1] = ts[0];
		return false;
	}

	return true;
}

bool geometricObject::parametersOfIntersection_1D(double* ts, double p, double r)
{
	// finds ts such that p+t*r = +-1
	if (fabs(r) < 1e-12)
	{
		if (fabs(p) < 1.0)
		{
			ts[0] = -OUT_OF_BOUNDS;
			ts[1] = OUT_OF_BOUNDS;

			return true;
		}
		else
			return false;
	}
	else
	{
		if (r > 0.0)
		{
			ts[0] = (-1.0 - p) / r;
			ts[1] = (1.0 - p) / r;
		}
		else
		{
			ts[1] = (-1.0 - p) / r;
			ts[0] = (1.0 - p) / r;
		}
		return true;
	}
}

bool geometricObject::parametersOfClippingPlaneIntersections(double* ts, double* p, double* r)
{
	for (int i = 0; i < numClippingPlanes; i++)
	{
		double p_dot_n = clippingPlanes[i][0] * p[0] + clippingPlanes[i][1] * p[1] + clippingPlanes[i][2] * p[2];
		double r_dot_n = clippingPlanes[i][0] * r[0] + clippingPlanes[i][1] * r[1] + clippingPlanes[i][2] * r[2];
		if (fabs(r_dot_n) < 1.0e-12)
		{
			if (p_dot_n < clippingPlanes[i][3])
				return false;
		}
		else if (r_dot_n > 0.0)
		{
			double temp = (clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			// restriction: t > temp
			if (ts[1] < temp)
				return false;
			if (temp > ts[0])
				ts[0] = temp;
		}
		else
		{
			// restriction: t < temp
			double temp = (clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			if (ts[0] > temp)
				return false;
			if (ts[1] > temp)
				ts[1] = temp;
		}
	}

	return true;
}

double geometricObject::dot(double* x, double* y, int N)
{
	if (x == NULL || y == NULL || N <= 0)
		return 0.0;
	else
	{
		double retVal = x[0] * y[0];
		for (int i = 1; i < N; i++)
			retVal += x[i] * y[i];
		return retVal;
	}
}

bool phantom::scale_phantom(float scale_x, float scale_y, float scale_z)
{
	for (int i = 0; i < int(objects.size()); i++)
	{
		objects[i].centers[0] *= scale_x;
		objects[i].centers[1] *= scale_y;
		objects[i].centers[2] *= scale_z;
		objects[i].radii[0] *= scale_x;
		objects[i].radii[1] *= scale_y;
		objects[i].radii[2] *= scale_z;
		//objects[i].clipCone[0] *= scale_x;
		//objects[i].clipCone[1] *= scale_x;
		for (int n = 0; n < objects[i].numClippingPlanes; n++)
		{
			//objects[i].clippingPlanes[n][0] *= scale_x;
			objects[i].clippingPlanes[n][1] *= scale_y;
			objects[i].clippingPlanes[n][2] *= scale_z;
			objects[i].clippingPlanes[n][3] *= scale_x;
		}
	}
	return true;
}

bool phantom::synthesizeSymmetry(float* f_radial, float* f)
{
	if (params == NULL || f_radial == NULL || f == NULL)
		return false;

	// params->numX = 1
	double beta = -params->axisOfSymmetry;
	double cos_beta = cos(beta * PI / 180.0);
	double sin_beta = sin(beta * PI / 180.0);

	//printf("beta = %f\n", beta);

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int i = 0; i < params->numZ; i++)
	{
		float* zSlice = &f[uint64(i)*uint64(params->numY*params->numY)];

		float* zSlice_radial = &f_radial[uint64(i) * uint64(params->numY)];

		float z = params->z_samples(i);
		for (int j = 0; j < params->numY; j++)
		{
			float y = j * params->voxelWidth + params->y_0();
			float* xLine = &zSlice[uint64(j) * uint64(params->numY)];

			if (beta == 0.0)
			{
				for (int k = 0; k < params->numY; k++)
				{
					float x = k * params->voxelWidth + params->y_0();
					float r = sqrt(x * x + y * y);
					if (y < 0.0)
						r = -r;
					float r_ind = y_inv(r);

					int r_low = int(r_ind);
					int r_high = r_low + 1;
					float dr = r_ind - float(r_low);
					if (r_ind < 0.0)
					{
						r_low = 0;
						r_high = 0;
						dr = 0.0;
					}
					else if (r_ind >= params->numY - 1)
					{
						r_low = params->numY - 1;
						r_high = r_low;
						dr = 0.0;
					}

					xLine[k] = (1.0 - dr) * zSlice_radial[r_low] + dr * zSlice_radial[r_high];
				}
			}
			else
			{
				for (int k = 0; k < params->numY; k++)
				{
					float x = k * params->voxelWidth + params->y_0();

					// FIXME: check rotation
					float x_rot = x*cos_beta - z*sin_beta;
					float y_rot = y;
					float z_rot = x*sin_beta + z*cos_beta;

					float r = sqrt(x_rot * x_rot + y_rot * y_rot);
					if (y < 0.0)
						r = -r;

					float r_ind = y_inv(r);
					float z_ind = z_inv(z_rot);

					int r_low = int(r_ind);
					int r_high = r_low + 1;
					float dr = r_ind - float(r_low);
					if (r_ind < 0.0)
					{
						r_low = 0;
						r_high = 0;
						dr = 0.0;
					}
					else if (r_ind >= params->numY - 1)
					{
						r_low = params->numY - 1;
						r_high = r_low;
						dr = 0.0;
					}

					int z_low = int(z_ind);
					int z_high = z_low + 1;
					float dz = z_ind - float(z_low);
					if (z_ind < 0.0)
					{
						z_low = 0;
						z_high = 0;
						dz = 0.0;
					}
					else if (z_ind >= params->numZ - 1)
					{
						z_low = params->numZ - 1;
						z_high = z_low;
						dz = 0.0;
					}
					xLine[k] = (1.0 - dr) * ((1.0 - dz) * f_radial[r_low+z_low * params->numY] + dz * f_radial[r_low+z_high * params->numY])
						+ dr * ((1.0 - dz) * f_radial[r_high+z_low * params->numY] + dz * f_radial[r_high+z_high * params->numY]);
				}
			}
		}
	}
	return true;
}
