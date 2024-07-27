////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include "analytic_ray_tracing_gpu.cuh"
#include "analytic_ray_tracing.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.h"

#ifndef PI
#define PI 3.141592653589793f
#endif

#ifndef OUT_OF_BOUNDS
#define OUT_OF_BOUNDS 1.0e12f;
#endif

//enum geometry_list { CONE = 0, PARALLEL = 1, FAN = 2, MODULAR = 3, CONE_PARALLEL = 4 };

__constant__ int d_oversampling;
__constant__ int d_geometry;
__constant__ int d_CONE;
__constant__ int d_PARALLEL;
__constant__ int d_FAN;
__constant__ int d_MODULAR;
__constant__ int d_CONE_PARALLEL;
__constant__ int d_detectorType;
__constant__ int d_FLAT;
__constant__ int d_CURVED;
__constant__ float d_sod;
__constant__ float d_sdd;
__constant__ float d_tau;
__constant__ int4 d_N_g;
__constant__ float4 d_T_g;
__constant__ float4 d_startVal_g;

//enum objectType_list { ELLIPSOID = 0, PARALLELEPIPED = 1, CYLINDER_X = 2, CYLINDER_Y = 3, CYLINDER_Z = 4, CONE_X = 5, CONE_Y = 6, CONE_Z = 7 };
#define d_ELLIPSOID 0
#define d_PARALLELEPIPED 1
#define d_CYLINDER_X 2
#define d_CYLINDER_Y 3
#define d_CYLINDER_Z 4
#define d_CONE_X  5
#define d_CONE_Y 6
#define d_CONE_Z 7

__device__ float u(const int i)
{
    return float(i) * d_T_g.z + d_startVal_g.z;
    /*
    if (d_geometry == d_PARALLEL)
    {
        return 0.0f;
    }
    else if (d_geometry == d_FAN)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        return 0.0f;
    }
    else
        return 0.0f;
    //*/
}

__device__ float v(const int i)
{
    return float(i) * d_T_g.y + d_startVal_g.y;
    /*
    if (d_geometry == d_PARALLEL)
    {
        return 0.0f;
    }
    else if (d_geometry == d_FAN)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        return 0.0f;
    }
    else
        return 0.0f;
    //*/
}

__device__ float z_source(const int i, const int k)
{
    return float(i) * d_T_g.w + d_startVal_g.w;
    /*
    if (d_geometry == d_PARALLEL)
    {
        return 0.0f;
    }
    else if (d_geometry == d_FAN)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE)
    {
        return 0.0f;
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        return 0.0f;
    }
    else
        return 0.0f;
    //*/
}

__device__ float3 setSourcePosition(const float phi, const int iProj, const int iRow, const int iCol, const float dv, const float du)
{
    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

    if (d_geometry == d_PARALLEL)
    {
        return make_float3(-(u(iCol) + du) * sin_phi,
            (u(iCol) + du) * cos_phi,
            v(iRow) + dv);
    }
    else if (d_geometry == d_FAN)
    {
        return make_float3(d_sod * cos_phi + d_tau * sin_phi,
            d_sod * sin_phi - d_tau * cos_phi,
            v(iRow) + dv);
    }
    else if (d_geometry == d_CONE)
    {
        return make_float3(d_sod * cos_phi + d_tau * sin_phi,
            d_sod * sin_phi - d_tau * cos_phi,
            z_source(iProj, 0));
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        const float s = u(iCol) + du;
        const float sqrt_R2_minus_s2 = sqrtf(d_sod * d_sod - s * s);
        return make_float3(-s * sin_phi + sqrt_R2_minus_s2 * cos_phi,
            s * cos_phi + sqrt_R2_minus_s2 * sin_phi,
            z_source(iProj, iCol));
    }
    else
        return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float3 setTrajectory(const float phi, const int iProj, const int iRow, const int iCol, const float dv, const float du)
{
    const float u_val = u(iCol) + du;
    const float v_val = v(iRow) + dv;

    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

    /*
    float* s = NULL;
    float* c = NULL;
    float* u_vec = NULL;
    float* v_vec = NULL;
    //*/

    if (d_geometry == d_PARALLEL)
    {
        return make_float3(-cos_phi, -sin_phi, 0.0f);
    }
    else if (d_geometry == d_FAN)
    {
        return make_float3(-(cos_phi + u_val * sin_phi), -(sin_phi - u_val * cos_phi), 0.0f);
    }
    else if (d_geometry == d_CONE)
    {
        if (d_detectorType == d_CURVED)
        {
            return make_float3(-cos(phi - u_val), -sin(phi - u_val), v_val);
        }
        else
        {
            return make_float3(-(cos_phi + u_val * sin_phi), -(sin_phi - u_val * cos_phi), v_val);
        }
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        return make_float3(-cos_phi, -sin_phi, v_val);
    }
    else
        return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ bool parametersOfIntersection_1D(float2& ts, float p, float r)
{
	// finds ts such that p+t*r = +-1
	if (fabs(r) < 1e-12f)
	{
		if (fabs(p) < 1.0f)
		{
			ts.x = -OUT_OF_BOUNDS;
			ts.y = OUT_OF_BOUNDS;

			return true;
		}
		else
			return false;
	}
	else
	{
		if (r > 0.0f)
		{
			ts.x = (-1.0f - p) / r;
			ts.y = (1.0f - p) / r;
		}
		else
		{
			ts.y = (-1.0f - p) / r;
			ts.x = (1.0f - p) / r;
		}
		return true;
	}
}


__device__ bool parametersOfClippingPlaneIntersections(float2& ts, float3 p, float3 r, geometricSolid* solid)
{
	for (int i = 0; i < solid->numClippingPlanes; i++)
	{
		float p_dot_n = solid->clippingPlanes[i][0] * p.x + solid->clippingPlanes[i][1] * p.y + solid->clippingPlanes[i][2] * p.z;
		float r_dot_n = solid->clippingPlanes[i][0] * r.x + solid->clippingPlanes[i][1] * r.y + solid->clippingPlanes[i][2] * r.z;
		if (fabs(r_dot_n) < 1.0e-12f)
		{
			if (p_dot_n < solid->clippingPlanes[i][3])
				return false;
		}
		else if (r_dot_n > 0.0f)
		{
			float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			// restriction: t > temp
			if (ts.y < temp)
				return false;
			if (temp > ts.x)
				ts.x = temp;
		}
		else
		{
			// restriction: t < temp
			float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			if (ts.x > temp)
				return false;
			if (ts.y > temp)
				ts.y = temp;
		}
	}

	return true;
}

__device__ bool intersectionEndPoints_centeredAndNormalized(float3& p, float3& r, float2& ts, geometricSolid* solid)
{
	ts.x = -OUT_OF_BOUNDS;
	ts.y = ts.x;

	// r != (0,0,1)
	float r_dot_r = r.x * r.x + r.y * r.y + r.z * r.z;
	float p_dot_r = r.x * p.x + r.y * p.y + r.z * p.z;

	if (solid->type == d_ELLIPSOID)
	{
		float disc = p_dot_r * p_dot_r + r_dot_r * (1.0f - (p.x * p.x + p.y * p.y + p.z * p.z));
		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			ts.x = (-p_dot_r - disc) / r_dot_r;
			ts.y = (-p_dot_r + disc) / r_dot_r;
		}
		else
			return false;
	}
	else if (solid->type == d_PARALLELEPIPED)
	{
		float2 tx;
		float2 ty;
		float2 tz;
		if (parametersOfIntersection_1D(tx, p.x, r.x) == true)
		{
			if (parametersOfIntersection_1D(ty, p.y, r.y) == true)
			{
				if (parametersOfIntersection_1D(tz, p.z, r.z) == true)
				{
					ts.x = max(max(tx.x, ty.x), tz.x);
					ts.y = min(min(tx.y, ty.y), tz.y);
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
	else if (solid->type == d_CYLINDER_Z)
	{
		//double r_dot_r_2D = r.x * r.x + r.y * r.y; // 3
		float r_dot_r_2D = r_dot_r - r.z * r.z; // 2
		float p_dor_r_2D = p_dot_r - p.z * r.z; // 2
		float disc = p_dor_r_2D * p_dor_r_2D - r_dot_r_2D * (p.x * p.x + p.y * p.y - 1.0f); // 7
		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-p_dor_r_2D - disc) / r_dot_r_2D; // 2
			float tmax = (-p_dor_r_2D + disc) / r_dot_r_2D; // 2

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.z, r.z) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.x == 0.0f && r.y == 0.0f && p.x * p.x + p.y * p.y <= 1.0f)
		{
			return parametersOfIntersection_1D(ts, p.z, r.z);
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_X) // ellipsoidal cross sections parallel to x-y axis
	{
		float r_dot_r_2D = r_dot_r - r.x * r.x;
		float disc = (p.z * r.z + p.y * r.y) * (p.z * r.z + p.y * r.y) - r_dot_r_2D * (p.z * p.z + p.y * p.y - 1.0f);
		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-(p.z * r.z + r.y * p.y) - disc) / r_dot_r_2D;
			float tmax = (-(p.z * r.z + r.y * p.y) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.x, r.x) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.y == 0.0f && r.z == 0.0f && p.y * p.y + p.z * p.z <= 1.0f)
		{
			return parametersOfIntersection_1D(ts, p.x, r.x);
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_Y) // ellipsoidal cross sections parallel to x-y axis
	{
		float r_dot_r_2D = r_dot_r - r.y * r.y;
		float disc = (p.x * r.x + p.z * r.z) * (p.x * r.x + p.z * r.z) - r_dot_r_2D * (p.x * p.x + p.z * p.z - 1.0f);
		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-(p.x * r.x + r.z * p.z) - disc) / r_dot_r_2D;
			float tmax = (-(p.x * r.x + r.z * p.z) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.y, r.y) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.x == 0.0f && r.z == 0.0f && p.x * p.x + p.z * p.z <= 1.0f)
		{
			return parametersOfIntersection_1D(ts, p.y, r.y);
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_Z)
	{
		float a = r.x * r.x + r.y * r.y - r.z * r.z;
		float b_half = p.x * r.x + p.y * r.y - p.z * r.z;
		float c = p.x * p.x + p.y * p.y - p.z * p.z;
		float disc = b_half * b_half - a * c;

		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			float theShift = 0.5f * (solid->clipCone.y + solid->clipCone.x);
			float theScale = 0.5f * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.z - theShift) / theScale, r.z / theScale) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_X)
	{
		float a = r.z * r.z + r.y * r.y - r.x * r.x;
		float b_half = p.z * r.z + p.y * r.y - p.x * r.x;
		float c = p.z * p.z + p.y * p.y - p.x * p.x;
		float disc = b_half * b_half - a * c;

		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			float theShift = 0.5f * (solid->clipCone.y + solid->clipCone.x);
			float theScale = 0.5f * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.x - theShift) / theScale, r.x / theScale) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_Y)
	{
		float a = r.x * r.x + r.z * r.z - r.y * r.y;
		float b_half = p.x * r.x + p.z * r.z - p.y * r.y;
		float c = p.x * p.x + p.z * p.z - p.y * p.y;
		float disc = b_half * b_half - a * c;

		if (disc > 0.0f)
		{
			disc = sqrtf(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			float theShift = 0.5f * (solid->clipCone.y + solid->clipCone.x);
			float theScale = 0.5f * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.y - theShift) / theScale, r.y / theScale) == true)
			{
				if (fabs(r.y) > 1.0e-12)
				{
					bool isInside_0 = false;
					float x_val, y_val, z_val;
					x_val = p.x + tz.x * r.x; x_val *= x_val;
					y_val = p.y + tz.x * r.y; y_val *= y_val;
					z_val = p.z + tz.x * r.z; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_0 = true;

					bool isInside_1 = false;
					x_val = p.x + tz.y * r.x; x_val *= x_val;
					y_val = p.y + tz.y * r.y; y_val *= y_val;
					z_val = p.z + tz.y * r.z; z_val *= z_val;
					if (x_val + z_val <= y_val)
						isInside_1 = true;
					if (isInside_0 == true)
					{
						if (isInside_1 == true)
						{
							//ts.x = max(tmin, tz.x);
							//ts.y = min(tmax, tz.y);
							ts.x = tz.x;
							ts.y = tz.y;
						}
						else
						{
							//insiders: tz.x, tmin, tmax
							if (tz.x <= tmin && tmin <= tz.y)
							{
								ts.x = tz.x;
								ts.y = tmin;
							}
							else
							{
								ts.x = tz.x;
								ts.y = tmax;
							}
						}
					}
					else
					{
						if (isInside_1 == true)
						{
							//insiders: tz.y, tmin, tmax
							if (tz.x <= tmin && tmin <= tz.y)
							{
								ts.x = tmin;
								ts.y = tz.y;
							}
							else
							{
								ts.x = tmax;
								ts.y = tz.y;
							}
						}
						else
						{
							// insiders: tmin, tmax
							if (tz.x <= tmin && tmax <= tz.y)
							{
								ts.x = tmin;
								ts.y = tmax;
							}
							else
							{
								ts.x = -OUT_OF_BOUNDS;
								ts.y = ts.x;
								return false;
							}
						}
					}
				}
				else
				{
					ts.x = tmin;
					ts.y = tmax;
				}
			}
		}
	}
	else
	{
		ts.x = -OUT_OF_BOUNDS;
		ts.y = -OUT_OF_BOUNDS;
		return false;
	}

	if (ts.x >= ts.y)
	{
		ts.x = -OUT_OF_BOUNDS;
		ts.y = ts.x;
		return false;
	}

	return true;
}

__device__ bool intersectionEndPoints(float3& p, float3& r, float2& ts, geometricSolid* solid)
{
    // assume ||r|| == 1 && r != (0,0,1) && axis[i] > 0 for i=0,1,2
    // alpha is rotation around x-y axis, currently there is no rotation for x-z or y-z axes
    float3 q;
    float3 Minv_r;

    if (solid->isRotated == false)
    {
        // Scale; (9,0) ops
        q.x = (p.x - solid->centers.x) / solid->radii.x;
        q.y = (p.y - solid->centers.y) / solid->radii.y;
        q.z = (p.z - solid->centers.z) / solid->radii.z;

        Minv_r.x = r.x / solid->radii.x;
        Minv_r.y = r.y / solid->radii.y;
        Minv_r.z = r.z / solid->radii.z;
    }
    else
    {
        float3 temp;

        // Shift
        q.x = p.x - solid->centers.x;
        q.y = p.y - solid->centers.y;
        q.z = p.z - solid->centers.z;

        // Rotate and Scale; (36, 0) ops
        temp.x = (q.x * solid->A[0 * 3 + 0] + q.y * solid->A[0 * 3 + 1] + q.z * solid->A[0 * 3 + 2]) / solid->radii.x;
        temp.y = (q.x * solid->A[1 * 3 + 0] + q.y * solid->A[1 * 3 + 1] + q.z * solid->A[1 * 3 + 2]) / solid->radii.y;
        temp.z = (q.x * solid->A[2 * 3 + 0] + q.y * solid->A[2 * 3 + 1] + q.z * solid->A[2 * 3 + 2]) / solid->radii.z;

        q.x = temp.x;
        q.y = temp.y;
        q.z = temp.z;

        Minv_r.x = (r.x * solid->A[0 * 3 + 0] + r.y * solid->A[0 * 3 + 1] + r.z * solid->A[0 * 3 + 2]) / solid->radii.x;
        Minv_r.y = (r.x * solid->A[1 * 3 + 0] + r.y * solid->A[1 * 3 + 1] + r.z * solid->A[1 * 3 + 2]) / solid->radii.y;
        Minv_r.z = (r.x * solid->A[2 * 3 + 0] + r.y * solid->A[2 * 3 + 1] + r.z * solid->A[2 * 3 + 2]) / solid->radii.z;
    }

    if (intersectionEndPoints_centeredAndNormalized(q, Minv_r, ts, solid) == false)
        return false;
    if (parametersOfClippingPlaneIntersections(ts, p, r, solid) == false)
    {
        ts.x = -OUT_OF_BOUNDS;
        ts.y = ts.x;
        return false;
    }
    if (ts.y > ts.x)
        return true;
    else
        return false;
}

__device__ void sort(float* v, int count)
{
	for (int i = 0; i < count; i++)
	{
		for (int j = i + 1; j < count; j++)
		{
			if (v[i] > v[j])
			{  // swap?
				const float tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}
}

__device__ float lineIntegral_geometricSolids(float3 p, float3 r, geometricSolid* solids, const int numObjects, float* floatData, int* intData)
{
	//*
	//vector<float> endPoints;
	//vector<int> objectIndices;
	//float* endPoints = (float*)malloc(size_t(2 * numObjects) * sizeof(float));
	//int* objectIndices = (int*)malloc(size_t(numObjects) * sizeof(int));
	//float* intersection_0 = (float*)malloc(size_t(2*numObjects) * sizeof(float));
	//float* intersection_1 = &intersection_0[numObjects];

	float* endPoints = &floatData[0];
	float* intersection_0 = &floatData[2 * numObjects];
	float* intersection_1 = &floatData[3 * numObjects];
	int* objectIndices = intData;

	int count = 0;
	//float* intersection_0 = &intersections[0];
	//float* intersection_1 = &intersections[numObjects];
	for (int i = 0; i < numObjects; i++)
	{
		float2 ts;
		if (intersectionEndPoints(p, r, ts, &solids[i]))
		{
			endPoints[2 * count + 0] = ts.x;
			endPoints[2 * count + 1] = ts.y;
			//endPoints.push_back(ts.x);
			//endPoints.push_back(ts.y);
			intersection_0[i] = ts.x;
			intersection_1[i] = ts.y;
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
	float retVal = 0.0f;
	if (count > 0)
	{
		sort(endPoints, 2 * count);
		//sort(endPoints.begin(), endPoints.end());
		for (int i = 0; i < numObjects - 1; i++)
		{
			// Consider the interval (allPoints[i], allPoints[i+1])
			float midPoint = (endPoints[i + 1] + endPoints[i]) * 0.5f;
			//for (int j = int(objects.size())-1; j >= 0; j--)
			for (int ind = numObjects - 1; ind >= 0; ind--)
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
						retVal += solids[j].val * (endPoints[i + 1] - endPoints[i]);
						break;
					}
				}
			}
		}
	}
	//free(endPoints);
	//free(objectIndices);
	//free(intersection_0);
	return retVal;
	//*/
}

__global__ void rayTracingKernel(float* g, const float* phis, geometricSolid* solids, const int numObjects, float* floatData, int* intData, const int i, const int j)
{
    //const int i = threadIdx.x + blockIdx.x * blockDim.x;
    //const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //const int k = threadIdx.z + blockIdx.z * blockDim.z;
	const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
        return;

    const float dv = 0.0f;
    const float du = 0.0f;
    const float phi = phis[i];
    const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
    const float3 r = setTrajectory(phi, i, j, k, dv, du);

    //const float val = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &allIntersections[k*2*numObjects]);
	const float val = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[k*4*numObjects], &intData[k * numObjects]);
	g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = val;
}

void setConstantMemoryGeometryParameters(parameters* params, int oversampling)
{
	cudaError_t cudaStatus;

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

	cudaMemcpyToSymbol(d_oversampling, &oversampling, sizeof(int));

    int geometry = params->geometry;
    int CONE = params->CONE;
    int PARALLEL = params->PARALLEL;
    int FAN = params->FAN;
    int MODULAR = params->MODULAR;
    int CONE_PARALLEL = params->CONE_PARALLEL;
    int detectorType = params->detectorType;
    int FLAT = params->FLAT;
    int CURVED = params->CURVED;
	cudaStatus = cudaMemcpyToSymbol(d_geometry, &geometry, sizeof(int));
    cudaMemcpyToSymbol(d_CONE, &CONE, sizeof(int));
    cudaMemcpyToSymbol(d_PARALLEL, &PARALLEL, sizeof(int));
    cudaMemcpyToSymbol(d_FAN, &FAN, sizeof(int));
    cudaMemcpyToSymbol(d_MODULAR, &MODULAR, sizeof(int));
    cudaMemcpyToSymbol(d_CONE_PARALLEL, &CONE_PARALLEL, sizeof(int));

    cudaMemcpyToSymbol(d_detectorType, &detectorType, sizeof(int));
    cudaMemcpyToSymbol(d_FLAT, &FLAT, sizeof(int));
    cudaMemcpyToSymbol(d_CURVED, &CURVED, sizeof(int));

    float sod = params->sod;
    float sdd = params->sdd;
    float tau = params->tau;
    cudaMemcpyToSymbol(d_sod, &sod, sizeof(float));
    cudaMemcpyToSymbol(d_sdd, &sdd, sizeof(float));
    cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaStatus = cudaMemcpyToSymbol(d_N_g, &N_g, sizeof(int4));
    cudaMemcpyToSymbol(d_T_g, &T_g, sizeof(float4));
    cudaMemcpyToSymbol(d_startVal_g, &startVal_g, sizeof(float4));
}

bool rayTrace_gpu(float* g, parameters* params, phantom* aPhantom, bool data_on_cpu, int oversampling)
{
    if (g == NULL || params == NULL || params->geometryDefined() == false)
        return false;
	oversampling = max(1, min(oversampling, 9));

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

	setConstantMemoryGeometryParameters(params, oversampling);

    float* dev_g = 0;
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

    float* dev_phis = copyAngleArrayToGPU(params);

	int numObjects = int(aPhantom->objects.size());
	geometricSolid* dev_solids = 0;
	geometricSolid* solids = new geometricSolid[numObjects];
	for (int i = 0; i < numObjects; i++)
	{
		solids[i].type = aPhantom->objects[i].type;
		solids[i].centers = make_float3(aPhantom->objects[i].centers[0], aPhantom->objects[i].centers[1], aPhantom->objects[i].centers[2]);
		solids[i].radii = make_float3(aPhantom->objects[i].radii[0], aPhantom->objects[i].radii[1], aPhantom->objects[i].radii[2]);
		solids[i].val = aPhantom->objects[i].val;
		for (int j = 0; j < 9; j++)
			solids[i].A[j] = aPhantom->objects[i].A[j];
		for (int j = 0; j < 6; j++)
		{
			for (int k = 0; k < 4; k++)
				solids[i].clippingPlanes[j][k] = aPhantom->objects[i].clippingPlanes[j][k];
		}
		solids[i].isRotated = aPhantom->objects[i].isRotated;
		solids[i].numClippingPlanes = aPhantom->objects[i].numClippingPlanes;
		solids[i].clipCone.x = aPhantom->objects[i].clipCone[0];
		solids[i].clipCone.y = aPhantom->objects[i].clipCone[1];
	}
	if ((cudaStatus = cudaMalloc((void**)&dev_solids, numObjects * sizeof(geometricSolid))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}
	//if ((cudaStatus = cudaMemcpy(dev_g, g, N * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	if ((cudaStatus = cudaMemcpy(dev_solids, solids, numObjects * sizeof(geometricSolid), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "failed to copy phantom data to device!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}
	delete[] solids;

	float* dev_floatData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_floatData, N_g.z*4*numObjects * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}
	int* dev_intData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_intData, N_g.z*numObjects * sizeof(int))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}

    //dim3 dimBlock = setBlockSize(N_g);
    //dim3 dimGrid = setGridSize(N_g, dimBlock);

	int blockSize = 8;
	int numBlocks = int(ceil(double(N_g.z) / double(blockSize)));

	for (int i = 0; i < N_g.x; i++)
	{
		for (int j = 0; j < N_g.y; j++)
			rayTracingKernel <<< numBlocks, blockSize >>> (dev_g, dev_phis, dev_solids, numObjects, dev_floatData, dev_intData, i, j);
		//float* g, const float* phis, geometricSolid* solids, const int numObjects, float* allIntersections
	}

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

    // Clean up
    cudaFree(dev_phis);
	cudaFree(dev_solids);
	cudaFree(dev_floatData);
	cudaFree(dev_intData);
    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
    }

    return true;
}
