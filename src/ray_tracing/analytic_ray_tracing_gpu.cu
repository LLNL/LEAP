////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CT simulation via analytic ray tracing
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cmath>
#include "analytic_ray_tracing_gpu.cuh"
#include "analytic_ray_tracing.h"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.h"
#include "physics/xrayphysics_c_interface.h"

#ifndef PI
#define PI 3.141592653589793f
#endif

#ifndef OUT_OF_BOUNDS
#define OUT_OF_BOUNDS 1.0e12f;
#endif

#define NUM_QUADS 10
#define MAX_MESH_INTERSECTIONS 64
#define MAX_MESH_INTERSECTIONS_BACKUP 128
//#define MAX_MESH_INTERSECTIONS 2

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
__constant__ float d_cos_tilt;
__constant__ float d_sin_tilt;
__constant__ float d_cos_pitch;
__constant__ float d_sin_pitch;
__constant__ int4 d_N_g;
__constant__ float4 d_T_g;
__constant__ float4 d_startVal_g;
__constant__ float d_source_height;
__constant__ float d_source_width;
__constant__ float d_u_quad_pitch;
__constant__ float d_u_quad_pitch_inv;
__constant__ float d_u_quad_start;
__constant__ float d_v_quad_pitch;
__constant__ float d_v_quad_pitch_inv;
__constant__ float d_v_quad_start;

__constant__ int d_NUM_MATERIAL_TYPES;
__constant__ int d_NUM_ENERGIES;

//enum objectType_list { ELLIPSOID = 0, PARALLELEPIPED = 1, CYLINDER_X = 2, CYLINDER_Y = 3, CYLINDER_Z = 4, CONE_X = 5, CONE_Y = 6, CONE_Z = 7 };
#define d_ELLIPSOID 0
#define d_PARALLELEPIPED 1
#define d_CYLINDER_X 2
#define d_CYLINDER_Y 3
#define d_CYLINDER_Z 4
#define d_CONE_X  5
#define d_CONE_Y 6
#define d_CONE_Z 7

__device__ float ray_triangle_intersection(const float3 p, const float3 r, const float* triangles, const int numTriangles)
{
	const float eps = 0.000001f;
	// const float eps = 0.000000001f;

	//printf("p = (%f, %f, %f); r = (%f, %f, %f)\n", p.x, p.y, p.z, r.x, r.y, r.z);

	float ts[MAX_MESH_INTERSECTIONS];
	int ind = 0;
	for (int i = 0; i < numTriangles; i++)  // 51 ops, 6 comparisons per loop
	{
		const float3 v1 = make_float3(triangles[i * 9 + 0], triangles[i * 9 + 1], triangles[i * 9 + 2]);
		const float3 v2 = make_float3(triangles[i * 9 + 3], triangles[i * 9 + 4], triangles[i * 9 + 5]);
		const float3 v3 = make_float3(triangles[i * 9 + 6], triangles[i * 9 + 7], triangles[i * 9 + 8]);

		const float3 edge1 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);  // 3
		const float3 edge2 = make_float3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);  // 3

		// pvec = cross(r, edge2);
		const float3 pvec = make_float3(r.y * edge2.z - r.z * edge2.y, r.z * edge2.x - r.x * edge2.z, r.x * edge2.y - r.y * edge2.x);            // 9
		const float det = edge1.x * pvec.x + edge1.y * pvec.y + edge1.z * pvec.z;  // 5
		if (fabs(det) < eps)
			continue;

		const float inv_det = 1.0f / det;                                                 // 1
		const float3 tvec = make_float3(p.x - v1.x, p.y - v1.y, p.z - v1.z);              // 3
		const float u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * inv_det;  // 6
		if (u < 0.0f || u > 1.0f)
			continue;

		// qvec = cross(tvec, edge1);
		const float3 qvec = make_float3(tvec.y * edge1.z - tvec.z * edge1.y, tvec.z * edge1.x - tvec.x * edge1.z, tvec.x * edge1.y - tvec.y * edge1.x);    // 9
		const float v = (r.x * qvec.x + r.y * qvec.y + r.z * qvec.z) * inv_det;  // 6
		if (v < 0.0f || u + v > 1.0f)
			continue;

		const float t = (edge2.x * qvec.x + edge2.y * qvec.y + edge2.z * qvec.z) * inv_det;  // 6
		//if (fabs(t) >= eps /* && minValue <= t && t <= maxValue*/)
		{
			//printf("intersection at t = %f for triangle %d\n", t, i);
			ts[ind] = t;
			ind += 1;
			if (ind >= MAX_MESH_INTERSECTIONS)
				break;
		}
	}

	if (ind == 0)
	{
		return 0.0f;
	}
	else if (ind == 1)
	{
		//return -1.0f;
		//return 0.0f;
		return NAN;
	}
	else if (ind == 2)
	{
		return fabs(ts[1] - ts[0]);
	}
	else
	{
		// bubble-sort
		for (int i = 0; i < ind; i++)
		{
			for (int j = i + 1; j < ind; j++)
			{
				if (ts[i] > ts[j])
				{
					// swap?
					const float tmp = ts[i];
					ts[i] = ts[j];
					ts[j] = tmp;
				}
			}
		}


		float accum = 0.0f;
		int i = 0;
		int j = 1;
		while (j < ind)
		{
			if (fabs(ts[i]-ts[j]) > eps)
			{
				accum += fabs(ts[i]-ts[j]);
				i = j+1;
				j = j+2;
			}
			else
				j += 1;
		}
		return accum;

		/*
		if (ind == 4)
		{
			return (ts[1] - ts[0]) + (ts[3] - ts[2]);
		}
		else if (ind == 6)
		{
			return (ts[1] - ts[0]) + (ts[3] - ts[2]) + (ts[5] - ts[4]);
		}
		else if (ind % 2 == 0)
		{  // if (ind == 8)
			float accum = 0.0f;
			for (int i = 0; i < ind-1; i+=2)
				accum += ts[i+1]-ts[i];
			return accum;
			//return (ts[1] - ts[0]) + (ts[3] - ts[2]) + (ts[5] - ts[4]) + (ts[7] - ts[6]);
		}
		else
		{
			//return -1.0f * ts[0];
			return 0.0f;
		}
		//*/
	}
}

__device__ inline float dot3(const float3 a, const float3 b)
{
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ inline float3 cross3(const float3 a, const float3 b)
{
    return make_float3(
        fmaf(a.y, b.z, -a.z * b.y),
        fmaf(a.z, b.x, -a.x * b.z),
        fmaf(a.x, b.y, -a.y * b.x)
    );
}

__device__ bool isinside(const float3 p, const float3* __restrict__ triangles, const int numTriangles, float4* __restrict__ AABB, float* __restrict__ upper_z)
{
	const float eps = 0.000001f;
	const float3 r = make_float3(0.0f, 0.0f, 1.0f);
	int num_itersections = 0;
	for (int i = 0; i < numTriangles; i++)
	{
		if (p.x < AABB[i].w || p.x > AABB[i].x || p.y < AABB[i].y || p.y > AABB[i].z || p.z > upper_z[i])
			continue;

		const float3 v1 = triangles[i*3 + 0];
		const float3 v2 = triangles[i*3 + 1];
		const float3 v3 = triangles[i*3 + 2];
		
		/*
		if (p.y < min(min(v1.y, v2.y), v3.y))
			continue;
		if (p.y > max(max(v1.y, v2.y), v3.y))
			continue;
		if (p.z < min(min(v1.z, v2.z), v3.z))
			continue;
		if (p.z > max(max(v1.z, v2.z), v3.z))
			continue;
		//*/

		const float3 edge1 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);  // 3
		const float3 edge2 = make_float3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);  // 3

		const float3 pvec = cross3(r, edge2); // 9
		const float det = dot3(edge1, pvec); // 5
		if (fabs(det) < eps)
			continue;

		const float inv_det = 1.0f / det;                                                 // 1
		const float3 tvec = make_float3(p.x - v1.x, p.y - v1.y, p.z - v1.z);              // 3
		const float u = dot3(tvec, pvec) * inv_det; // 6
		if (u < 0.0f || u > 1.0f)
			continue;

		const float3 qvec = cross3(tvec, edge1); // 9
		const float v = dot3(r, qvec) * inv_det; // 6
		if (v < 0.0f || u + v > 1.0f)
			continue;

		if (dot3(edge2, qvec) * inv_det > 0.0f) // 6
			num_itersections += 1;
	}
	if (num_itersections % 2 == 0)
		return false;
	else
		return true;
}

__global__ void voxelizeKernel(float* __restrict__ f, const int4 N, const float4 T, const float4 startVal, const float3* __restrict__ triangles, const int numTriangles, const float val, const float4 lower_corner, const float4 upper_corner, float4* __restrict__ AABB, float* __restrict__ upper_z, const int oversampling)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= N.x || j >= N.y || k >= N.z)
        return;

	const float3 vox = make_float3(T.x*i + startVal.x, T.y*j + startVal.y, T.z*k + startVal.z);
	uint64 ind = uint64(k) * uint64(N.y * N.x) + uint64(j * N.x + i);
	if (vox.x < lower_corner.x || vox.x > upper_corner.x || vox.y < lower_corner.y || vox.y > upper_corner.y || vox.z < lower_corner.z || vox.z > upper_corner.z)
		f[ind] = 0.0f;
	else
	{
		if (oversampling <= 1)
		{
			if (isinside(vox, triangles, numTriangles, AABB, upper_z))
				f[ind] = val;
			else
				f[ind] = 0.0f;
		}
		else
		{
			const float d = 0.25f*T.x;
			int count = 0;
			for (int idx = -1; idx <= 1; idx++)
			{
				for (int idy = -1; idy <= 1; idy++)
				{
					for (int idz = -1; idz <= 1; idz++)
					{
						const float3 dvox = make_float3(vox.x + idx*d, vox.y + idy*d, vox.z + idz*d);
						if (isinside(dvox, triangles, numTriangles, AABB, upper_z))
							count += 1;
					}
				}
			}
			f[ind] = val * float(count) / 27.0f;
		}
	}
}

__device__ float ray_triangle_intersection_with_AABB_backup(const float3 p, const float3 r, const float3* __restrict__ triangles, const int numTriangles, const float v_coord, const float u_coord, const float4* AABB, const int* quads, const int* counts)
{
	const int u_quad_ind = max(0, min(NUM_QUADS-1, int(0.5f + (u_coord - d_u_quad_start) * d_u_quad_pitch_inv)));
	const int v_quad_ind = max(0, min(NUM_QUADS-1, int(0.5f + (v_coord - d_v_quad_start) * d_v_quad_pitch_inv)));
	const int count = counts[v_quad_ind*NUM_QUADS + u_quad_ind];
	if (count <= 0)
		return 0.0f;
	const int* inds = &quads[(v_quad_ind*NUM_QUADS + u_quad_ind)*numTriangles];

	const float min_width = 0.000000001f;
	//const float min_width = 0.00001f;
	const float eps = 0.000001f;
	// const float eps = 0.000000001f;

	float ts[MAX_MESH_INTERSECTIONS_BACKUP];
	int ind = 0;
	for (int ii = 0; ii < count; ii++)
	{
		int i = inds[ii];
		if (AABB[i].w <= u_coord && u_coord <= AABB[i].x && AABB[i].y <= v_coord && v_coord <= AABB[i].z)
		{
			const float3 v1 = triangles[i*3 + 0];
			const float3 v2 = triangles[i*3 + 1];
			const float3 v3 = triangles[i*3 + 2];

			const float3 edge1 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);  // 3
			const float3 edge2 = make_float3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);  // 3

			const float3 pvec = cross3(r, edge2); // 9
			const float det = dot3(edge1, pvec); // 5
			if (fabs(det) < eps)
				continue;

			const float inv_det = 1.0f / det;                                                 // 1
			const float3 tvec = make_float3(p.x - v1.x, p.y - v1.y, p.z - v1.z);              // 3
			const float u = dot3(tvec, pvec) * inv_det; // 6
			if (u < 0.0f || u > 1.0f)
				continue;

			const float3 qvec = cross3(tvec, edge1); // 9
			const float v = dot3(r, qvec) * inv_det; // 6
			if (v < 0.0f || u + v > 1.0f)
				continue;

			const float t = dot3(edge2, qvec) * inv_det; // 6

			// insert x-ray hit distance
			if (ind == 0)
			{
				ts[0] = t;
				ind = 1;
			}
			else if (ind == 1)
			{
				if (ts[0]+min_width < t)
				{
					ts[1] = t;
					ind = 2;
				}
				else if (t+min_width < ts[0])
				{
					ts[1] = ts[0];
					ts[0] = t;
					ind = 2;
				}
			}
			else
			{
				int pos = 0;
				while (pos < ind && ts[pos] < t)
					pos++;
				// ts[pos-1] <= t <= ts[pos]
				if ((pos == ind || t + min_width < ts[pos]) && (pos == 0 || ts[pos-1]+min_width < t))
				{
					// shift elements to the right
					for (int j = ind; j > pos; j--)
						ts[j] = ts[j - 1];

					// insert new element
					ts[pos] = t;
					ind++;
					if (ind >= MAX_MESH_INTERSECTIONS_BACKUP)
					{
						//printf("WARNING: maximum number of intersections exceeded!\n");
						break;
					}
				}
			}
		}
	}

	if (ind == 0)
	{
		return 0.0f;
	}
	else if ((ind & 1) != 0)
	{
		return NAN;
	}
	else if (ind == 2)
	{
		return ts[1] - ts[0];
	}
	else
	{
		float accum = ts[1]-ts[0];
		for (int i = 2; i < ind-1; i+=2)
		{
			//if (i+1 < ind)
				accum += ts[i+1]-ts[i];
		}
		return accum;
	}
}

__device__ float ray_triangle_intersection_with_AABB(const float3 p, const float3 r, const float3* __restrict__ triangles, const int numTriangles, const float v_coord, const float u_coord, const float4* AABB, const int* quads, const int* counts)
{
	const int u_quad_ind = max(0, min(NUM_QUADS-1, int(0.5f + (u_coord - d_u_quad_start) * d_u_quad_pitch_inv)));
	const int v_quad_ind = max(0, min(NUM_QUADS-1, int(0.5f + (v_coord - d_v_quad_start) * d_v_quad_pitch_inv)));
	const int count = counts[v_quad_ind*NUM_QUADS + u_quad_ind];
	if (count <= 0)
		return 0.0f;
	const int* inds = &quads[(v_quad_ind*NUM_QUADS + u_quad_ind)*numTriangles];

	const float min_width = 0.000000001f;
	//const float min_width = 0.00001f;
	//const float eps = 0.00001f;
	const float eps = 0.000001f;
	// const float eps = 0.000000001f;

	float ts[MAX_MESH_INTERSECTIONS];
	int ind = 0;
	for (int ii = 0; ii < count; ii++)
	{
		int i = inds[ii];
		if (AABB[i].w <= u_coord && u_coord <= AABB[i].x && AABB[i].y <= v_coord && v_coord <= AABB[i].z)
		{
			const float3 v1 = triangles[i*3 + 0];
			const float3 v2 = triangles[i*3 + 1];
			const float3 v3 = triangles[i*3 + 2];

			const float3 edge1 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);  // 3
			const float3 edge2 = make_float3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);  // 3

			const float3 pvec = cross3(r, edge2); // 9
			const float det = dot3(edge1, pvec); // 5
			if (fabs(det) < eps)
				continue;

			const float inv_det = 1.0f / det;                                                 // 1
			const float3 tvec = make_float3(p.x - v1.x, p.y - v1.y, p.z - v1.z);              // 3
			const float u = dot3(tvec, pvec) * inv_det; // 6
			if (u < 0.0f || u > 1.0f)
				continue;

			const float3 qvec = cross3(tvec, edge1); // 9
			const float v = dot3(r, qvec) * inv_det; // 6
			if (v < 0.0f || u + v > 1.0f)
				continue;

			const float t = dot3(edge2, qvec) * inv_det; // 6

			// insert x-ray hit distance
			if (ind == 0)
			{
				ts[0] = t;
				ind = 1;
			}
			else if (ind == 1)
			{
				if (ts[0]+min_width < t)
				{
					ts[1] = t;
					ind = 2;
				}
				else if (t+min_width < ts[0])
				{
					ts[1] = ts[0];
					ts[0] = t;
					ind = 2;
				}
			}
			else
			{
				int pos = 0;
				while (pos < ind && ts[pos] < t)
					pos++;
				// ts[pos-1] <= t <= ts[pos]
				if ((pos == ind || t + min_width < ts[pos]) && (pos == 0 || ts[pos-1]+min_width < t))
				{
					// shift elements to the right
					for (int j = ind; j > pos; j--)
						ts[j] = ts[j - 1];

					// insert new element
					ts[pos] = t;
					ind++;
					if (ind >= MAX_MESH_INTERSECTIONS)
						return ray_triangle_intersection_with_AABB_backup(p, r, triangles, numTriangles, v_coord, u_coord, AABB, quads, counts);
				}
			}
		}
	}

	if (ind == 0)
	{
		return 0.0f;
	}
	else if ((ind & 1) != 0)
	{
		return NAN;
	}
	else if (ind == 2)
	{
		return ts[1] - ts[0];
	}
	else
	{
		float accum = ts[1]-ts[0];
		for (int i = 2; i < ind-1; i+=2)
		{
			//if (i+1 < ind)
				accum += ts[i+1]-ts[i];
		}
		return accum;
	}
}

__device__ float u(const int i)
{
    return float(i) * d_T_g.z + d_startVal_g.z;
}

__device__ float v(const int i)
{
    return float(i) * d_T_g.y + d_startVal_g.y;
}

__device__ float z_source(const float phi, const int k)
{
	if (d_geometry == d_CONE_PARALLEL)
	{
		//const float alpha = asin(u(k) / d_sod) + asin(d_tau / d_sod);
		return (phi + asin(u(k) / d_sod) + asin(d_tau / d_sod)) * d_T_g.w + d_startVal_g.w;
	}
	else
	    return phi * d_T_g.w + d_startVal_g.w;
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
            z_source(phi, 0));
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
        const float s = u(iCol) + du;
        const float sqrt_R2_minus_s2 = sqrtf(d_sod * d_sod - s * s);
        return make_float3(-s * sin_phi + sqrt_R2_minus_s2 * cos_phi,
            s * cos_phi + sqrt_R2_minus_s2 * sin_phi,
            z_source(phi, iCol));
    }
    else
        return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float3 setModuleCenter(const float phi)
{
	// FIXME: currently this only works for cone-beam geometries
	const float cos_phi = cos(phi);
	const float sin_phi = sin(phi);

	return make_float3(d_sod * cos_phi + d_tau * sin_phi - cos_phi*d_cos_pitch*d_sdd,
					   d_sod * sin_phi - d_tau * cos_phi - sin_phi*d_cos_pitch*d_sdd,
					   z_source(phi, 0) + d_sin_pitch*d_sdd);
}

__device__ float3 setRowVector(const float phi)
{
	// FIXME: currently this only works for cone-beam geometries
	const float cos_phi = cos(phi);
	const float sin_phi = sin(phi);

	return make_float3(sin_phi * d_sin_tilt + cos_phi * d_sin_pitch * d_cos_tilt,
					  -cos_phi * d_sin_tilt + sin_phi * d_sin_pitch * d_cos_tilt,
					  d_cos_pitch * d_cos_tilt);
}

__device__ float3 setColVector(const float phi)
{
	// FIXME: currently this only works for cone-beam geometries
	const float cos_phi = cos(phi);
	const float sin_phi = sin(phi);

	return make_float3(-sin_phi * d_cos_tilt + cos_phi * d_sin_pitch * d_sin_tilt,
						cos_phi * d_cos_tilt + sin_phi * d_sin_pitch * d_sin_tilt,
						d_cos_pitch * d_sin_tilt);
}

__device__ float3 setTrajectory(const float phi, const int iProj, const int iRow, const int iCol, const float dv, const float du)
{
    const float u_val = u(iCol) + du;
    const float v_val = v(iRow) + dv;

    const float cos_phi = cos(phi);
    const float sin_phi = sin(phi);

	/*
	float u_tilt = u_val;
	float v_tilt = v_val;
	if (d_sin_tilt != 0.0f || d_sin_pitch != 0.0f)
	{
		u_tilt = u_val * d_cos_tilt - v_val * d_sin_tilt;
		v_tilt = u_val * d_sin_tilt + v_val * d_cos_tilt;
	}
	//*/

    if (d_geometry == d_PARALLEL)
    {
        const float3 r = make_float3(-cos_phi, -sin_phi, 0.0f);
		const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
		return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
    }
    else if (d_geometry == d_FAN)
    {
		const float3 r = make_float3(-(cos_phi + u_val * sin_phi), -(sin_phi - u_val * cos_phi), 0.0f);
		const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
		return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
    }
    else if (d_geometry == d_CONE)
    {
        if (d_detectorType == d_CURVED)
        {
			const float3 r = make_float3(-cos(phi - u_val), -sin(phi - u_val), v_val);
			const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
			return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
        }
        else
        {
			if (d_sin_pitch != 0.0f)
			{
				const float3 r = make_float3(-cos_phi*d_cos_pitch + u_val*(-sin_phi*d_cos_tilt + cos_phi*d_sin_pitch*d_sin_tilt) + v_val*(sin_phi*d_sin_tilt + cos_phi*d_sin_pitch*d_cos_tilt),
				-sin_phi*d_cos_pitch + u_val*(cos_phi*d_cos_tilt + sin_phi*d_sin_pitch*d_sin_tilt) + v_val*(-cos_phi*d_sin_tilt + sin_phi*d_sin_pitch*d_cos_tilt),
				d_sin_pitch + u_val*(d_cos_pitch*d_sin_tilt) + v_val*(d_cos_pitch*d_cos_tilt));

				const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
				return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
			}
			else
			{
				const float u_tilt = u_val * d_cos_tilt - v_val * d_sin_tilt;
				const float v_tilt = u_val * d_sin_tilt + v_val * d_cos_tilt;
				const float3 r = make_float3(-(cos_phi + u_tilt * sin_phi), -(sin_phi - u_tilt * cos_phi), v_tilt);
				const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
				return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
			}
        }
    }
    else if (d_geometry == d_CONE_PARALLEL)
    {
		const float3 r = make_float3(-cos_phi, -sin_phi, v_val);
		const float r_mag_inv = rsqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
		return make_float3(r.x * r_mag_inv, r.y * r_mag_inv, r.z * r_mag_inv);
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
		const float p_dot_n = solid->clippingPlanes[i][0] * p.x + solid->clippingPlanes[i][1] * p.y + solid->clippingPlanes[i][2] * p.z;
		const float r_dot_n = solid->clippingPlanes[i][0] * r.x + solid->clippingPlanes[i][1] * r.y + solid->clippingPlanes[i][2] * r.z;
		if (fabs(r_dot_n) < 1.0e-12f)
		{
			if (p_dot_n < solid->clippingPlanes[i][3])
				return false;
		}
		else if (r_dot_n > 0.0f)
		{
			const float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			// restriction: t > temp
			if (ts.y < temp)
				return false;
			if (temp > ts.x)
				ts.x = temp;
		}
		else
		{
			// restriction: t < temp
			const float temp = (solid->clippingPlanes[i][3] - p_dot_n) / r_dot_n;
			if (ts.x > temp)
				return false;
			if (ts.y > temp)
				ts.y = temp;
		}
	}

	return true;
}

__device__ bool intersectionEndPoints_centeredAndNormalized(double3& p, double3& r, float2& ts, geometricSolid* solid)
{
	ts.x = -OUT_OF_BOUNDS;
	ts.y = ts.x;

	// r != (0,0,1)
	const double r_dot_r = r.x * r.x + r.y * r.y + r.z * r.z;
	const double p_dot_r = r.x * p.x + r.y * p.y + r.z * p.z;

	if (solid->type == d_ELLIPSOID)
	{
		double disc = p_dot_r * p_dot_r + r_dot_r * (1.0 - (p.x * p.x + p.y * p.y + p.z * p.z));
		if (disc > 0.0)
		{
			disc = sqrt(disc);
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
		const double r_dot_r_2D = r_dot_r - r.z * r.z; // 2
		const double p_dor_r_2D = p_dot_r - p.z * r.z; // 2
		double disc = p_dor_r_2D * p_dor_r_2D - r_dot_r_2D * (p.x * p.x + p.y * p.y - 1.0); // 7
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-p_dor_r_2D - disc) / r_dot_r_2D; // 2
			const float tmax = (-p_dor_r_2D + disc) / r_dot_r_2D; // 2

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
		const double r_dot_r_2D = r_dot_r - r.x * r.x;
		double disc = (p.z * r.z + p.y * r.y) * (p.z * r.z + p.y * r.y) - r_dot_r_2D * (p.z * p.z + p.y * p.y - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-(p.z * r.z + r.y * p.y) - disc) / r_dot_r_2D;
			const float tmax = (-(p.z * r.z + r.y * p.y) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.x, r.x) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.y == 0.0 && r.z == 0.0 && p.y * p.y + p.z * p.z <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p.x, r.x);
		}
		else
			return false;
	}
	else if (solid->type == d_CYLINDER_Y) // ellipsoidal cross sections parallel to x-y axis
	{
		const double r_dot_r_2D = r_dot_r - r.y * r.y;
		double disc = (p.x * r.x + p.z * r.z) * (p.x * r.x + p.z * r.z) - r_dot_r_2D * (p.x * p.x + p.z * p.z - 1.0);
		if (disc > 0.0)
		{
			disc = sqrt(disc);
			const float tmin = (-(p.x * r.x + r.z * p.z) - disc) / r_dot_r_2D;
			const float tmax = (-(p.x * r.x + r.z * p.z) + disc) / r_dot_r_2D;

			float2 tz;
			if (parametersOfIntersection_1D(tz, p.y, r.y) == true)
			{
				ts.x = max(tmin, tz.x);
				ts.y = min(tmax, tz.y);
			}
			else
				return false;
		}
		else if (r.x == 0.0 && r.z == 0.0 && p.x * p.x + p.z * p.z <= 1.0)
		{
			return parametersOfIntersection_1D(ts, p.y, r.y);
		}
		else
			return false;
	}
	else if (solid->type == d_CONE_Z)
	{
		double a = r.x * r.x + r.y * r.y - r.z * r.z;
		const double b_half = p.x * r.x + p.y * r.y - p.z * r.z;
		const double c = p.x * p.x + p.y * p.y - p.z * p.z;
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

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
		double a = r.z * r.z + r.y * r.y - r.x * r.x;
		const double b_half = p.z * r.z + p.y * r.y - p.x * r.x;
		const double c = p.z * p.z + p.y * p.y - p.x * p.x;
		double disc = b_half * b_half - a * c;

		if (disc > 0.0)
		{
			disc = sqrt(disc);
			float tmin = (-b_half - disc) / a;
			float tmax = (-b_half + disc) / a;
			if (tmin > tmax)
			{
				a = tmin;
				tmin = tmax;
				tmax = a;
			}

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

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
		double a = r.x * r.x + r.z * r.z - r.y * r.y;
		const double b_half = p.x * r.x + p.z * r.z - p.y * r.y;
		const double c = p.x * p.x + p.z * p.z - p.y * p.y;
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

			const double theShift = 0.5 * (solid->clipCone.y + solid->clipCone.x);
			const double theScale = 0.5 * (solid->clipCone.y - solid->clipCone.x);

			float2 tz;
			if (parametersOfIntersection_1D(tz, (p.y - theShift) / theScale, r.y / theScale) == true)
			{
				if (fabs(r.y) > 1.0e-12f)
				{
					bool isInside_0 = false;
					double x_val, y_val, z_val;
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
    double3 q;
	double3 Minv_r;

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
		double3 temp;

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

__device__ float lineIntegral_geometricSolids(float3 p, float3 r, geometricSolid* solids, const int numObjects, float* floatData, int* intData, TEX_DATA xsec)
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
		if (d_NUM_ENERGIES > 0)
		{
			for (int iEnergy = 0; iEnergy < d_NUM_ENERGIES; iEnergy++)
			{
				float muL = 0.0f;
				// loop through intersections
				for (int i = 0; i < 2*count - 1; i++)
				{
					// Consider the interval (allPoints[i], allPoints[i+1])
					// find the material that occupies this space
					const float midPoint = (endPoints[i + 1] + endPoints[i]) * 0.5f;
					for (int ind = count - 1; ind >= 0; ind--)
					{
						const int j = objectIndices[ind];
						//if (objects[j].val != 0.0)
						{
							// Find which object this interval belongs to
							if (intersection_0[j] <= midPoint && midPoint <= intersection_1[j])
							{
								muL += solids[j].val * (endPoints[i + 1] - endPoints[i]) * TEX3D(xsec, iEnergy, solids[j].materialType+1, 0);
								//const float rhoL = solids[j].val * (endPoints[i + 1] - endPoints[i]);
								break;
							}
						}
					}
				}

				retVal += TEX3D(xsec, iEnergy, 0, 0)*expf(-muL);
			}
			retVal = -logf(retVal);
		}
		else
		{
			for (int i = 0; i < 2*count - 1; i++)
			{
				// Consider the interval (allPoints[i], allPoints[i+1])
				const float midPoint = (endPoints[i + 1] + endPoints[i]) * 0.5f;
				for (int ind = count - 1; ind >= 0; ind--)
				{
					const int j = objectIndices[ind];
					//if (objects[j].val != 0.0)
					{
						// Find which object this interval belongs to
						if (intersection_0[j] <= midPoint && midPoint <= intersection_1[j])
						{
							retVal += solids[j].val * (endPoints[i + 1] - endPoints[i]);
							break;
						}
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

__global__ void rayTracingKernel_modular(float* g, const float* phis, geometricSolid* solids, const int numObjects, float* floatData, int* intData, const uint64 ichunk, const int chunkSize, const float* sourcePositions, const float* moduleCenters, const float* rowVectors, const float* colVectors, TEX_DATA xsec)
{
	//const int i = threadIdx.x + blockIdx.x * blockDim.x;
	//const int j = threadIdx.y + blockIdx.y * blockDim.y;
	//const int k = threadIdx.z + blockIdx.z * blockDim.z;
	const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;

	uint64 ind = ichunk * chunkSize + iprocess;
	int k = ind % d_N_g.z;
	ind = (ind - k) / d_N_g.z;
	int j = ind % d_N_g.y;
	int i = (ind - j) / d_N_g.y;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
		return;

	const float* sourcePosition = &sourcePositions[3 * i];
	const float* moduleCenter = &moduleCenters[3 * i];
	const float* v_vec = &rowVectors[3 * i];
	const float* u_vec = &colVectors[3 * i];

	const float v_val = v(j);
	const float u_val = u(k);

	const float3 sourcePos = make_float3(sourcePosition[0], sourcePosition[1], sourcePosition[2]);
	const float3 detPos = make_float3(moduleCenter[0] + u_val * u_vec[0] + v_val * v_vec[0], moduleCenter[1] + u_val * u_vec[1] + v_val * v_vec[1], moduleCenter[2] + u_val * u_vec[2] + v_val * v_vec[2]);

	if (d_oversampling <= 1)
	{
		const float r_mag_inv = rsqrtf((detPos.x - sourcePos.x) * (detPos.x - sourcePos.x) + (detPos.y - sourcePos.y) * (detPos.y - sourcePos.y) + (detPos.z - sourcePos.z) * (detPos.z - sourcePos.z));
		const float3 r = make_float3((detPos.x - sourcePos.x) * r_mag_inv, (detPos.y - sourcePos.y) * r_mag_inv, (detPos.z - sourcePos.z) * r_mag_inv);

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects], xsec);
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float dv = j_os * T_v_os;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du = k_os * T_u_os;

				const float3 detPos_mod = make_float3(detPos.x + du * u_vec[0] + dv * v_vec[0], detPos.y + du * u_vec[1] + dv * v_vec[1], detPos.z + du * u_vec[2] + dv * v_vec[2]);
				const float r_mag_inv = rsqrtf((detPos_mod.x - sourcePos.x) * (detPos_mod.x - sourcePos.x) + (detPos_mod.y - sourcePos.y) * (detPos_mod.y - sourcePos.y) + (detPos_mod.z - sourcePos.z) * (detPos_mod.z - sourcePos.z));
				const float3 r = make_float3((detPos_mod.x - sourcePos.x) * r_mag_inv, (detPos_mod.y - sourcePos.y) * r_mag_inv, (detPos_mod.z - sourcePos.z) * r_mag_inv);

				accum += expf(-lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects], xsec));
			}
		}

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = -log(accum / float(d_oversampling * d_oversampling));
	}
}

__global__ void rayTracingKernel(float* g, const float* phis, geometricSolid* solids, const int numObjects, float* floatData, int* intData, const uint64 ichunk, const int chunkSize, TEX_DATA xsec)
{
    //const int i = threadIdx.x + blockIdx.x * blockDim.x;
    //const int j = threadIdx.y + blockIdx.y * blockDim.y;
    //const int k = threadIdx.z + blockIdx.z * blockDim.z;
	const int iprocess = threadIdx.x + blockIdx.x * blockDim.x;

	uint64 ind = ichunk * chunkSize + iprocess;
	int k = ind % d_N_g.z;
	ind = (ind - k) / d_N_g.z;
	int j = ind % d_N_g.y;
	int i = (ind - j) / d_N_g.y;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
        return;

	const uint64 ind_out = uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k);

	const float phi = phis[i];
	if (d_oversampling <= 1)
	{
		const float dv = 0.0f;
		const float du = 0.0f;

		const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
		const float3 r = setTrajectory(phi, i, j, k, dv, du);

		g[ind_out] = lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects], xsec);
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);
		const int os_radius = (d_oversampling - 1) / 2;
		float accum = 0.0f;

		if (d_geometry == d_CONE && (d_source_height > 0.0f || d_source_width > 0.0f))
		{
			//#############################################################################################################
			const float3 moduleCenter = setModuleCenter(phi);
			const float3 rowVector = setRowVector(phi);
			const float3 colVector = setColVector(phi);
			const float3 sourcePos = setSourcePosition(phi, i, j, k, 0.0f, 0.0f);

			const float source_height_shift = d_source_height / float(d_oversampling + 1);
			const float source_width_shift = d_source_width / float(d_oversampling + 1);

			for (int j_os = -os_radius; j_os <= os_radius; j_os++)
			{
				const float dv = j_os * T_v_os;
				const float v_coord = d_sdd*(v(j) + dv);

				const float source_dz = j_os * source_height_shift;

				for (int k_os = -os_radius; k_os <= os_radius; k_os++)
				{
					const float du = k_os * T_u_os;
					const float u_coord = d_sdd*(u(k) + du);

					const float source_dx = k_os * source_width_shift;

					const float3 sourcePos_shift = make_float3(sourcePos.x + colVector.x*source_dx, sourcePos.y + colVector.y*source_dx, sourcePos.z + source_dz);

					const float3 r_temp = make_float3(moduleCenter.x + v_coord*rowVector.x + u_coord*colVector.x - sourcePos_shift.x, moduleCenter.y + v_coord*rowVector.y + u_coord*colVector.y - sourcePos_shift.y, moduleCenter.z + v_coord*rowVector.z + u_coord*colVector.z - sourcePos_shift.z);
					const float r_mag_inv = rsqrtf(r_temp.x * r_temp.x + r_temp.y * r_temp.y + r_temp.z * r_temp.z);
					const float3 r = make_float3(r_temp.x * r_mag_inv, r_temp.y * r_mag_inv, r_temp.z * r_mag_inv);

					accum += expf(-lineIntegral_geometricSolids(sourcePos_shift, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects], xsec));
				}
			}
			//#############################################################################################################
		}
		else
		{
			for (int j_os = -os_radius; j_os <= os_radius; j_os++)
			{
				const float dv = j_os * T_v_os;

				for (int k_os = -os_radius; k_os <= os_radius; k_os++)
				{
					const float du = k_os * T_u_os;

					const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
					const float3 r = setTrajectory(phi, i, j, k, dv, du);

					accum += expf(-lineIntegral_geometricSolids(sourcePos, r, solids, numObjects, &floatData[iprocess * 4 * numObjects], &intData[iprocess * numObjects], xsec));
				}
			}
		}

		g[ind_out] = -log(accum / float(d_oversampling * d_oversampling));
	}
}

__global__ void rayTracingMeshKernel(float* g, const float* phis, const float* triangles, const int numTriangles)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
        return;

	const float phi = phis[i];
	const float dv = 0.0f;
	const float du = 0.0f;

	const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
	const float3 r = setTrajectory(phi, i, j, k, dv, du);

	if (d_oversampling <= 1)
	{
		const float dv = 0.0f;
		const float du = 0.0f;

		const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
		const float3 r = setTrajectory(phi, i, j, k, dv, du);

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = ray_triangle_intersection(sourcePos, r, triangles, numTriangles);
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float dv = j_os * T_v_os;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du = k_os * T_u_os;

				const float3 sourcePos = setSourcePosition(phi, i, j, k, dv, du);
				const float3 r = setTrajectory(phi, i, j, k, dv, du);

				accum += expf(-ray_triangle_intersection(sourcePos, r, triangles, numTriangles));
			}
		}

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = -log(accum / float(d_oversampling * d_oversampling));
	}
	//g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = ray_triangle_intersection(sourcePos, r, triangles, numTriangles);
}

__global__ void rayTracingMeshKernel_modular(float* g, const float* phis, const float* triangles, const int numTriangles, const float* sourcePositions, const float* moduleCenters, const float* rowVectors, const float* colVectors)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= d_N_g.x || j >= d_N_g.y || k >= d_N_g.z)
		return;

	const float* sourcePosition = &sourcePositions[3 * i];
	const float* moduleCenter = &moduleCenters[3 * i];
	const float* v_vec = &rowVectors[3 * i];
	const float* u_vec = &colVectors[3 * i];

	const float v_val = v(j);
	const float u_val = u(k);

	const float3 sourcePos = make_float3(sourcePosition[0], sourcePosition[1], sourcePosition[2]);
	const float3 detPos = make_float3(moduleCenter[0] + u_val * u_vec[0] + v_val * v_vec[0], moduleCenter[1] + u_val * u_vec[1] + v_val * v_vec[1], moduleCenter[2] + u_val * u_vec[2] + v_val * v_vec[2]);

	//const float r_mag_inv = rsqrtf((detPos.x - sourcePos.x) * (detPos.x - sourcePos.x) + (detPos.y - sourcePos.y) * (detPos.y - sourcePos.y) + (detPos.z - sourcePos.z) * (detPos.z - sourcePos.z));
	//const float3 r = make_float3((detPos.x - sourcePos.x) * r_mag_inv, (detPos.y - sourcePos.y) * r_mag_inv, (detPos.z - sourcePos.z) * r_mag_inv);
	//g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = ray_triangle_intersection(sourcePos, r, triangles, numTriangles);
	if (d_oversampling <= 1)
	{
		const float r_mag_inv = rsqrtf((detPos.x - sourcePos.x) * (detPos.x - sourcePos.x) + (detPos.y - sourcePos.y) * (detPos.y - sourcePos.y) + (detPos.z - sourcePos.z) * (detPos.z - sourcePos.z));
		const float3 r = make_float3((detPos.x - sourcePos.x) * r_mag_inv, (detPos.y - sourcePos.y) * r_mag_inv, (detPos.z - sourcePos.z) * r_mag_inv);

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = ray_triangle_intersection(sourcePos, r, triangles, numTriangles);
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float dv = j_os * T_v_os;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du = k_os * T_u_os;

				const float3 detPos_mod = make_float3(detPos.x + du * u_vec[0] + dv * v_vec[0], detPos.y + du * u_vec[1] + dv * v_vec[1], detPos.z + du * u_vec[2] + dv * v_vec[2]);
				const float r_mag_inv = rsqrtf((detPos_mod.x - sourcePos.x) * (detPos_mod.x - sourcePos.x) + (detPos_mod.y - sourcePos.y) * (detPos_mod.y - sourcePos.y) + (detPos_mod.z - sourcePos.z) * (detPos_mod.z - sourcePos.z));
				const float3 r = make_float3((detPos_mod.x - sourcePos.x) * r_mag_inv, (detPos_mod.y - sourcePos.y) * r_mag_inv, (detPos_mod.z - sourcePos.z) * r_mag_inv);

				accum += expf(-ray_triangle_intersection(sourcePos, r, triangles, numTriangles));
			}
		}

		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = -log(accum / float(d_oversampling * d_oversampling));
	}
}

__device__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_int = (int*)address;  // Treat the float pointer as an int pointer
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        // Use atomicCAS to set the maximum value atomically
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void initializeAABB(const int iphi, const float3 sourcePos, const float3 moduleCenter, const float3 detNormal, const float3 rowVector, const float3 colVector, const float3* triangles, const int numTriangles, float4* AABB, int* quads, int* counts, int* bbox)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numTriangles)
		return;

	if (d_oversampling > 1 && (d_source_height > 0.0f || d_source_width > 0.0f))
	{
		bool initialized = false;
		const float source_height_shift = 0.5f * d_source_height;
		const float source_width_shift = 0.5f * d_source_width;
		for (int ivert = 0; ivert < 3; ivert++)
		{
			//const float3 v = make_float3(triangles[i * 9 + ivert * 3 + 0], triangles[i * 9 + ivert * 3 + 1], triangles[i * 9 + ivert * 3 + 2]);
			const float3 v = triangles[i*3 + ivert];

			for (int dh = -1; dh <= 1; dh++)
			{
				const float source_dz = dh * source_height_shift;
				for (int dw = -1; dw <= 1; dw++)
				{
					const float source_dx = dw * source_width_shift;

					const float3 sourcePos_shift = make_float3(sourcePos.x + colVector.x*source_dx, sourcePos.y + colVector.y*source_dx, sourcePos.z + source_dz);

					const float3 r = make_float3(v.x - sourcePos_shift.x, v.y - sourcePos_shift.y, v.z - sourcePos_shift.z);
					const float t = ((moduleCenter.x - sourcePos_shift.x) * detNormal.x + (moduleCenter.y - sourcePos_shift.y) * detNormal.y + (moduleCenter.z - sourcePos_shift.z) * detNormal.z) / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
					const float u_coord = (sourcePos_shift.x-moduleCenter.x + t*r.x) * colVector.x + (sourcePos_shift.y-moduleCenter.y + t*r.y) * colVector.y + (sourcePos_shift.z-moduleCenter.z + t*r.z) * colVector.z;
					const float v_coord = (sourcePos_shift.x-moduleCenter.x + t*r.x) * rowVector.x + (sourcePos_shift.y-moduleCenter.y + t*r.y) * rowVector.y + (sourcePos_shift.z-moduleCenter.z + t*r.z) * rowVector.z;

					if (initialized)
					{
						AABB[i].w = min(AABB[i].w, u_coord);
						AABB[i].x = max(AABB[i].x, u_coord);
						AABB[i].y = min(AABB[i].y, v_coord);
						AABB[i].z = max(AABB[i].z, v_coord);
					}
					else
					{
						AABB[i].w = u_coord;
						AABB[i].x = u_coord;
						AABB[i].y = v_coord;
						AABB[i].z = v_coord;
						initialized = true;
					}
				}
			}
		}
	}
	else
	{
		for (int ivert = 0; ivert < 3; ivert++)
		{
			//const float3 v = make_float3(triangles[i * 9 + ivert * 3 + 0], triangles[i * 9 + ivert * 3 + 1], triangles[i * 9 + ivert * 3 + 2]);
			const float3 v = triangles[i*3 + ivert];
		
			const float3 r = make_float3(v.x - sourcePos.x, v.y - sourcePos.y, v.z - sourcePos.z);
			const float t = ((moduleCenter.x - sourcePos.x) * detNormal.x + (moduleCenter.y - sourcePos.y) * detNormal.y + (moduleCenter.z - sourcePos.z) * detNormal.z) / (r.x * detNormal.x + r.y * detNormal.y + r.z * detNormal.z);
			const float u_coord = (sourcePos.x-moduleCenter.x + t*r.x) * colVector.x + (sourcePos.y-moduleCenter.y + t*r.y) * colVector.y + (sourcePos.z-moduleCenter.z + t*r.z) * colVector.z;
			const float v_coord = (sourcePos.x-moduleCenter.x + t*r.x) * rowVector.x + (sourcePos.y-moduleCenter.y + t*r.y) * rowVector.y + (sourcePos.z-moduleCenter.z + t*r.z) * rowVector.z;

			if (ivert == 0)
			{
				AABB[i].w = u_coord;
				AABB[i].x = u_coord;
				AABB[i].y = v_coord;
				AABB[i].z = v_coord;
			}
			else
			{
				AABB[i].w = min(AABB[i].w, u_coord);
				AABB[i].x = max(AABB[i].x, u_coord);
				AABB[i].y = min(AABB[i].y, v_coord);
				AABB[i].z = max(AABB[i].z, v_coord);
			}
		}
	}

	for (int iv_quad = 0; iv_quad < NUM_QUADS; iv_quad++)
	{
		const float v_quad = iv_quad*d_v_quad_pitch + d_v_quad_start;
		if (min(AABB[i].z, v_quad+0.5f*d_v_quad_pitch) - max(AABB[i].y, v_quad-0.5f*d_v_quad_pitch) > 0.0f)
		{
			for (int iu_quad = 0; iu_quad < NUM_QUADS; iu_quad++)
			{
				const float u_quad = iu_quad*d_u_quad_pitch + d_u_quad_start;
				if (min(AABB[i].x, u_quad+0.5f*d_u_quad_pitch) - max(AABB[i].w, u_quad-0.5f*d_u_quad_pitch) > 0.0f)
				{
					const int quad_ind = iv_quad*NUM_QUADS + iu_quad;
					quads[quad_ind*numTriangles + atomicAdd(&counts[quad_ind], 1)] = i;
				}
			}
		}
	}

	/*
	if (AABB[i].x >= 0.0 && AABB[i].z >= 0.0)
	{
		quads[0*numTriangles + atomicAdd(&counts[0], 1)] = i;
	}
	if (AABB[i].w < 0.0 && AABB[i].z >= 0.0)
	{
		quads[1*numTriangles + atomicAdd(&counts[1], 1)] = i;
	}
	if (AABB[i].w < 0.0 && AABB[i].y < 0.0)
	{
		quads[2*numTriangles + atomicAdd(&counts[2], 1)] = i;
	}
	if (AABB[i].x >= 0.0 && AABB[i].y < 0.0)
	{
		quads[3*numTriangles + atomicAdd(&counts[3], 1)] = i;
	}
	//*/
	//*
	atomicMin(&(bbox[0]), int(floor((AABB[i].w-d_startVal_g.z)/d_T_g.z)));
	atomicMax(&(bbox[1]), int(ceil((AABB[i].x-d_startVal_g.z)/d_T_g.z)));
	atomicMin(&(bbox[2]), int(floor((AABB[i].y-d_startVal_g.y)/d_T_g.y)));
	atomicMax(&(bbox[3]), int(ceil((AABB[i].z-d_startVal_g.y)/d_T_g.y)));
	//*/
}

__global__ void rayTracingMeshKernelWithAABB(float* g, const int i, const float3 sourcePos, const float3 moduleCenter, const float3 rowVector, const float3 colVector, const float3* triangles, const int numTriangles, const float val, const float4* AABB, const int* quads, const int* counts, const int* bbox, TEX_DATA xsec)
{
	const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

	/*
	if (j == 0 && k == 0)
	{
		for (int iv_quad = 0; iv_quad < NUM_QUADS; iv_quad++)
		{
			for (int iu_quad = 0; iu_quad < NUM_QUADS; iu_quad++)
			{
				printf("counts[%d][%d] = %d\n", iv_quad, iu_quad, counts[iv_quad*NUM_QUADS + iu_quad]);
			}
		}
	}
	//*/

	//const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= d_N_g.y || k >= d_N_g.z)
        return;
	//*
	if (k < bbox[0] || bbox[1] < k || j < bbox[2] || bbox[3] < j)
	{
		g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = 0.0f;
		return;
	}
	//*/

	if (d_oversampling <= 1)
	{
		const float v_coord = v(j);
		const float u_coord = u(k);
		const float3 r_temp = make_float3(moduleCenter.x + v_coord*rowVector.x + u_coord*colVector.x - sourcePos.x, moduleCenter.y + v_coord*rowVector.y + u_coord*colVector.y - sourcePos.y, moduleCenter.z + v_coord*rowVector.z + u_coord*colVector.z - sourcePos.z);
		const float r_mag_inv = rsqrtf(r_temp.x * r_temp.x + r_temp.y * r_temp.y + r_temp.z * r_temp.z);
		const float3 r = make_float3(r_temp.x * r_mag_inv, r_temp.y * r_mag_inv, r_temp.z * r_mag_inv);

		const float lineLength = ray_triangle_intersection_with_AABB(sourcePos, r, triangles, numTriangles, v_coord, u_coord, AABB, quads, counts);
		if (isnan(lineLength))
		{
			g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = NAN;
		}
		else
		{
			if (d_NUM_ENERGIES > 0 && lineLength > 0.0f)
			{
				float retVal = 0.0f;
				for (int iEnergy = 0; iEnergy < d_NUM_ENERGIES; iEnergy++)
				{
					float muL = val * lineLength * TEX3D(xsec, iEnergy, 1, 0);
					retVal += TEX3D(xsec, iEnergy, 0, 0)*expf(-muL);
				}
				retVal = -logf(retVal);
				g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = retVal;
			}
			else
			{
				g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = val*lineLength;
			}
		}
	}
	else
	{
		const float T_v_os = d_T_g.y / float(d_oversampling + 1);
		const float T_u_os = d_T_g.z / float(d_oversampling + 1);

		const float source_height_shift = d_source_height / float(d_oversampling + 1);
		const float source_width_shift = d_source_width / float(d_oversampling + 1);

		const int os_radius = (d_oversampling - 1) / 2;

		float accum = 0.0f;
		int count = 0;
		for (int j_os = -os_radius; j_os <= os_radius; j_os++)
		{
			const float dv = j_os * T_v_os;
			const float v_coord = v(j) + dv;

			const float source_dz = j_os * source_height_shift;

			for (int k_os = -os_radius; k_os <= os_radius; k_os++)
			{
				const float du = k_os * T_u_os;
				const float u_coord = u(k) + du;

				const float source_dx = k_os * source_width_shift;

				const float3 sourcePos_shift = make_float3(sourcePos.x + colVector.x*source_dx, sourcePos.y + colVector.y*source_dx, sourcePos.z + source_dz);

				const float3 r_temp = make_float3(moduleCenter.x + v_coord*rowVector.x + u_coord*colVector.x - sourcePos_shift.x, moduleCenter.y + v_coord*rowVector.y + u_coord*colVector.y - sourcePos_shift.y, moduleCenter.z + v_coord*rowVector.z + u_coord*colVector.z - sourcePos_shift.z);
				const float r_mag_inv = rsqrtf(r_temp.x * r_temp.x + r_temp.y * r_temp.y + r_temp.z * r_temp.z);
				const float3 r = make_float3(r_temp.x * r_mag_inv, r_temp.y * r_mag_inv, r_temp.z * r_mag_inv);

				const float lineLength = ray_triangle_intersection_with_AABB(sourcePos_shift, r, triangles, numTriangles, v_coord, u_coord, AABB, quads, counts);
				if (!isnan(lineLength))
				{
					float retVal = 0.0f;
					if (d_NUM_ENERGIES > 0 && lineLength > 0.0f)
					{
						for (int iEnergy = 0; iEnergy < d_NUM_ENERGIES; iEnergy++)
						{
							float muL = val * lineLength * TEX3D(xsec, iEnergy, 1, 0);
							retVal += TEX3D(xsec, iEnergy, 0, 0)*expf(-muL);
						}
					}
					else
						retVal = expf(-val*lineLength);

					accum += retVal;
					count += 1;
				}
			}
		}

		if (count > 0)
			g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = -logf(accum / float(count));
		else
			g[uint64(i) * uint64(d_N_g.z * d_N_g.y) + uint64(j * d_N_g.z + k)] = NAN;
	}
}


void setConstantMemoryGeometryParameters(parameters* params, int oversampling, bool doNormalize)
{
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, doNormalize);

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
	cudaMemcpyToSymbol(d_geometry, &geometry, sizeof(int));
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
	float cos_tilt = 1.0;
	float sin_tilt = 0.0;
	float cos_pitch = 1.0;
	float sin_pitch = 0.0;
	if (params->geometry == parameters::CONE)
	{
		cos_tilt = cos(params->tiltAngle * PI / 180.0);
		sin_tilt = sin(params->tiltAngle * PI / 180.0);

		cos_pitch = cos(params->pitchAngle * PI / 180.0);
		sin_pitch = sin(params->pitchAngle * PI / 180.0);
	}
    cudaMemcpyToSymbol(d_sod, &sod, sizeof(float));
    cudaMemcpyToSymbol(d_sdd, &sdd, sizeof(float));
    cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_cos_tilt, &cos_tilt, sizeof(float));
	cudaMemcpyToSymbol(d_sin_tilt, &sin_tilt, sizeof(float));
	cudaMemcpyToSymbol(d_cos_pitch, &cos_pitch, sizeof(float));
	cudaMemcpyToSymbol(d_sin_pitch, &sin_pitch, sizeof(float));
	cudaMemcpyToSymbol(d_N_g, &N_g, sizeof(int4));
    cudaMemcpyToSymbol(d_T_g, &T_g, sizeof(float4));
    cudaMemcpyToSymbol(d_startVal_g, &startVal_g, sizeof(float4));

	if (oversampling > 1)
	{
		cudaMemcpyToSymbol(d_source_height, &(params->source_size[0]), sizeof(float));
		cudaMemcpyToSymbol(d_source_width, &(params->source_size[1]), sizeof(float));
	}
	else
	{
		float zero = 0.0;
		cudaMemcpyToSymbol(d_source_height, &zero, sizeof(float));
		cudaMemcpyToSymbol(d_source_width, &zero, sizeof(float));
	}

	//NUM_QUADS
	float u_span = N_g.z*T_g.z;
	float u_quad_pitch = u_span / float(NUM_QUADS);
	float u_quad_pitch_inv = float(NUM_QUADS) / u_span;
	float u_quad_start = startVal_g.z - 0.5*(T_g.z - u_span / float(NUM_QUADS));

	float v_span = N_g.y*T_g.y;
	float v_quad_pitch = v_span / float(NUM_QUADS);
	float v_quad_pitch_inv = float(NUM_QUADS) / v_span;
	float v_quad_start = startVal_g.y - 0.5*(T_g.y - v_span / float(NUM_QUADS));

	cudaMemcpyToSymbol(d_u_quad_pitch, &u_quad_pitch, sizeof(float));
	cudaMemcpyToSymbol(d_u_quad_pitch_inv, &u_quad_pitch_inv, sizeof(float));
	cudaMemcpyToSymbol(d_u_quad_start, &u_quad_start, sizeof(float));
	cudaMemcpyToSymbol(d_v_quad_pitch, &v_quad_pitch, sizeof(float));
	cudaMemcpyToSymbol(d_v_quad_pitch_inv, &v_quad_pitch_inv, sizeof(float));
	cudaMemcpyToSymbol(d_v_quad_start, &v_quad_start, sizeof(float));
}

bool rayTrace_gpu(float* g, parameters* params, phantom* aPhantom, float* spectralResponse, float* energies, int N_energies, bool data_on_cpu, int oversampling)
{
    if (g == NULL || params == NULL || params->geometryDefined() == false)
        return false;
	oversampling = max(1, min(oversampling, 11));
	if (oversampling % 2 == 0)
		oversampling += 1;
	oversampling = max(1, min(oversampling, 11));

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

	setConstantMemoryGeometryParameters(params, oversampling);

	float cm_sq_to_mm_sq = 100.0;

	//float* dev_xsec = 0;
	TEX_DATA xsec_txt = {};
    TEX_ARRAY xsec_array = NULL;
	if (spectralResponse != NULL && energies != NULL && N_energies > 0)
	{
		normalizeSpectrum(spectralResponse, energies, N_energies);

		//printf("doing a polychromatic simulation!\n");
		float* xsec = new float[N_energies*(int(aPhantom->materialTypes.size())+1)];
		for (int i = 0; i < N_energies; i++)
			xsec[i] = spectralResponse[i];
		for (int i = 0; i < int(aPhantom->materialTypes.size()); i++)
		{
			for (int j = 0; j < N_energies; j++)
				xsec[N_energies*(i+1) + j] = sigmaCompound(aPhantom->materialTypes[i].c_str(), energies[j]) * cm_sq_to_mm_sq;
		}
		//for (int j = 0; j < N_energies; j++)
		//	printf("%f %f %f\n", xsec[j], xsec[1*N_energies+j], xsec[2*N_energies+j]);

		xsec_array = loadTexture_from_cpu(xsec_txt, xsec, make_int3(1, int(aPhantom->materialTypes.size())+1, N_energies), false, false);

		delete [] xsec;
	}
	else
	{
		N_energies = 0;
		//printf("doing a monochromatic simulation!\n");
	}
	int num_materialTypes = aPhantom->materialTypes.size();
	cudaMemcpyToSymbol(d_NUM_MATERIAL_TYPES, &num_materialTypes, sizeof(int));
	cudaMemcpyToSymbol(d_NUM_ENERGIES, &N_energies, sizeof(int));

    float* dev_g = 0;
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections[%d][%d][%d]) failed!\n", params->numAngles, params->numRows, params->numCols);
        }
    }
    else
        dev_g = g;

    float* dev_phis = copyAngleArrayToGPU(params);

	float* dev_sourcePositions = 0;
	float* dev_moduleCenters = 0;
	float* dev_rowVectors = 0;
	float* dev_colVectors = 0;
	if (params->geometry == parameters::MODULAR)
	{
		if (cudaSuccess != cudaMalloc((void**)&dev_sourcePositions, 3 * params->numAngles * sizeof(float)))
			fprintf(stderr, "cudaMalloc failed!\n");
		if (cudaMemcpy(dev_sourcePositions, params->sourcePositions, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
			fprintf(stderr, "cudaMemcpy(sourcePositions) failed!\n");

		if (cudaSuccess != cudaMalloc((void**)&dev_moduleCenters, 3 * params->numAngles * sizeof(float)))
			fprintf(stderr, "cudaMalloc failed!\n");
		if (cudaMemcpy(dev_moduleCenters, params->moduleCenters, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
			fprintf(stderr, "cudaMemcpy(moduleCenters) failed!\n");

		if (cudaSuccess != cudaMalloc((void**)&dev_rowVectors, 3 * params->numAngles * sizeof(float)))
			fprintf(stderr, "cudaMalloc failed!\n");
		if (cudaMemcpy(dev_rowVectors, params->rowVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
			fprintf(stderr, "cudaMemcpy(rowVectors) failed!\n");

		if (cudaSuccess != cudaMalloc((void**)&dev_colVectors, 3 * params->numAngles * sizeof(float)))
			fprintf(stderr, "cudaMalloc failed!\n");
		if (cudaMemcpy(dev_colVectors, params->colVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
			fprintf(stderr, "cudaMemcpy(colVectors) failed!\n");
	}

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
		solids[i].materialType = aPhantom->getMaterialType(i);
		//printf("object %d is material type %d and density %f\n", i, solids[i].materialType, solids[i].val);
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

	int num_gpu_cores = max(1024, getSPcores(params->whichGPU));
	if (params->projectionData_numberOfElements() < uint64(num_gpu_cores))
		num_gpu_cores = int(params->projectionData_numberOfElements());
	uint64 numChunks = uint64(ceil(double(params->projectionData_numberOfElements()) / double(num_gpu_cores)));
	//printf("number of cores = %d, number of chunks = %d\n", num_gpu_cores, int(numChunks));

	int blockSize = 8;
	int numBlocks = int(ceil(double(num_gpu_cores) / double(blockSize)));
	int numDataCopies = numBlocks * blockSize;

	float* dev_floatData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_floatData, numDataCopies * 4 * numObjects * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}
	int* dev_intData = 0;
	if ((cudaStatus = cudaMalloc((void**)&dev_intData, numDataCopies * numObjects * sizeof(int))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(phantom data) failed!\n");
	}

	for (uint64 ichunk = 0; ichunk < numChunks; ichunk++)
	{
		if (params->geometry == parameters::MODULAR)
		{
			rayTracingKernel_modular <<< numBlocks, blockSize >>> (dev_g, dev_phis, dev_solids, numObjects, dev_floatData, dev_intData, ichunk, num_gpu_cores, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, xsec_txt);
		}
		else
		{
			rayTracingKernel <<< numBlocks, blockSize >>> (dev_g, dev_phis, dev_solids, numObjects, dev_floatData, dev_intData, ichunk, num_gpu_cores, xsec_txt);
		}
		//for (int j = 0; j < N_g.y; j++)
		//	rayTracingKernel <<< numBlocks, blockSize >>> (dev_g, dev_phis, dev_solids, numObjects, dev_floatData, dev_intData, i, j);
		//float* g, const float* phis, geometricSolid* solids, const int numObjects, float* allIntersections
		//cudaMemset(dev_floatData, 0, numDataCopies * 4 * numObjects * sizeof(float));
		//cudaMemset(dev_intData, 0, numDataCopies * numObjects * sizeof(int));
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
	if (dev_sourcePositions != 0)
		cudaFree(dev_sourcePositions);
	if (dev_moduleCenters != 0)
		cudaFree(dev_moduleCenters);
	if (dev_rowVectors != 0)
		cudaFree(dev_rowVectors);
	if (dev_colVectors != 0)
		cudaFree(dev_colVectors);
    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
    }

	//cudaFree(dev_xsec);
	freeTexture(xsec_array, xsec_txt);
	//dev_xsec = 0;
	xsec_txt = {};
	xsec_array = NULL;

    return true;
}

bool voxelizeMesh_gpu(float* f, parameters* params, phantom* aPhantom, float val, bool data_on_cpu, int oversampling)
{
	meshObject* mesh = aPhantom->meshes[aPhantom->meshes.size()-1];
	float* triangles = mesh->triangles;
	int N_triangles = mesh->numTriangles;
	if (triangles == NULL || N_triangles <= 0)
		return false;

	//////////////////////////////////////////////////////////////////////////////////
	if (f == NULL || params == NULL || params->volumeDefined() == false)
        return false;
	oversampling = max(1, min(oversampling, 11));
	if (oversampling % 2 == 0)
		oversampling += 1;
	oversampling = max(1, min(oversampling, 11));

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;	

	float* dev_f = 0;
	int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume[%d][%d][%d]) failed!\n", params->numZ, params->numY, params->numX);
			return false;
        }
    }
    else
        dev_f = f;

	float AABB_all[6];
	AABB_all[0] = triangles[0];
	AABB_all[1] = triangles[0];
	AABB_all[2] = triangles[1];
	AABB_all[3] = triangles[1];
	AABB_all[4] = triangles[2];
	AABB_all[5] = triangles[2];
	// triangles: N_triangles * 3 * 3
	//*
	float3* triangles_in_float3 = (float3*) malloc(sizeof(float3)*3*N_triangles);
	for (int i = 0; i < 3*N_triangles; i++)
	{
		AABB_all[0] = min(AABB_all[0], triangles[i*3 + 0]);
		AABB_all[1] = max(AABB_all[1], triangles[i*3 + 0]);
		AABB_all[2] = min(AABB_all[2], triangles[i*3 + 1]);
		AABB_all[3] = max(AABB_all[3], triangles[i*3 + 1]);
		AABB_all[4] = min(AABB_all[4], triangles[i*3 + 2]);
		AABB_all[5] = max(AABB_all[5], triangles[i*3 + 2]);
		triangles_in_float3[i] = make_float3(triangles[i*3 + 0], triangles[i*3 + 1], triangles[i*3 + 2]);
	}
	float3* dev_triangles = copy1Dfloat3ToGPU(triangles_in_float3, 3*N_triangles, params->whichGPU);
	free(triangles_in_float3);

	float4 lower_corner;
	lower_corner.x = AABB_all[0];
	lower_corner.y = AABB_all[2];
	lower_corner.z = AABB_all[4];

	float4 upper_corner;
	upper_corner.x = AABB_all[1];
	upper_corner.y = AABB_all[3];
	upper_corner.z = AABB_all[5];

	/*
	float dims[3];
	dims[0] = AABB_all[1]-AABB_all[0];
	dims[1] = AABB_all[3]-AABB_all[2];
	dims[2] = AABB_all[5]-AABB_all[4];

	float3 r;
	if (dims[0] < min(dims[1], dims[2]))
		r = make_float3(1.0f, 0.0f, 0.0f);
	else if (dims[1] < min(dims[0], dims[2]))
		r = make_float3(0.0f, 1.0f, 0.0f);
	else
		r = make_float3(0.0f, 0.0f, 1.0f);
	//*/

	//printf("r = %f, %f, %f\n", r.x, r.y, r.z);

	float* upper_z = new float[N_triangles];
	float4* AABB = (float4*) malloc(sizeof(float4)*N_triangles);
	for (int i = 0; i < N_triangles; i++)
	{
		int tri_offset = 9*i;
		float* aTriangle = &triangles[tri_offset];
		float3 v1 = make_float3(aTriangle[0], aTriangle[1], aTriangle[2]);
		float3 v2 = make_float3(aTriangle[3], aTriangle[4], aTriangle[5]);
		float3 v3 = make_float3(aTriangle[6], aTriangle[7], aTriangle[8]);
		AABB[i].w = min(min(v1.x, v2.x), v3.x);
		AABB[i].x = max(max(v1.x, v2.x), v3.x);
		AABB[i].y = min(min(v1.y, v2.y), v3.y);
		AABB[i].z = max(max(v1.y, v2.y), v3.y);
	    upper_z[i] = max(max(v1.z, v2.z), v3.z);
	}
	float4* dev_AABB = copy1Dfloat4ToGPU(AABB, N_triangles, params->whichGPU);
	float* dev_upper_z = copy1DdataToGPU(upper_z, N_triangles, params->whichGPU);
	free(AABB);
	delete [] upper_z;

	//analyticRayTracing geometryTools(params);
	dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

	voxelizeKernel <<< dimGrid, dimBlock >>> (dev_f, N_f, T_f, startVal_f, dev_triangles, N_triangles, val, lower_corner, upper_corner, dev_AABB, dev_upper_z, oversampling);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (data_on_cpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
	if (dev_triangles != 0)
		cudaFree(dev_triangles);
	if (dev_AABB != 0)
		cudaFree(dev_AABB);
	if (dev_upper_z != 0)
		cudaFree(dev_upper_z);
    if (data_on_cpu)
    {
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

bool rayTraceMesh_gpu(float* g, parameters* params, phantom* aPhantom,  float* spectralResponse, float* energies, int N_energies, bool data_on_cpu, int oversampling, int which)
{
	meshObject* mesh = nullptr;
	if (which < 0 || which >= aPhantom->meshes.size()-1)
		mesh = aPhantom->meshes[aPhantom->meshes.size()-1];
	else
		mesh = aPhantom->meshes[which];
	float* triangles = mesh->triangles;
	int N_triangles = mesh->numTriangles;
	if (triangles == NULL || N_triangles <= 0)
		return false;

	//////////////////////////////////////////////////////////////////////////////////
	if (g == NULL || params == NULL || params->geometryDefined() == false)
        return false;
	oversampling = max(1, min(oversampling, 11));
	if (oversampling % 2 == 0)
		oversampling += 1;
	oversampling = max(1, min(oversampling, 11));

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

	setConstantMemoryGeometryParameters(params, oversampling, false);

	float cm_sq_to_mm_sq = 100.0;
	TEX_DATA xsec_txt = {};
    TEX_ARRAY xsec_array = NULL;
	if (spectralResponse != NULL && energies != NULL && N_energies > 0)
	{
		normalizeSpectrum(spectralResponse, energies, N_energies);

		float* xsec = new float[N_energies*(int(aPhantom->materialTypes.size())+1)];
		for (int i = 0; i < N_energies; i++)
			xsec[i] = spectralResponse[i];
		for (int i = 0; i < int(aPhantom->materialTypes.size()); i++)
		{
			for (int j = 0; j < N_energies; j++)
				xsec[N_energies*(i+1) + j] = sigmaCompound(aPhantom->materialTypes[i].c_str(), energies[j]) * cm_sq_to_mm_sq;
		}

		xsec_array = loadTexture_from_cpu(xsec_txt, xsec, make_int3(1, int(aPhantom->materialTypes.size())+1, N_energies), false, false);

		delete [] xsec;
	}
	else
	{
		N_energies = 0;
		//printf("doing a monochromatic simulation!\n");
	}
	int num_materialTypes = aPhantom->materialTypes.size();
	cudaMemcpyToSymbol(d_NUM_MATERIAL_TYPES, &num_materialTypes, sizeof(int));
	cudaMemcpyToSymbol(d_NUM_ENERGIES, &N_energies, sizeof(int));
	/////////////////////////////////////////////////////////////////////////////////

    float* dev_g = 0;
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections[%d][%d][%d]) failed!\n", params->numAngles, params->numRows, params->numCols);
        }
    }
    else
        dev_g = g;

	// triangles: N_triangles * 3 * 3
	//*
	float3* triangles_in_float3 = (float3*) malloc(sizeof(float3)*3*N_triangles);
	for (int i = 0; i < 3*N_triangles; i++)
		triangles_in_float3[i] = make_float3(triangles[i*3 + 0], triangles[i*3 + 1], triangles[i*3 + 2]);
	float3* dev_triangles = copy1Dfloat3ToGPU(triangles_in_float3, 3*N_triangles, params->whichGPU);
	free(triangles_in_float3);
	//*/
	//float* dev_triangles = copy1DdataToGPU(triangles, N_triangles*9, params->whichGPU);

	float4* dev_AABB = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_AABB, N_triangles * sizeof(float4)))
		fprintf(stderr, "cudaMalloc failed!\n");
	
	int* dev_quads = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_quads, NUM_QUADS*NUM_QUADS*N_triangles * sizeof(int)))
		fprintf(stderr, "cudaMalloc failed!\n");

	int* dev_counts = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_counts, NUM_QUADS*NUM_QUADS * sizeof(int)))
		fprintf(stderr, "cudaMalloc failed!\n");

	int bbox[4] = {params->numCols-1, 0, params->numRows-1,0};
	//int bbox[4] = {0, params->numCols-1, 0, params->numRows-1}; // ignores box
	int* dev_bbox = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_bbox, 4 * sizeof(int)))
		fprintf(stderr, "cudaMalloc failed!\n");

	analyticRayTracing geometryTools(params);
	dim3 dimBlock(8, 8);
    dim3 dimGrid(int(ceil(double(params->numRows) / double(dimBlock.x))), int(ceil(double(params->numCols) / double(dimBlock.y))));
	for (int i = 0; i < params->numAngles; i++)
	{
		double temp[3];
		geometryTools.setSourcePosition(i, 0, 0, temp);
		float3 sourcePos = make_float3(temp[0], temp[1], temp[2]);
		geometryTools.setModuleCenter(i, temp);
		float3 moduleCenter = make_float3(temp[0], temp[1], temp[2]);
		geometryTools.setRowVector(i, temp);
		float3 rowVec = make_float3(temp[0], temp[1], temp[2]);
		geometryTools.setColVector(i, temp);
		float3 colVec = make_float3(temp[0], temp[1], temp[2]);
		geometryTools.setDetectorNormal(i, temp);
		float3 detNormal = make_float3(temp[0], temp[1], temp[2]);

		cudaMemset(dev_counts, 0, NUM_QUADS*NUM_QUADS*sizeof(int));
		cudaMemcpy(dev_bbox, bbox, 4*sizeof(int), cudaMemcpyHostToDevice);
		initializeAABB <<< int(ceil(float(N_triangles)/256.0)), 256 >>> (i, sourcePos, moduleCenter, detNormal, rowVec, colVec, dev_triangles, N_triangles, dev_AABB, dev_quads, dev_counts, dev_bbox);
		rayTracingMeshKernelWithAABB <<< dimGrid, dimBlock >>> (dev_g, i, sourcePos, moduleCenter, rowVec, colVec, dev_triangles, N_triangles, mesh->val, dev_AABB, dev_quads, dev_counts, dev_bbox, xsec_txt);
	}
	replaceNAN(dev_g, make_int3(params->numAngles, params->numRows, params->numCols), params->whichGPU);

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
	if (dev_triangles != 0)
		cudaFree(dev_triangles);
	if (dev_bbox != 0)
		cudaFree(dev_bbox);
	if (dev_counts != 0)
		cudaFree(dev_counts);
	if (dev_quads != 0)
		cudaFree(dev_quads);
	if (dev_AABB != 0)
		cudaFree(dev_AABB);
    if (data_on_cpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
    }

	freeTexture(xsec_array, xsec_txt);
	xsec_txt = {};
	xsec_array = NULL;

    return true;
}
