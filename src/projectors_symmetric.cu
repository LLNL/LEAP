////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for cylindrically symmetric projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors_symmetric.cuh"
#include "cuda_utils.h"
//using namespace std;

__global__ void AbelConeInverseKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	
	const float T_u_inv = 1.0f / T_g.z;
	const float T_v_inv = 1.0f / T_g.y;
	const float u_0 = startVals_g.z;
	const float v_0 = startVals_g.y;

	const float z_val = T_f.z * float(k) + startVals_f.z;
	const float r_val = T_f.y * float(j) + startVals_f.y;

	const int N_phi = N_g.z + ((N_g.z + 1) % 2);
	const float T_phi = 2.0f * 3.1415926535897932385f / float(N_phi);
	const float R_sq = R * R;

	float curVal = 0.0f;
	for (int i = 0; i < N_phi; i++)
	{
		const float phi = float(i)*T_phi - 0.5f * 3.1415926535897932385f;
		const float cos_phi = cos(phi);
		const float sin_phi = sin(phi);

		const float v_denom_inv = 1.0f / (R + r_val * sin_phi * cos_beta - z_val * sin_beta);
		float s_val = (tau - r_val * cos_phi) * v_denom_inv;
		if (r_val*s_val < 0.0f)
			s_val *= -1.0f;
		const float s_arg = (s_val - u_0) * T_u_inv;
		const float v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) * v_denom_inv - v_0) * T_v_inv;
		const float theWeight = R_sq * v_denom_inv * v_denom_inv;

		//if (r_val*s_val >= 0.0f)
			curVal += theWeight * tex3D<float>(g, s_arg + 0.5f, v_arg + 0.5f, 0.5f);
		//else
		//	curVal += theWeight * tex3D<float>(g, s_arg + 0.5f, v_arg + 0.5f, 0.5f);
	}

	int ind;
	if (volumeDimensionOrder == 0)
		ind = j * N_f.z + k;
	else
		ind = k * N_f.y + j;
	const float undoFBPscaling = T_f.x * T_f.y * T_f.z / (R * T_g.y * R * T_g.z);
	f[ind] = curVal * undoFBPscaling / (2.0f * float(N_phi));
}

__global__ void AbelParallelInverseKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, float tau, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}

	const float T_u_inv = 1.0f / T_g.z;
	const float T_v_inv = 1.0f / T_g.y;
	const float u_0 = startVals_g.z;
	const float v_0 = startVals_g.y;

	const float z_val = T_f.z * float(k) + startVals_f.z;
	const float r_val = T_f.y * float(j) + startVals_f.y;

	const int N_phi = N_g.z + ((N_g.z + 1) % 2);
	const float T_phi = 2.0f * 3.1415926535897932385f / float(N_phi);

	float curVal = 0.0f;
	for (int i = 0; i < N_phi; i++)
	{
		const float phi = float(i) * T_phi - 0.5f * 3.1415926535897932385f;
		const float cos_phi = cos(phi);
		const float sin_phi = sin(phi);

		float s_val = r_val * cos_phi;
		if (r_val * s_val < 0.0f)
			s_val *= -1.0f;

		const float s_arg = (s_val - u_0) * T_u_inv;
		const float v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) - v_0) * T_v_inv;

		//if (r_val * s_val >= 0.0f)
			curVal += tex3D<float>(g, s_arg + 0.5f, v_arg + 0.5f, 0.5f);
	}

	int ind;
	if (volumeDimensionOrder == 0)
		ind = j * N_f.z + k;
	else
		ind = k * N_f.y + j;
	const float undoFBPscaling = T_f.x * T_f.y / (T_g.z);
	f[ind] = curVal * undoFBPscaling / (2.0f * float(N_phi));
}

__global__ void AbelConeBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float tan_beta = sin_beta / cos_beta;
	const float sec_beta = 1.0f / cos_beta;

	const float r_unbounded = T_f.y * float(j) + startVals_f.y;
	const float r = fabs(r_unbounded);
	const float r_inner = r - 0.5f * T_f.y;
	const float r_outer = r + 0.5f * T_f.y;

	const float disc_shift_inner = (r_inner * r_inner - tau * tau) * sec_beta * sec_beta; // r_inner^2
	const float disc_shift_outer = (r_outer * r_outer - tau * tau) * sec_beta * sec_beta; // r_outer^2

	const float z = T_f.z * float(k) + startVals_f.z;
	const float Tv_inv = 1.0f / T_g.y;

	const float Z = (R + z * sin_beta) * sec_beta; // nominal value: R
	const float z_slope = (z + R * sin_beta) * sec_beta; // nominal value: 0

	const float ind_split = -startVals_g.z / T_g.z;
	int iu_max_left, iu_min_right;
	if (fabs(ind_split - floor(0.5f + ind_split)) < 1.0e-4)
	{
		iu_max_left = int(floor(0.5f+ind_split));
		iu_min_right = iu_max_left;
	}
	else
	{
		iu_max_left = int(ind_split);
		iu_min_right = int(ceil(ind_split));
	}

	int iu_min;
	int iu_max;
	if (r_unbounded < 0.0f)
	{
		// left half
		iu_min = 0;
		//iu_max = int(-startVals_g.z / T_g.z);
		iu_max = iu_max_left;
	}
	else
	{
		// right half
		//iu_min = int(ceil(-startVals_g.z / T_g.z));
		iu_min = iu_min_right;
		iu_max = N_g.z - 1;
	}

	float curVal = 0.0f;
	for (int iu = iu_min; iu <= iu_max; iu++)
	{
		const float u = fabs(T_g.z * float(iu) + startVals_g.z);
		float disc_outer = u * u * (r_outer * r_outer - Z * Z) + 2.0f * Z * sec_beta * tau * u + disc_shift_outer; // u^2*(r^2 - R^2) + r^2
		if (disc_outer > 0.0f)
		{
			const float b_ti = Z * sec_beta + tau * u;
			const float a_ti_inv = 1.0f / (u * u + sec_beta * sec_beta);
			float disc_inner = u * u * (r_inner * r_inner - Z * Z) + 2.0f * Z * sec_beta * tau * u + disc_shift_inner; // disc_outer > disc_inner
			if (disc_inner > 0.0f)
			{
				disc_inner = sqrt(disc_inner);
				disc_outer = sqrt(disc_outer);
				const float t_1st_low = (b_ti - disc_outer) * a_ti_inv;
				const float t_1st_high = (b_ti - disc_inner) * a_ti_inv;
				const float v_1st_arg = 2.0f * z_slope / (t_1st_low + t_1st_high) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_1st_arg * v_1st_arg) * (t_1st_high - t_1st_low) * tex3D<float>(g, float(iu) + 0.5f, (v_1st_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);

				const float t_2nd_low = (b_ti + disc_inner) * a_ti_inv;
				const float t_2nd_high = (b_ti + disc_outer) * a_ti_inv;
				const float v_2nd_arg = 2.0f * z_slope / (t_2nd_low + t_2nd_high) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_2nd_arg * v_2nd_arg) * (t_2nd_high - t_2nd_low) * tex3D<float>(g, float(iu) + 0.5f, (v_2nd_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
			else
			{
				disc_outer = sqrt(disc_outer);
				const float v_arg = z_slope / (b_ti * a_ti_inv) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_arg * v_arg) * 2.0f * disc_outer * a_ti_inv * tex3D<float>(g, float(iu) + 0.5f, (v_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
		}
	}

	int ind;
	if (volumeDimensionOrder == 0)
		ind = j * N_f.z + k;
	else
		ind = k * N_f.y + j;
	f[ind] = curVal;
}

__device__ float AbelConeProjectorKernel_left(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int j, int k, int volumeDimensionOrder)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	const float v = float(j) * T_g.y + startVals_g.y;
	const float u = fabs(float(k) * T_g.z + startVals_g.z);

	const float X = cos_beta - v * sin_beta;

	const float sec_sq_plus_u_sq = X * X + u * u;
	const float b_ti = X * R * cos_beta + u * tau;
	const float a_ti_inv = 1.0f / sec_sq_plus_u_sq;
	float disc_ti_shift = -(u * R * cos_beta - tau * X) * (u * R * cos_beta - tau * X);

	if (fabs(disc_ti_shift) < 1.0e-8)
		disc_ti_shift = 0.0;
	if (disc_ti_shift > 0.0f || fabs(sec_sq_plus_u_sq) < 1.0e-8)
		return 0.0f;

	float curVal = 0.0;
	//const int N_r = int(0.5 + 0.5 * N_f.y);
	//const int N_r = int(0.5 - startVals_f.y / T_f.y);

	const float r_center_ind = -startVals_f.y / T_f.y;
	const int N_r_left = int(floor(r_center_ind)) + 1;
	//const int N_r_right = N_f.y - N_r_left;
	const float r_max = max((N_f.y - 1) * T_f.y + startVals_f.y, fabs(startVals_f.y));

	//const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	//const float r_max = (float)(N_f.y - 1) * T_f.y + startVals_f.y;
	//const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);
	//const float r_min = 0.5f * T_f.y;

	const float z_shift = (-R * sin_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta + v * cos_beta) / T_f.z;

	int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev * r_prev * sec_sq_plus_u_sq < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev * sec_sq_plus_u_sq);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
		//int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		int rInd_min_minus = max(0, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f)));
		if (r_absoluteMinimum < r_max)
		{
			if (volumeDimensionOrder == 0)
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, 0.5f);
			}
			else
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			}
		}
	}

	for (int ir = rInd_min; ir < N_r_left; ir++)
	{
		r_prev = (float)ir * T_f.y;
		const float r_next = (float)(ir + 1) * T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next * sec_sq_plus_u_sq);

		if (volumeDimensionOrder == 0)
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, 0.5f);
		}
		else
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
		}

		// update radius and sqrt for t calculation
		//r_prev = r_next;
		disc_sqrt_prev = disc_sqrt_next;
	}
	return curVal * sqrt(1.0f + u * u + v * v);
}

__device__ float AbelConeProjectorKernel_right(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int j, int k, int volumeDimensionOrder)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	const float v = float(j) * T_g.y + startVals_g.y;
	const float u = fabs(float(k) * T_g.z + startVals_g.z);

	const float X = cos_beta - v * sin_beta;

	const float sec_sq_plus_u_sq = X * X + u * u;
	const float b_ti = X * R * cos_beta + u * tau;
	const float a_ti_inv = 1.0f / sec_sq_plus_u_sq;
	float disc_ti_shift = -(u * R * cos_beta - tau * X) * (u * R * cos_beta - tau * X);

	if (fabs(disc_ti_shift) < 1.0e-8)
		disc_ti_shift = 0.0;
	if (disc_ti_shift > 0.0f || fabs(sec_sq_plus_u_sq) < 1.0e-8)
		return 0.0f;

	float curVal = 0.0;
	//const int N_r = int(0.5 + 0.5 * N_f.y);
	//const int N_r = int(0.5 - startVals_f.y / T_f.y);

	const float r_center_ind = -startVals_f.y / T_f.y;
	const int N_r_left = int(floor(r_center_ind)) + 1;
	const int N_r_right = N_f.y - N_r_left;
	const float r_max = max((N_f.y - 1) * T_f.y + startVals_f.y, fabs(startVals_f.y));

	//const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	//const float r_max = (float)(N_f.y - 1) * T_f.y + startVals_f.y;
	//const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);
	//const float r_min = 0.5f * T_f.y;

	const float z_shift = (-R * sin_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta + v * cos_beta) / T_f.z;

	int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev * r_prev * sec_sq_plus_u_sq < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev * sec_sq_plus_u_sq);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
		//int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		int rInd_min_minus = max(0, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f)));
		if (r_absoluteMinimum < r_max)
		{
			if (volumeDimensionOrder == 0)
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, 0.5f);
			}
			else
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			}
		}
	}

	for (int ir = rInd_min; ir < N_r_right; ir++)
	{
		r_prev = (float)ir * T_f.y;
		const float r_next = (float)(ir + 1) * T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next * sec_sq_plus_u_sq);

		if (volumeDimensionOrder == 0)
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + ir) + 0.5f, 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + ir) + 0.5f, 0.5f);
		}
		else
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + ir) + 0.5f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + ir) + 0.5f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
		}

		// update radius and sqrt for t calculation
		//r_prev = r_next;
		disc_sqrt_prev = disc_sqrt_next;
	}
	return curVal * sqrt(1.0f + u * u + v * v);
	//return 0.0f;
}

__global__ void AbelConeProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;
	else if (float(k) * T_g.z + startVals_g.z < 0.0)
		g[j * N_g.z + k] = AbelConeProjectorKernel_left(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, R, D, axisOfSymmetry, tau, j, k, volumeDimensionOrder);
	else
		g[j * N_g.z + k] = AbelConeProjectorKernel_right(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, R, D, axisOfSymmetry, tau, j, k, volumeDimensionOrder);
}

__device__ float AbelParallelBeamProjectorKernel_left(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int j, int k, int volumeDimensionOrder)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float sec_beta = 1.0f / cos_beta;

	const float v = (float)j * T_g.y + startVals_g.y;
	const float u = fabs((float)k * T_g.z + startVals_g.z);

	const float b_ti = v * sin_beta;
	const float a_ti_inv = sec_beta;
	const float disc_ti_shift = -u * u;

	float curVal = 0.0;
	//const int N_r = int(0.5 + 0.5 * N_f.y);
	//const int N_r = int(0.5 - startVals_f.y / T_f.y);

	const float r_center_ind = -startVals_f.y / T_f.y;
	const int N_r_left = int(floor(r_center_ind)) + 1;
	//const int N_r_right = N_f.y - N_r_left;
	const float r_max = max((N_f.y - 1) * T_f.y + startVals_f.y, fabs(startVals_f.y));

	//const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	//const float r_max = (float)(N_f.y - 1) * T_f.y + startVals_f.y;
	//const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);
	//const float r_min = 0.5f * T_f.y;

	const float z_shift = (v * cos_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta) / T_f.z;

	int rInd_min = (int)ceil(u / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	/*
	if (disc_ti_shift + r_prev * r_prev < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	//*/
	float disc_sqrt_prev = sqrt(max(0.0f, disc_ti_shift + r_prev * r_prev));

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = u;
		//int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		int rInd_min_minus = max(0, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f)));
		if (r_absoluteMinimum < r_max)
		{
			if (volumeDimensionOrder == 0)
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, 0.5f);
			}
			else
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - rInd_min_minus) + 0.5f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			}
		}
	}

	for (int ir = rInd_min; ir < N_r_left; ir++)
	{
		r_prev = ir * T_f.y;
		disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev);
		const float r_next = (ir + 1) * T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next);

		if (volumeDimensionOrder == 0)
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, 0.5f);
		}
		else
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left - 1 - ir) + 0.5f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
		}
	}
	return curVal;
}

__device__ float AbelParallelBeamProjectorKernel_right(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int j, int k, int volumeDimensionOrder)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float sec_beta = 1.0f / cos_beta;

	const float v = (float)j * T_g.y + startVals_g.y;
	const float u = fabs((float)k * T_g.z + startVals_g.z);

	const float b_ti = v * sin_beta;
	const float a_ti_inv = sec_beta;
	const float disc_ti_shift = -u * u;

	float curVal = 0.0;
	//const int N_r = int(0.5 + 0.5 * N_f.y);
	//const int N_r = int(0.5 - startVals_f.y / T_f.y);

	const float r_center_ind = -startVals_f.y / T_f.y;
	const int N_r_left = int(floor(r_center_ind)) + 1;
	const int N_r_right = N_f.y - N_r_left;
	const float r_max = max((N_f.y - 1) * T_f.y + startVals_f.y, fabs(startVals_f.y));

	//const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	//const float r_max = (float)(N_f.y - 1) * T_f.y + startVals_f.y;
	//const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);
	//const float r_min = 0.5f * T_f.y;

	const float z_shift = (v * cos_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta) / T_f.z;

	int rInd_min = (int)ceil(u / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	/*
	if (disc_ti_shift + r_prev * r_prev < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	//*/
	float disc_sqrt_prev = sqrt(max(0.0f, disc_ti_shift + r_prev * r_prev));

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = u;
		//int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		int rInd_min_minus = max(0, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f)));
		if (r_absoluteMinimum < r_max)
		{
			if (volumeDimensionOrder == 0)
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, 0.5f);
			}
			else
			{
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, (b_ti - 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
				curVal += max(0.0f, disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + rInd_min_minus) + 0.5f, (b_ti + 0.5f * (disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			}
		}
	}

	for (int ir = rInd_min; ir < N_r_right; ir++)
	{
		r_prev = ir * T_f.y;
		disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev);
		const float r_next = (ir + 1) * T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next);

		if (volumeDimensionOrder == 0)
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + ir) + 0.5f, 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f, (float)(N_r_left + ir) + 0.5f, 0.5f);
		}
		else
		{
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + ir) + 0.5f, (b_ti - 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
			curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * tex3D<float>(f, 0.5f, (float)(N_r_left + ir) + 0.5f, (b_ti + 0.5f * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift + 0.5f);
		}
	}
	return curVal;
}

__global__ void AbelParallelBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;
	else if (float(k) * T_g.z + startVals_g.z < 0.0)
		g[j * N_g.z + k] = AbelParallelBeamProjectorKernel_left(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, axisOfSymmetry, j, k, volumeDimensionOrder);
	else
		g[j * N_g.z + k] = AbelParallelBeamProjectorKernel_right(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, axisOfSymmetry, j, k, volumeDimensionOrder);
}

__global__ void AbelParallelBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float tan_beta = sin_beta / cos_beta;
	const float sec_beta = 1.0f / cos_beta;

	const float r_unbounded = T_f.y * (float)j + startVals_f.y;
	const float r = fabs(r_unbounded);
	const float r_inner = r - 0.5f * T_f.y;
	const float r_outer = r + 0.5f * T_f.y;

	const float disc_shift_inner = r_inner * r_inner; // r_inner^2
	const float disc_shift_outer = r_outer * r_outer; // r_outer^2

	const float z = T_f.z * (float)k + startVals_f.z;
	const float Tv_inv = 1.0f / T_g.y;

	const float ind_split = -startVals_g.z / T_g.z;
	int iu_max_left, iu_min_right;
	if (fabs(ind_split - floor(0.5f + ind_split)) < 1.0e-4)
	{
		iu_max_left = int(floor(0.5f + ind_split));
		iu_min_right = iu_max_left;
	}
	else
	{
		iu_max_left = int(ind_split);
		iu_min_right = int(ceil(ind_split));
	}

	int iu_min;
	int iu_max;
	if (r_unbounded < 0.0f)
	{
		// left half
		iu_min = 0;
		//iu_max = int(-startVals_g.z / T_g.z);
		iu_max = iu_max_left;
	}
	else
	{
		// right half
		//iu_min = int(ceil(-startVals_g.z / T_g.z));
		iu_min = iu_min_right;
		iu_max = N_g.z - 1;
	}

	float curVal = 0.0;
	for (int iu = iu_min; iu <= iu_max; iu++)
	{
		const float u = fabs(T_g.z * (float)iu + startVals_g.z);
		float disc_outer = disc_shift_outer - u * u; // u^2*(r^2 - R^2) + r^2
		if (disc_outer > 0.0f)
		{
			const float b_ti = z * tan_beta;
			const float a_ti_inv = cos_beta;
			float disc_inner = disc_shift_inner - u * u; // disc_outer > disc_inner
			if (disc_inner > 0.0f)
			{
				disc_inner = sqrt(disc_inner);
				disc_outer = sqrt(disc_outer);
				const float t_1st_low = (b_ti - disc_outer) * a_ti_inv;
				const float t_1st_high = (b_ti - disc_inner) * a_ti_inv;
				const float v_1st_arg = -0.5f * (t_1st_low + t_1st_high) * tan_beta + z * sec_beta;
				curVal += (t_1st_high - t_1st_low) * tex3D<float>(g, float(iu) + 0.5f, (v_1st_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);

				const float t_2nd_low = (b_ti + disc_inner) * a_ti_inv;
				const float t_2nd_high = (b_ti + disc_outer) * a_ti_inv;
				const float v_2nd_arg = -0.5f * (t_2nd_low + t_2nd_high) * tan_beta + z * sec_beta;
				curVal += (t_2nd_high - t_2nd_low) * tex3D<float>(g, float(iu) + 0.5f, (v_2nd_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
			else
			{
				disc_outer = sqrt(disc_outer);
				const float v_arg = -(b_ti * a_ti_inv) * tan_beta + z * sec_beta;
				curVal += 2.0f * disc_outer * a_ti_inv * tex3D<float>(g, float(iu) + 0.5f, (v_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
		}
	}

	int ind;
	if (volumeDimensionOrder == 0)
		ind = j * N_f.z + k;
	else
		ind = k * N_f.y + j;
	f[ind] = curVal * sqrt(1.0f + tan_beta * tan_beta);
}

bool project_symmetric(float*& g, float* f, parameters* params, bool data_on_cpu)
{
	if (params->isSymmetric() == false)
		return false;
	if (params->geometry != parameters::CONE && params->geometry != parameters::PARALLEL)
		return false;
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate projection data on GPU
	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else
		dev_g = g;

	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	if (data_on_cpu)
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	else
		dev_f = f;

	bool useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_g);
	dim3 dimGrid = setGridSize(N_g, dimBlock);
	if (params->geometry == parameters::CONE)
		AbelConeProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->axisOfSymmetry * PI / 180.0, 0.0, params->volumeDimensionOrder);
	else if (params->geometry == parameters::PARALLEL)
		AbelParallelBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->axisOfSymmetry * PI / 180.0, params->volumeDimensionOrder);

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
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_symmetric(float* g, float*& f, parameters* params, bool data_on_cpu)
{
	if (params->isSymmetric() == false)
		return false;
	if (params->geometry != parameters::CONE && params->geometry != parameters::PARALLEL)
		return false;
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate volume data on GPU
	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else
		dev_f = f;

	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

	float rFOVsq = params->rFOV() * params->rFOV();

	if (data_on_cpu)
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	else
		dev_g = g;

	bool useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_f);
	dim3 dimGrid = setGridSize(N_f, dimBlock);
	if (params->geometry == parameters::CONE)
		AbelConeBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->axisOfSymmetry * PI / 180.0, 0.0, params->volumeDimensionOrder);
	else if (params->geometry == parameters::PARALLEL)
		AbelParallelBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->axisOfSymmetry * PI / 180.0, params->volumeDimensionOrder);

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
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool inverse_symmetric(float* g, float*& f, parameters* params, bool data_on_cpu)
{
	if (params->isSymmetric() == false)
		return false;
	if (params->geometry != parameters::CONE && params->geometry != parameters::PARALLEL)
		return false;
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate volume data on GPU
	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else
		dev_f = f;

	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

	float rFOVsq = params->rFOV() * params->rFOV();

	if (data_on_cpu)
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	else
		dev_g = g;

	bool useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_f);
	dim3 dimGrid = setGridSize(N_f, dimBlock);
	if (params->geometry == parameters::CONE)
		AbelConeInverseKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->axisOfSymmetry * PI / 180.0, params->tau, params->volumeDimensionOrder);
	else if (params->geometry == parameters::PARALLEL)
		AbelParallelInverseKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->axisOfSymmetry * PI / 180.0, params->tau, params->volumeDimensionOrder);

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
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}
