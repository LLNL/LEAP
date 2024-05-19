////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for CPU cylindrically symmetric projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors_symmetric_cpu.h"

using namespace std;

//########################################################################################################################################################################
//########################################################################################################################################################################
//### Projectors for Symmetric Objects
//########################################################################################################################################################################
//########################################################################################################################################################################
bool CPUinverse_symmetric(float* g, float* f, parameters* params)
{
	if (params == NULL || g == NULL || f == NULL)
		return false;
	float T_u = params->pixelWidth;
	float T_v = params->pixelHeight;
	float u_0 = params->u_0();
	float v_0 = params->v_0();
	if (params->geometry == parameters::CONE)
	{
		T_u = T_u / params->sdd;
		T_v = T_v / params->sdd;
		u_0 = u_0 / params->sdd;
		v_0 = v_0 / params->sdd;
	}

	float cos_beta = cos(params->axisOfSymmetry * PI / 180.0);
	float sin_beta = sin(params->axisOfSymmetry * PI / 180.0);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	int N_phi = params->numCols + ((params->numCols + 1) % 2);
	float T_phi = 2.0f * 3.1415926535897932385f / float(N_phi);
	float R_sq = params->sod * params->sod;

	float undoFBPscaling = 1.0;
	if (params->geometry == parameters::PARALLEL)
		undoFBPscaling = params->voxelWidth * params->voxelWidth / (params->pixelWidth);
	else
		undoFBPscaling = params->voxelWidth * params->voxelWidth * params->voxelHeight / (params->sod*params->sod*T_u*T_v);
	float scaling = undoFBPscaling / (2.0 * float(N_phi));
	//float scaling = params->pixelWidth * 0.25 * PI / (2.0 * float(N_phi));
	//if (params->geometry == parameters::PARALLEL)
	//	scaling = params->pixelWidth / (2.0 * float(N_phi));

	params->setToZero(f, params->numZ * params->numY * params->numX);

	// XYZ and numX=1

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	for (int iy = 0; iy < params->numY; iy++)
	{
		float r_val = params->voxelWidth * float(iy) + params->y_0();
		float* zLine = &f[iy * params->numZ];
		for (int iphi = 0; iphi < N_phi; iphi++)
		{
			float phi = float(iphi) * T_phi - 0.5 * PI;
			float cos_phi = cos(phi);
			float sin_phi = sin(phi);

			if (params->geometry == parameters::CONE)
			{
				for (int iz = 0; iz < params->numZ; iz++)
				{
					float z_val = params->voxelHeight * float(iz) + params->z_0();

					float v_denom_inv = 1.0f / (params->sod + r_val * sin_phi * cos_beta - z_val * sin_beta);
					float s_val = (params->tau - r_val * cos_phi) * v_denom_inv;
					if (r_val * s_val < 0.0)
						s_val *= -1.0;
					float s_arg = (s_val - u_0) / T_u;
					float v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) * v_denom_inv - v_0) / T_v;
					float theWeight = R_sq * v_denom_inv * v_denom_inv * scaling;

					if (s_arg < 0.0 || s_arg > float(params->numCols - 1))
						continue;

					s_arg = max(float(0.0), min(s_arg, float(params->numCols - 1)));
					int s_low = int(s_arg);
					int s_high = min(s_low + 1, params->numCols - 1);
					float ds = s_arg - float(s_low);

					v_arg = max(float(0.0), min(v_arg, float(params->numRows - 1)));
					int v_low = int(v_arg);
					int v_high = min(v_low + 1, params->numRows - 1);
					float dv = v_arg - float(v_low);

					zLine[iz] += ((1.0 - dv) * ((1.0 - ds) * g[v_low * params->numCols + s_low] + ds * g[v_low * params->numCols + s_high])
						+ dv * ((1.0 - ds) * g[v_high * params->numCols + s_low] + ds * g[v_high * params->numCols + s_high])) * theWeight;
				}
			}
			else
			{
				float s_val = r_val * cos_phi;

				if (r_val * s_val < 0.0)
					s_val *= -1.0;

				float s_arg = (s_val - u_0) / T_u;

				if (s_arg < 0.0 || s_arg > float(params->numCols - 1))
					continue;

				s_arg = max(float(0.0), min(s_arg, float(params->numCols-1)));
				int s_low = int(s_arg);
				int s_high = min(s_low + 1, params->numCols-1);
				float ds = s_arg - float(s_low);

				for (int iz = 0; iz < params->numZ; iz++)
				{
					float z_val = params->voxelHeight * float(iz) + params->z_0();
					float v_arg = ((r_val * sin_phi * sin_beta + z_val * cos_beta) - v_0) / T_v;
					float theWeight = scaling;

					zLine[iz] += ((1.0 - ds) * g[iz*params->numCols + s_low] + ds* g[iz * params->numCols + s_high]) * theWeight;
				}
			}
		}
	}

	if (params->volumeDimensionOrder == 1)
		reorder_YZ_to_ZY(f, params);

	return true;
}

bool CPUproject_symmetric(float* g, float* f, parameters* params)
{
	if (params == NULL)
		return false;
	else
	{
		bool retVal = true;
		float* f_YZ = f;
		if (params->volumeDimensionOrder == 1)
			f_YZ = reorder_ZY_to_YZ_copy(f, params);

		if (params->geometry == parameters::CONE)
			retVal = CPUproject_AbelCone(g, f_YZ, params);
		else if (params->geometry == parameters::PARALLEL)
			retVal = CPUproject_AbelParallel(g, f_YZ, params);
		else
			retVal = false;

		if (params->volumeDimensionOrder == 1)
			free(f_YZ);

		return retVal;
	}
}

bool CPUbackproject_symmetric(float* g, float* f, parameters* params)
{
	if (params == NULL)
		return false;
	else
	{
		bool retVal = true;
		if (params->geometry == parameters::CONE)
			retVal = CPUbackproject_AbelCone(g, f, params);
		else if (params->geometry == parameters::PARALLEL)
			retVal = CPUbackproject_AbelParallel(g, f, params);
		else
			retVal = false;

		if (params->volumeDimensionOrder == 1)
			reorder_YZ_to_ZY(f, params);
		return retVal;
	}
}

bool CPUproject_AbelCone(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;

	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	double cos_beta = cos(params->axisOfSymmetry*PI / 180.0);
	double sin_beta = sin(params->axisOfSymmetry*PI / 180.0);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	double T_v = params->pixelHeight / params->sdd;
	double v_0 = params->v_0() / params->sdd;
	double T_u = params->pixelWidth / params->sdd;
	double u_0 = params->u_0() / params->sdd;
	double x_0 = params->x_0();
	double y_0 = params->y_0();
	double z_0 = params->z_0();


	double r_center_ind = -y_0 / params->voxelWidth;
	int N_r_left = int(floor(r_center_ind)) + 1;
	int N_r_right = params->numY - N_r_left;

	double r_max = max((params->numY - 1) * params->voxelWidth + y_0, fabs(y_0));

	int N_r = int(0.5 - y_0 / params->voxelWidth);
	//double r_max = (params->numY - 1)*params->voxelWidth + y_0;

	double Rcos_sq_plus_tau_sq = params->sod*params->sod*cos_beta*cos_beta + params->tau*params->tau;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < params->numRows; j++)
	{
		double v = j * T_v + v_0;
		double X = cos_beta - v * sin_beta;

		double z_shift = (-params->sod*sin_beta - z_0) / params->voxelHeight;
		double z_slope = (sin_beta + v * cos_beta) / params->voxelHeight;

		float* projLine = &g[j*params->numCols];

		for (int k = 0; k < params->numCols; k++)
		{
			double u_unbounded = k * T_u + u_0;
			double u = fabs(u_unbounded);

			int rInd_max;
			if (u_unbounded < 0.0)
				rInd_max = N_r_left;
			else
				rInd_max = N_r_right;
			double r_min = 0.5 * params->voxelWidth;// r_min = fabs((float)(rInd_floor)*params->voxelWidth + y_0);

			double sec_sq_plus_u_sq = X * X + u * u;
			double b_ti = X * params->sod*cos_beta + u * params->tau;
			double a_ti_inv = 1.0 / sec_sq_plus_u_sq;
			double disc_ti_shift = b_ti * b_ti - sec_sq_plus_u_sq * Rcos_sq_plus_tau_sq; // new

			if (fabs(disc_ti_shift) < 1.0e-8)
				disc_ti_shift = 0.0;
			if (fabs(sec_sq_plus_u_sq) < 1.0e-8 || disc_ti_shift > 0.0)
			{
				projLine[k] = 0.0;
				continue;
			}

			int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / params->voxelWidth);
			double r_prev = double(rInd_min)*params->voxelWidth;
			double disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq);

			double curVal = 0.0;

			// Go back one sample and check
			if (rInd_min >= 1)
			{
				double r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
				//int rInd_min_minus = max(0, min(N_r - 1, int(ceil((r_absoluteMinimum) / params->voxelWidth - 1.0))));
				int rInd_min_minus = max(0, int(ceil((r_absoluteMinimum) / params->voxelWidth - 1.0)));

				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r_left - 1 - rInd_min_minus;
				else
					ir_shifted_or_flipped = N_r_left + rInd_min_minus;

				if (r_absoluteMinimum < r_max && disc_sqrt_prev > 0.0 && 0 <= ir_shifted_or_flipped && ir_shifted_or_flipped <= params->numY - 1)
				{
					double iz_arg_low = (b_ti - 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
					if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
					{
						int iz_arg_low_floor = int(iz_arg_low);
						double dz = iz_arg_low - double(iz_arg_low_floor);
						int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);
						curVal += disc_sqrt_prev * a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_low_ceil]);
					}

					double iz_arg_high = (b_ti + 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
					if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
					{
						int iz_arg_high_floor = int(iz_arg_high);
						double dz = iz_arg_high - double(iz_arg_high_floor);
						int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);
						curVal += disc_sqrt_prev * a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_high_ceil]);
					}
				}
			}

			for (int ir = rInd_min; ir < rInd_max; ir++) // FIXME
			{
				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r_left - 1 - ir;
				else
					ir_shifted_or_flipped = N_r_left + ir;

				double r_next = r_prev + params->voxelWidth;
				double disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next*sec_sq_plus_u_sq);

				if (0 <= ir_shifted_or_flipped && ir_shifted_or_flipped <= params->numY - 1)
				{
					// Negative t interval
					// low:  (b_ti - disc_sqrt_next) * a_ti_inv
					// high: (b_ti - disc_sqrt_prev) * a_ti_inv

					// Positive t interval
					// low:  (b_ti + disc_sqrt_prev) * a_ti_inv
					// high: (b_ti + disc_sqrt_next) * a_ti_inv

					//(b_ti - disc_sqrt_next) * a_ti_inv + (b_ti - disc_sqrt_prev) * a_ti_inv
					double iz_arg_low = (b_ti - 0.5 * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift;
					if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
					{
						int iz_arg_low_floor = int(iz_arg_low);
						double dz = iz_arg_low - double(iz_arg_low_floor);
						int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);

						curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * ((1.0 - dz) * f[ir_shifted_or_flipped * params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped * params->numZ + iz_arg_low_ceil]);
					}

					double iz_arg_high = (b_ti + 0.5 * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift;
					if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
					{
						int iz_arg_high_floor = int(iz_arg_high);
						double dz = iz_arg_high - double(iz_arg_high_floor);
						int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);

						curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * ((1.0 - dz) * f[ir_shifted_or_flipped * params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped * params->numZ + iz_arg_high_ceil]);
					}
				}

				// update radius and sqrt for t calculation
				r_prev = r_next;
				disc_sqrt_prev = disc_sqrt_next;
			}
			projLine[k] = curVal * sqrt(1.0 + u * u + v * v);
		}
	}
	return true;
}

bool CPUbackproject_AbelCone(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	//params->setToZero(f, params->numX*params->numY*params->numZ);
	double cos_beta = cos(params->axisOfSymmetry*PI / 180.0);
	double sin_beta = sin(params->axisOfSymmetry*PI / 180.0);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}
	double tan_beta = sin_beta / cos_beta;
	double sec_beta = 1.0 / cos_beta;

	double T_v = params->pixelHeight / params->sdd;
	double v_0 = params->v_0() / params->sdd;
	double T_u = params->pixelWidth / params->sdd;
	double u_0 = params->u_0() / params->sdd;
	double T_y = params->voxelWidth;
	double T_z = params->voxelHeight;
	double x_0 = params->x_0();
	double y_0 = params->y_0();
	double z_0 = params->z_0();

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < params->numY; j++)
	{
		float* zLine = &f[j*params->numZ];
		double r_unbounded = j * T_y + y_0;
		double r = fabs(r_unbounded);
		double r_inner = r - 0.5*T_y;
		double r_outer = r + 0.5*T_y;

		//*
		double ind_split = -u_0 / T_u;
		int iu_max_left, iu_min_right;
		if (fabs(ind_split - floor(0.5 + ind_split)) < 1.0e-8)
		{
			iu_max_left = int(floor(0.5 + ind_split));
			iu_min_right = iu_max_left;
		}
		else
		{
			iu_max_left = int(ind_split);
			iu_min_right = int(ceil(ind_split));
		}

		int iu_min;
		int iu_max;
		if (r_unbounded < 0.0)
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
			iu_max = params->numCols - 1;
		}
		//*/

		/*
		int iu_min;
		int iu_max;
		if (r_unbounded < 0.0)
		{
			// left half
			iu_min = 0;
			iu_max = int(-u_0 / T_u);
		}
		else
		{
			// right half
			iu_min = int(ceil(-u_0 / T_u));
			iu_max = params->numCols - 1;
		}
		//*/

		double disc_shift_inner = (r_inner*r_inner - params->tau*params->tau)*sec_beta*sec_beta; // r_inner^2
		double disc_shift_outer = (r_outer*r_outer - params->tau*params->tau)*sec_beta*sec_beta; // r_outer^2

		for (int k = 0; k < params->numZ; k++)
		{
			double z = k * T_z + z_0;

			double Z = (params->sod + z * sin_beta)*sec_beta; // nominal value: R
			double z_slope = (z + params->sod*sin_beta)*sec_beta; // nominal value: 0

			double curVal = 0.0;
			for (int iu = iu_min; iu <= iu_max; iu++)
			{
				double u = fabs(iu * T_u + u_0);

				double disc_outer = u * u*(r_outer*r_outer - Z * Z) + 2.0*Z*sec_beta*params->tau*u + disc_shift_outer; // u^2*(r^2 - R^2) + r^2
				if (disc_outer > 0.0)
				{
					//realnum* projCol = aProj[iu];

					double b_ti = Z * sec_beta + params->tau*u;
					double a_ti_inv = 1.0 / (u*u + sec_beta * sec_beta);

					double disc_inner = u * u*(r_inner*r_inner - Z * Z) + 2.0*Z*sec_beta*params->tau*u + disc_shift_inner; // disc_outer > disc_inner
					if (disc_inner > 0.0)
					{
						disc_inner = sqrt(disc_inner);
						disc_outer = sqrt(disc_outer);
						// first t interval
						// t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti-sqrt(disc_inner))*a_ti_inv
						double t_1st_low = (b_ti - disc_outer)*a_ti_inv;
						double t_1st_high = (b_ti - disc_inner)*a_ti_inv;
						double v_1st_arg = 2.0*z_slope / (t_1st_low + t_1st_high) - tan_beta;

						double theWeight_1st = sqrt(1.0 + u * u + v_1st_arg * v_1st_arg + tan_beta * tan_beta) * (t_1st_high - t_1st_low);

						v_1st_arg = max(0.0, min(double(params->numRows - 1.001), (v_1st_arg - v_0) / T_v));
						int v_1st_arg_floor = int(v_1st_arg);
						double dv_1st = v_1st_arg - double(v_1st_arg_floor);
						curVal += theWeight_1st * ((1.0 - dv_1st)*g[v_1st_arg_floor*params->numCols + iu] + dv_1st * g[(v_1st_arg_floor + 1)*params->numCols + iu]);

						// second t interval
						// t interval: (b_ti+sqrt(disc_inner))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
						double t_2nd_low = (b_ti + disc_inner)*a_ti_inv;
						double t_2nd_high = (b_ti + disc_outer)*a_ti_inv;
						double v_2nd_arg = 2.0*z_slope / (t_2nd_low + t_2nd_high) - tan_beta;

						double theWeight_2nd = sqrt(1.0 + u * u + v_2nd_arg * v_2nd_arg + tan_beta * tan_beta) * (t_2nd_high - t_2nd_low);

						v_2nd_arg = max(0.0, min(double(params->numRows - 1.001), (v_2nd_arg - v_0) / T_v));
						int v_2nd_arg_floor = int(v_2nd_arg);
						double dv_2nd = v_2nd_arg - double(v_2nd_arg_floor);
						curVal += theWeight_2nd * ((1.0 - dv_2nd)*g[v_2nd_arg_floor*params->numCols + iu] + dv_2nd * g[(v_2nd_arg_floor + 1)*params->numCols + iu]);
					}
					else
					{
						disc_outer = sqrt(disc_outer);
						// t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
						// t interval midpoint: b_ti*a_ti_inv

						// take mid value for interval to find iv
						double v_arg = z_slope / (b_ti*a_ti_inv) - tan_beta;

						double theWeight = sqrt(1.0 + u * u + v_arg * v_arg + tan_beta * tan_beta) * 2.0*disc_outer*a_ti_inv;

						v_arg = max(0.0, min(double(params->numRows - 1.001), (v_arg - v_0) / T_v));
						int v_arg_floor = int(v_arg);
						double dv = v_arg - double(v_arg_floor);
						curVal += theWeight * ((1.0 - dv)*g[v_arg_floor*params->numCols + iu] + dv * g[(v_arg_floor + 1)*params->numCols + iu]);
					}
				}
			}
			zLine[k] = curVal;
		}
	}
	return f;
}

bool CPUproject_AbelParallel(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;

	double cos_beta = cos(params->axisOfSymmetry*PI / 180.0);
	double sin_beta = sin(params->axisOfSymmetry*PI / 180.0);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}
	//double tan_beta = sin_beta/cos_beta;
	double sec_beta = 1.0 / cos_beta;

	double T_v = params->pixelHeight;
	double v_0 = params->v_0();
	double T_u = params->pixelWidth;
	double u_0 = params->u_0();
	double x_0 = params->x_0();
	double y_0 = params->y_0();
	double z_0 = params->z_0();
	double T_z = params->voxelHeight;

	//int N_r = int(0.5 + 0.5*params->numY);
	int N_r = int(0.5 - y_0 / params->voxelWidth);
	double T_r = params->voxelWidth;

	double r_center_ind = -y_0 / params->voxelWidth;
	int N_r_left = int(floor(r_center_ind)) + 1;
	int N_r_right = params->numY - N_r_left;
	double r_max = max((params->numY - 1) * params->voxelWidth + y_0, fabs(y_0));

	//double r_max = (params->numY - 1)*params->voxelWidth + y_0;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < params->numRows; j++)
	{
		double v = j * T_v + v_0;//cos_beta;
		double Y = sin_beta + v * cos_beta;
		double X = cos_beta - v * sin_beta;

		double z_shift = (v*cos_beta - z_0) / T_z;
		double z_slope = sin_beta / T_z;

		for (int k = 0; k < params->numCols; k++)
		{
			double u_unbounded = k * T_u + u_0;
			double u = fabs(u_unbounded);

			int rInd_max;
			if (u_unbounded < 0.0)
				rInd_max = N_r_left;
			else
				rInd_max = N_r_right;
			double r_min = 0.5 * T_r;

			double b_ti = v * sin_beta;
			double a_ti_inv = sec_beta;
			double disc_ti_shift = -u * u;

			int rInd_min = (int)ceil(u / T_r);
			double r_prev = double(rInd_min)*T_r;
			double disc_sqrt_prev = sqrt(max(0.0, disc_ti_shift + r_prev * r_prev));

			double curVal = 0.0;

			//*
			//####################################################################################
			// Go back one sample and check
			if (rInd_min >= 1)
			{
				double r_absoluteMinimum = u;
				//double disc_sqrt_check = sqrt(disc_ti_shift + r_absoluteMinimum*r_absoluteMinimum*sec_sq_plus_u_sq);
				//int rInd_min_minus = max(0, int(floor(0.5+r_absoluteMinimum/T_r)));
				//int rInd_min_minus = max(0, min(N_r - 1, int(ceil(r_absoluteMinimum / T_r - 1.0))));
				int rInd_min_minus = max(0, int(ceil(r_absoluteMinimum / T_r - 1.0)));

				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r_left - 1 - rInd_min_minus;
				else
					ir_shifted_or_flipped = N_r_left + rInd_min_minus;

				if (r_absoluteMinimum < r_max && 0 <= ir_shifted_or_flipped && ir_shifted_or_flipped <= params->numY - 1)
				{
					double iz_arg_low = (b_ti - 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
					if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
					{
						int iz_arg_low_floor = int(iz_arg_low);
						double dz = iz_arg_low - double(iz_arg_low_floor);
						int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);

						curVal += max(0.0, disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_low_ceil]);
					}

					double iz_arg_high = (b_ti + 0.5*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
					if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
					{
						int iz_arg_high_floor = int(iz_arg_high);
						double dz = iz_arg_high - double(iz_arg_high_floor);
						int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);

						curVal += max(0.0, disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_high_ceil]);
					}
				}
			}
			//####################################################################################
			//*/

			for (int ir = rInd_min; ir < rInd_max; ir++) // FIXME
			{
				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r_left - 1 - ir;
				else
					ir_shifted_or_flipped = N_r_left + ir;

				double r_next = r_prev + T_r;
				double disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next);

				if (0 <= ir_shifted_or_flipped && ir_shifted_or_flipped <= params->numY - 1)
				{
					// Negative t interval
					// low:  (b_ti - disc_sqrt_next) * a_ti_inv
					// high: (b_ti - disc_sqrt_prev) * a_ti_inv

					// Positive t interval
					// low:  (b_ti + disc_sqrt_prev) * a_ti_inv
					// high: (b_ti + disc_sqrt_next) * a_ti_inv

					//(b_ti - disc_sqrt_next) * a_ti_inv + (b_ti - disc_sqrt_prev) * a_ti_inv
					double iz_arg_low = (b_ti - 0.5 * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift;
					if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
					{
						int iz_arg_low_floor = int(iz_arg_low);
						double dz = iz_arg_low - double(iz_arg_low_floor);
						int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);

						curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * ((1.0 - dz) * f[ir_shifted_or_flipped * params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped * params->numZ + iz_arg_low_ceil]);
					}

					double iz_arg_high = (b_ti + 0.5 * (disc_sqrt_next + disc_sqrt_prev)) * a_ti_inv * z_slope + z_shift;
					if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
					{
						int iz_arg_high_floor = int(iz_arg_high);
						double dz = iz_arg_high - double(iz_arg_high_floor);
						int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);

						curVal += (disc_sqrt_next - disc_sqrt_prev) * a_ti_inv * ((1.0 - dz) * f[ir_shifted_or_flipped * params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped * params->numZ + iz_arg_high_ceil]);
					}
				}

				// update radius and sqrt for t calculation
				r_prev = r_next;
				disc_sqrt_prev = disc_sqrt_next;
			}
			g[j*params->numCols + k] = curVal;
		}
	}

	return true;
}

bool CPUbackproject_AbelParallel(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;

	double cos_beta = cos(params->axisOfSymmetry*PI / 180.0);
	double sin_beta = sin(params->axisOfSymmetry*PI / 180.0);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}
	double tan_beta = sin_beta / cos_beta;
	double sec_beta = 1.0 / cos_beta;

	double T_v = params->pixelHeight;
	double v_0 = params->v_0();
	double T_u = params->pixelWidth;
	double u_0 = params->u_0();
	double T_y = params->voxelWidth;
	double T_z = params->voxelHeight;
	double x_0 = params->x_0();
	double y_0 = params->y_0();
	double z_0 = params->z_0();

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < params->numY; j++)
	{
		float* zLine = &f[j*params->numZ];
		double r_unbounded = j * T_y + y_0;
		double r = fabs(r_unbounded);
		double r_inner = r - 0.5*T_y;
		double r_outer = r + 0.5*T_y;

		double ind_split = -u_0 / T_u;
		int iu_max_left, iu_min_right;
		if (fabs(ind_split - floor(0.5 + ind_split)) < 1.0e-8)
		{
			iu_max_left = int(floor(0.5 + ind_split));
			iu_min_right = iu_max_left;
		}
		else
		{
			iu_max_left = int(ind_split);
			iu_min_right = int(ceil(ind_split));
		}

		int iu_min;
		int iu_max;
		if (r_unbounded < 0.0)
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
			iu_max = params->numCols - 1;
		}

		/*
		int iu_min;
		int iu_max;
		if (r_unbounded < 0.0)
		{
			// left half
			iu_min = 0;
			iu_max = int(-u_0 / T_u);
		}
		else
		{
			// right half
			iu_min = int(ceil(-u_0 / T_u));
			iu_max = params->numCols - 1;
		}
		//*/

		double disc_shift_inner = r_inner * r_inner; // r_inner^2
		double disc_shift_outer = r_outer * r_outer; // r_outer^2

		for (int k = 0; k < params->numZ; k++)
		{
			double z = k * T_z + z_0;

			//double Z = (g->R + z*sin_beta)*sec_beta; // nominal value: R
			//double z_slope = (z + g->R*sin_beta)*sec_beta; // nominal value: 0

			double curVal = 0.0;
			for (int iu = iu_min; iu <= iu_max; iu++)
			{
				double u = fabs(iu * T_u + u_0);

				double disc_outer = disc_shift_outer - u * u; // u^2*(r^2 - R^2) + r^2
				if (disc_outer > 0.0)
				{
					//realnum* projCol = aProj[iu];

					double b_ti = z * tan_beta;
					double a_ti_inv = cos_beta;

					double disc_inner = disc_shift_inner - u * u; // disc_outer > disc_inner
					if (disc_inner > 0.0)
					{
						disc_inner = sqrt(disc_inner);
						disc_outer = sqrt(disc_outer);
						// first t interval
						// t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti-sqrt(disc_inner))*a_ti_inv
						double t_1st_low = (b_ti - disc_outer)*a_ti_inv;
						double t_1st_high = (b_ti - disc_inner)*a_ti_inv;
						double v_1st_arg = -0.5*(t_1st_low + t_1st_high)*tan_beta + z * sec_beta;

						//double theWeight_1st = sqrt(1.0 + u*u + v_1st_arg*v_1st_arg + tan_beta*tan_beta) * (t_1st_high - t_1st_low);

						v_1st_arg = max(0.0, min(double(params->numZ - 1.001), (v_1st_arg - z_0) / T_z));
						int v_1st_arg_floor = int(v_1st_arg);
						double dv_1st = v_1st_arg - double(v_1st_arg_floor);
						curVal += (t_1st_high - t_1st_low)*((1.0 - dv_1st)*g[v_1st_arg_floor*params->numCols + iu] + dv_1st * g[(v_1st_arg_floor + 1)*params->numCols + iu]);

						// second t interval
						// t interval: (b_ti+sqrt(disc_inner))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
						double t_2nd_low = (b_ti + disc_inner)*a_ti_inv;
						double t_2nd_high = (b_ti + disc_outer)*a_ti_inv;
						double v_2nd_arg = -0.5*(t_2nd_low + t_2nd_high)*tan_beta + z * sec_beta;

						//double theWeight_2nd = sqrt(1.0 + u*u + v_2nd_arg*v_2nd_arg + tan_beta*tan_beta) * (t_2nd_high - t_2nd_low);

						v_2nd_arg = max(0.0, min(double(params->numZ - 1.001), (v_2nd_arg - z_0) / T_z));
						int v_2nd_arg_floor = int(v_2nd_arg);
						double dv_2nd = v_2nd_arg - double(v_2nd_arg_floor);
						curVal += (t_2nd_high - t_2nd_low) * ((1.0 - dv_2nd)*g[v_2nd_arg_floor*params->numCols + iu] + dv_2nd * g[(v_2nd_arg_floor + 1)*params->numCols + iu]);
					}
					else
					{
						disc_outer = sqrt(disc_outer);
						// t interval: (b_ti-sqrt(disc_outer))*a_ti_inv to (b_ti+sqrt(disc_outer))*a_ti_inv
						// t interval midpoint: b_ti*a_ti_inv

						// take mid value for interval to find iv
						double v_arg = -(b_ti*a_ti_inv)*tan_beta + z * sec_beta;

						//double theWeight = sqrt(1.0 + u*u + v_arg*v_arg + tan_beta*tan_beta) * 2.0*disc_outer*a_ti_inv;

						v_arg = max(0.0, min(double(params->numZ - 1.001), (v_arg - z_0) / T_z));
						int v_arg_floor = int(v_arg);
						double dv = v_arg - double(v_arg_floor);
						curVal += 2.0*disc_outer*a_ti_inv*((1.0 - dv)*g[v_arg_floor*params->numCols + iu] + dv * g[(v_arg_floor + 1)*params->numCols + iu]);
					}
				}
			}
			zLine[k] = curVal * sqrt(1.0 + tan_beta * tan_beta);
		}
	}
	return true;
}

bool reorder_ZY_to_YZ(float* f, parameters* params)
{
	if (f == NULL || params == NULL)
		return false;
	else
	{
		float* f_copy = (float*)malloc(sizeof(float) * params->numZ * params->numY);
		memcpy(f_copy, f, size_t(sizeof(float) * params->numZ * params->numY));
		for (int iy = 0; iy < params->numY; iy++)
		{
			for (int iz = 0; iz < params->numZ; iz++)
				f[iy*params->numZ + iz] = f_copy[iz*params->numY + iy];
		}

		free(f_copy);
		return true;
	}
}

bool reorder_YZ_to_ZY(float* f, parameters* params)
{
	if (f == NULL || params == NULL)
		return false;
	else
	{
		float* f_copy = (float*)malloc(sizeof(float) * params->numZ * params->numY);
		memcpy(f_copy, f, size_t(sizeof(float) * params->numZ * params->numY));
		for (int iz = 0; iz < params->numZ; iz++)
		{
			for (int iy = 0; iy < params->numY; iy++)
				f[iz * params->numY + iy] = f_copy[iy * params->numZ + iz];
		}

		free(f_copy);
		return true;
	}
}

float* reorder_ZY_to_YZ_copy(float* f, parameters* params)
{
	if (f == NULL || params == NULL)
		return NULL;
	else
	{
		float* f_copy = (float*)malloc(sizeof(float) * params->numZ * params->numY);
		for (int iy = 0; iy < params->numY; iy++)
		{
			for (int iz = 0; iz < params->numZ; iz++)
				f_copy[iy * params->numZ + iz] = f[iz * params->numY + iy];
		}

		return f_copy;
	}
}
