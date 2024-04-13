////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// C++ module for CPU Siddon projector (deprecated)
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors_Siddon_cpu.h"
#include "cpu_utils.h"

using namespace std;


bool CPUproject_cone(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	windowFOV_cpu(f, params);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < params->numAngles; i++)
	{
		float cos_phi = cos(params->phis[i]);
		float sin_phi = sin(params->phis[i]);
		float sourcePos[3];
		sourcePos[0] = params->sod*cos_phi;
		sourcePos[1] = params->sod*sin_phi;
		sourcePos[2] = 0.0f;

		float* aProj = &g[i*params->numRows*params->numCols];

		omp_set_num_threads(num_thread_row);
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < params->numRows; j++)
		{
			float v = params->pixelHeight*j + params->v_0();
			float* aLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
			{
				float u = params->pixelWidth*k + params->u_0();

				float traj[3];
				traj[0] = -params->sdd*cos_phi - u * sin_phi;
				traj[1] = -params->sdd*sin_phi + u * cos_phi;
				traj[2] = v;
				aLine[k] = projectLine(f, params, sourcePos, traj);
			}
		}
	}
	return true;
}

bool CPUbackproject_cone(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	params->setToZero(f, params->numX*params->numY*params->numZ);
	const float x_width = 0.5f*params->voxelWidth;
	const float y_width = 0.5f*params->voxelWidth;
	const float z_width = 0.5f*params->voxelHeight;

    float rFOVsq = params->rFOV()*params->rFOV();
    
	double R = params->sod;
	double D = params->sdd;
	double v_0 = params->v_0();
	double u_0 = params->u_0();

	const float voxelToMagnifiedDetectorPixelRatio_u = params->voxelWidth / (R / D * params->pixelWidth);
	const float voxelToMagnifiedDetectorPixelRatio_v = params->voxelHeight / (R / D * params->pixelHeight);

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	const int searchWidth_v = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_v);

	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int ix = 0; ix < params->numX; ix++)
		{
			const float x = ix * params->voxelWidth + params->x_0();
			float* xSlice = &f[ix * params->numY * params->numZ];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				const float cos_phi = cos(params->phis[iphi]);
				const float sin_phi = sin(params->phis[iphi]);

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();

					if (x * x + y * y <= rFOVsq)
					{
						float* zLine = &xSlice[iy * params->numZ];
						for (int iz = 0; iz < params->numZ; iz++)
						{
							const float z = iz * params->voxelHeight + params->z_0();

							float pos[3];
							pos[0] = x - R * cos_phi;
							pos[1] = y - R * sin_phi;
							pos[2] = z;

							const float v_denom = R - x * cos_phi - y * sin_phi;
							const int u_arg_mid = int(0.5 + (D * (y * cos_phi - x * sin_phi) / v_denom - u_0) / params->pixelWidth);
							const int v_arg_mid = int(0.5 + (D * z / v_denom - v_0) / params->pixelHeight);

							int iv_min = max(0, v_arg_mid - searchWidth_v);
							int iv_max = min(params->numRows - 1, v_arg_mid + searchWidth_v);
							int iu_min = max(0, u_arg_mid - searchWidth_u);
							int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

							float val = 0.0f;
							for (int iv = iv_min; iv <= iv_max; iv++)
							{
								const float v = iv * params->pixelHeight + v_0;
								for (int iu = iu_min; iu <= iu_max; iu++)
								{
									const float u = iu * params->pixelWidth + u_0;

									float traj[3];
									const float trajLength_inv = 1.0 / sqrt(D * D + u * u + v * v);
									traj[0] = (-D * cos_phi - u * sin_phi) * trajLength_inv;
									traj[1] = (-D * sin_phi + u * cos_phi) * trajLength_inv;
									traj[2] = v * trajLength_inv;

									float t_max = float(1.0e16);
									float t_min = float(-1.0e16);
									if (traj[0] != 0.0f)
									{
										const float t_a = (pos[0] + x_width) / traj[0];
										const float t_b = (pos[0] - x_width) / traj[0];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}
									if (traj[1] != 0.0f)
									{
										const float t_a = (pos[1] + y_width) / traj[1];
										const float t_b = (pos[1] - y_width) / traj[1];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}
									if (traj[2] != 0.0f)
									{
										const float t_a = (pos[2] + z_width) / traj[2];
										const float t_b = (pos[2] - z_width) / traj[2];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}

									val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
								}
							}
							zLine[iz] += val;
						}
					}
				}
			}
		}
	}
	else
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int iz = 0; iz < params->numZ; iz++)
		{
			const float z = iz * params->voxelHeight + params->z_0();
			float* zSlice = &f[iz * params->numY * params->numX];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				const float cos_phi = cos(params->phis[iphi]);
				const float sin_phi = sin(params->phis[iphi]);

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();
					float* xLine = &zSlice[iy * params->numX];
					for (int ix = 0; ix < params->numX; ix++)
					{
						const float x = ix * params->voxelWidth + params->x_0();

						if (x * x + y * y <= rFOVsq)
						{
							float pos[3];
							pos[0] = x - R * cos_phi;
							pos[1] = y - R * sin_phi;
							pos[2] = z;

							const float v_denom = R - x * cos_phi - y * sin_phi;
							const int u_arg_mid = int(0.5 + (D * (y * cos_phi - x * sin_phi) / v_denom - u_0) / params->pixelWidth);
							const int v_arg_mid = int(0.5 + (D * z / v_denom - v_0) / params->pixelHeight);

							int iv_min = max(0, v_arg_mid - searchWidth_v);
							int iv_max = min(params->numRows - 1, v_arg_mid + searchWidth_v);
							int iu_min = max(0, u_arg_mid - searchWidth_u);
							int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

							float val = 0.0f;
							for (int iv = iv_min; iv <= iv_max; iv++)
							{
								const float v = iv * params->pixelHeight + v_0;
								for (int iu = iu_min; iu <= iu_max; iu++)
								{
									const float u = iu * params->pixelWidth + u_0;

									float traj[3];
									const float trajLength_inv = 1.0 / sqrt(D * D + u * u + v * v);
									traj[0] = (-D * cos_phi - u * sin_phi) * trajLength_inv;
									traj[1] = (-D * sin_phi + u * cos_phi) * trajLength_inv;
									traj[2] = v * trajLength_inv;

									float t_max = float(1.0e16);
									float t_min = float(-1.0e16);
									if (traj[0] != 0.0f)
									{
										const float t_a = (pos[0] + x_width) / traj[0];
										const float t_b = (pos[0] - x_width) / traj[0];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}
									if (traj[1] != 0.0f)
									{
										const float t_a = (pos[1] + y_width) / traj[1];
										const float t_b = (pos[1] - y_width) / traj[1];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}
									if (traj[2] != 0.0f)
									{
										const float t_a = (pos[2] + z_width) / traj[2];
										const float t_b = (pos[2] - z_width) / traj[2];
										t_max = min(t_max, max(t_b, t_a));
										t_min = max(t_min, min(t_b, t_a));
									}

									val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
								}
							}
							xLine[ix] += val;
						}
					}
				}
			}
		}
	}
	return true;
}

bool CPUproject_parallel(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	windowFOV_cpu(f, params);
	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < params->numAngles; i++)
	{
		float cos_phi = cos(params->phis[i]);
		float sin_phi = sin(params->phis[i]);

		float traj[3];
		traj[0] = cos_phi;
		traj[1] = sin_phi;
		traj[2] = 0.0;

		float* aProj = &g[i*params->numRows*params->numCols];

		omp_set_num_threads(num_thread_row);
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < params->numRows; j++)
		{
			double v = params->pixelHeight*j + params->v_0();
			float* aLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
			{
				double u = params->pixelWidth*k + params->u_0();

				float sourcePos[3];
				sourcePos[0] = -u * sin_phi;
				sourcePos[1] = u * cos_phi;
				sourcePos[2] = v;
				aLine[k] = projectLine(f, params, sourcePos, traj);
			}
		}
	}
	return true;
}

bool CPUbackproject_parallel(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	params->setToZero(f, params->numX*params->numY*params->numZ);
	const float x_width = 0.5f*params->voxelWidth;
	const float y_width = 0.5f*params->voxelWidth;
	const float z_width = 0.5f*params->voxelHeight;
    
    float rFOVsq = params->rFOV()*params->rFOV();

	const float voxelToMagnifiedDetectorPixelRatio_u = params->voxelWidth / params->pixelWidth;

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	//const int searchWidth_v = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_v);

	double v_0 = params->v_0();
	double u_0 = params->u_0();

	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int ix = 0; ix < params->numX; ix++)
		{
			const float x = ix * params->voxelWidth + params->x_0();
			float* xSlice = &f[ix * params->numY * params->numZ];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				const float cos_phi = cos(params->phis[iphi]);
				const float sin_phi = sin(params->phis[iphi]);
				float traj[3];
				traj[0] = cos_phi;
				traj[1] = sin_phi;
				//traj[2] = 0.0;

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();

					if (x * x + y * y <= rFOVsq)
					{
						float* zLine = &xSlice[iy * params->numZ];
						for (int iz = 0; iz < params->numZ; iz++)
						{
							const float z = iz * params->voxelHeight + params->z_0();

							const int u_arg_mid = int(0.5 + (y * cos_phi - x * sin_phi - u_0) / params->pixelWidth);
							const int v_arg_mid = int(0.5 + (z - v_0) / params->pixelHeight);
							const int iv = v_arg_mid;

							int iu_min = max(0, u_arg_mid - searchWidth_u);
							int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

							const float v = iv * params->pixelHeight + v_0;
							float val = 0.0f;
							for (int iu = iu_min; iu <= iu_max; iu++)
							{
								const float u = iu * params->pixelWidth + u_0;
								float pos[3];
								pos[0] = x + u * sin_phi;
								pos[1] = y - u * cos_phi;
								//pos[2] = z - v; // does not matter

								float t_max = float(1.0e16);
								float t_min = float(-1.0e16);
								if (traj[0] != 0.0f)
								{
									const float t_a = (pos[0] + x_width) / traj[0];
									const float t_b = (pos[0] - x_width) / traj[0];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[1] != 0.0f)
								{
									const float t_a = (pos[1] + y_width) / traj[1];
									const float t_b = (pos[1] - y_width) / traj[1];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}

								val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
							}
							zLine[iz] += val;
						}
					}
				}
			}
		}
	}
	else
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int iz = 0; iz < params->numZ; iz++)
		{
			const float z = iz * params->voxelHeight + params->z_0();
			float* zSlice = &f[iz * params->numY * params->numX];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				const float cos_phi = cos(params->phis[iphi]);
				const float sin_phi = sin(params->phis[iphi]);
				float traj[3];
				traj[0] = cos_phi;
				traj[1] = sin_phi;
				//traj[2] = 0.0;

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();
					float* xLine = &zSlice[iy * params->numX];
					for (int ix = 0; ix < params->numX; ix++)
					{
						const float x = ix * params->voxelWidth + params->x_0();

						if (x * x + y * y <= rFOVsq)
						{
							const int u_arg_mid = int(0.5 + (y * cos_phi - x * sin_phi - u_0) / params->pixelWidth);
							const int v_arg_mid = int(0.5 + (z - v_0) / params->pixelHeight);
							const int iv = v_arg_mid;

							int iu_min = max(0, u_arg_mid - searchWidth_u);
							int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

							const float v = iv * params->pixelHeight + v_0;
							float val = 0.0f;
							for (int iu = iu_min; iu <= iu_max; iu++)
							{
								const float u = iu * params->pixelWidth + u_0;
								float pos[3];
								pos[0] = x + u * sin_phi;
								pos[1] = y - u * cos_phi;
								//pos[2] = z - v; // does not matter

								float t_max = float(1.0e16);
								float t_min = float(-1.0e16);
								if (traj[0] != 0.0f)
								{
									const float t_a = (pos[0] + x_width) / traj[0];
									const float t_b = (pos[0] - x_width) / traj[0];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[1] != 0.0f)
								{
									const float t_a = (pos[1] + y_width) / traj[1];
									const float t_b = (pos[1] - y_width) / traj[1];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}

								val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
							}
							xLine[ix] += val;
						}
					}
				}
			}
		}
	}
	return true;
}

bool CPUproject_fan(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;

	windowFOV_cpu(f, params);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < params->numAngles; i++) {
		float cos_phi = cos(params->phis[i]);
		float sin_phi = sin(params->phis[i]);
		float* aProj = &g[i*params->numRows*params->numCols];

		omp_set_num_threads(num_thread_row);
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < params->numRows; j++) {
			double v = params->pixelHeight*j + params->v_0();
			float* aLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++) {
				double u = params->pixelWidth*k + params->u_0();

				float sourcePos[3];
				sourcePos[0] = params->sod * cos_phi;
				sourcePos[1] = params->sod * sin_phi;
				sourcePos[2] = v;

				//float traj[3];
				//float angle = params->phis[i] - atan(u);
				//traj[0] = -cos(angle);
				//traj[1] = -sin(angle);
				//traj[2] = 0;
				float traj[3];
				traj[0] = -params->sdd*cos_phi - u * sin_phi;
				traj[1] = -params->sdd*sin_phi + u * cos_phi;
				traj[2] = 0;

				aLine[k] = projectLine(f, params, sourcePos, traj);
			}
		}
	}

	return true;
}

bool CPUbackproject_fan(float* g , float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	

	return true;
}

bool CPUproject_modular(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < params->numAngles; i++)
	{
		float cos_phi = cos(params->phis[i]);
		float sin_phi = sin(params->phis[i]);

		float sourcePos[3];
		sourcePos[0] = params->sourcePositions[3 * i + 0];
		sourcePos[1] = params->sourcePositions[3 * i + 1];
		sourcePos[2] = params->sourcePositions[3 * i + 2];

		float* aProj = &g[i*params->numRows*params->numCols];

		omp_set_num_threads(num_thread_row);
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < params->numRows; j++)
		{
			double v = params->pixelHeight*j + params->v_0();
			float* aLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
			{
				double u = params->pixelWidth*k + params->u_0();

				float traj[3];
				traj[0] = params->moduleCenters[3 * i + 0] + v * params->rowVectors[3 * i + 0] + u * params->colVectors[3 * i + 0] - sourcePos[0];
				traj[1] = params->moduleCenters[3 * i + 1] + v * params->rowVectors[3 * i + 1] + u * params->colVectors[3 * i + 1] - sourcePos[1];
				traj[2] = params->moduleCenters[3 * i + 2] + v * params->rowVectors[3 * i + 2] + u * params->colVectors[3 * i + 2] - sourcePos[2];
				aLine[k] = projectLine(f, params, sourcePos, traj);
			}
		}
	}
	return true;
}

bool CPUbackproject_modular(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	params->setToZero(f, params->numX*params->numY*params->numZ);
	const float x_width = 0.5f*params->voxelWidth;
	const float y_width = 0.5f*params->voxelWidth;
	const float z_width = 0.5f*params->voxelHeight;

	double v_0 = params->v_0();
	double u_0 = params->u_0();

	const float R = sqrt(params->sourcePositions[0] * params->sourcePositions[0] + params->sourcePositions[1] * params->sourcePositions[1] + params->sourcePositions[2] * params->sourcePositions[2]);
	const float D = R + sqrt(params->moduleCenters[0] * params->moduleCenters[0] + params->moduleCenters[1] * params->moduleCenters[1] + params->moduleCenters[2] * params->moduleCenters[2]);

	const float voxelToMagnifiedDetectorPixelRatio_u = params->voxelWidth / (R / D * params->pixelWidth);
	const float voxelToMagnifiedDetectorPixelRatio_v = params->voxelHeight / (R / D * params->pixelHeight);

	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	const int searchWidth_v = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_v);

	if (params->volumeDimensionOrder == parameters::XYZ)
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int ix = 0; ix < params->numX; ix++)
		{
			const float x = ix * params->voxelWidth + params->x_0();
			float* xSlice = &f[ix * params->numY * params->numZ];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				float* sourcePosition = &(params->sourcePositions[3 * iphi]);
				float* moduleCenter = &(params->moduleCenters[3 * iphi]);
				float* v_vec = &(params->rowVectors[3 * iphi]);
				float* u_vec = &(params->colVectors[3 * iphi]);

				float c_minus_p[3];
				c_minus_p[0] = moduleCenter[0] - sourcePosition[0];
				c_minus_p[1] = moduleCenter[1] - sourcePosition[1];
				c_minus_p[2] = moduleCenter[2] - sourcePosition[2];

				float v_cross_u[3];
				v_cross_u[0] = v_vec[1] * u_vec[2] - v_vec[2] * u_vec[1];
				v_cross_u[1] = v_vec[2] * u_vec[0] - v_vec[0] * u_vec[2];
				v_cross_u[2] = v_vec[0] * u_vec[1] - v_vec[1] * u_vec[0];

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();
					float* zLine = &xSlice[iy * params->numZ];
					for (int iz = 0; iz < params->numZ; iz++)
					{
						const float z = iz * params->voxelHeight + params->z_0();

						float pos[3];
						pos[0] = x - sourcePosition[0];
						pos[1] = y - sourcePosition[1];
						pos[2] = z - sourcePosition[2];

						const float lineLength = (pos[0] * v_cross_u[0] + pos[1] * v_cross_u[1] + pos[2] * v_cross_u[2]) / (c_minus_p[0] * v_cross_u[0] + c_minus_p[1] * v_cross_u[1] + c_minus_p[2] * v_cross_u[2]);
						const float u_arg = (pos[0] * u_vec[0] + pos[1] * u_vec[1] + pos[2] * u_vec[2]) / lineLength - (c_minus_p[0] * u_vec[0] + c_minus_p[1] * u_vec[1] + c_minus_p[2] * u_vec[2]);
						const float v_arg = (pos[0] * v_vec[0] + pos[1] * v_vec[1] + pos[2] * v_vec[2]) / lineLength - (c_minus_p[0] * v_vec[0] + c_minus_p[1] * v_vec[1] + c_minus_p[2] * v_vec[2]);

						const int u_arg_mid = int(0.5 + (u_arg - u_0) / params->pixelWidth);
						const int v_arg_mid = int(0.5 + (v_arg - v_0) / params->pixelHeight);

						int iv_min = max(0, v_arg_mid - searchWidth_v);
						int iv_max = min(params->numRows - 1, v_arg_mid + searchWidth_v);
						int iu_min = max(0, u_arg_mid - searchWidth_u);
						int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

						float val = 0.0f;
						for (int iv = iv_min; iv <= iv_max; iv++)
						{
							const float v = iv * params->pixelHeight + v_0;
							for (int iu = iu_min; iu <= iu_max; iu++)
							{
								const float u = iu * params->pixelWidth + u_0;

								float traj[3];
								traj[0] = c_minus_p[0] + u * u_vec[0] + v * v_vec[0];
								traj[1] = c_minus_p[1] + u * u_vec[1] + v * v_vec[1];
								traj[2] = c_minus_p[2] + u * u_vec[2] + v * v_vec[2];
								const float trajMag_inv = 1.0 / sqrt(traj[0] * traj[0] + traj[1] * traj[1] + traj[2] * traj[2]);
								traj[0] *= trajMag_inv;
								traj[1] *= trajMag_inv;
								traj[2] *= trajMag_inv;

								float t_max = float(1.0e16);
								float t_min = float(-1.0e16);
								if (traj[0] != 0.0f)
								{
									const float t_a = (pos[0] + x_width) / traj[0];
									const float t_b = (pos[0] - x_width) / traj[0];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[1] != 0.0f)
								{
									const float t_a = (pos[1] + y_width) / traj[1];
									const float t_b = (pos[1] - y_width) / traj[1];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[2] != 0.0f)
								{
									const float t_a = (pos[2] + z_width) / traj[2];
									const float t_b = (pos[2] - z_width) / traj[2];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}

								val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
							}
						}
						zLine[iz] += val;
					}
				}
			}
		}
	}
	else
	{
		int num_threads = omp_get_num_procs();
		omp_set_num_threads(num_threads);
		#pragma omp parallel for schedule(dynamic)
		for (int iz = 0; iz < params->numZ; iz++)
		{
			const float z = iz * params->voxelHeight + params->z_0();
			float* zSlice = &f[iz * params->numY * params->numX];

			for (int iphi = 0; iphi < params->numAngles; iphi++)
			{
				float* aProj = &g[iphi * params->numCols * params->numRows];
				float* sourcePosition = &(params->sourcePositions[3 * iphi]);
				float* moduleCenter = &(params->moduleCenters[3 * iphi]);
				float* v_vec = &(params->rowVectors[3 * iphi]);
				float* u_vec = &(params->colVectors[3 * iphi]);

				float c_minus_p[3];
				c_minus_p[0] = moduleCenter[0] - sourcePosition[0];
				c_minus_p[1] = moduleCenter[1] - sourcePosition[1];
				c_minus_p[2] = moduleCenter[2] - sourcePosition[2];

				float v_cross_u[3];
				v_cross_u[0] = v_vec[1] * u_vec[2] - v_vec[2] * u_vec[1];
				v_cross_u[1] = v_vec[2] * u_vec[0] - v_vec[0] * u_vec[2];
				v_cross_u[2] = v_vec[0] * u_vec[1] - v_vec[1] * u_vec[0];

				for (int iy = 0; iy < params->numY; iy++)
				{
					const float y = iy * params->voxelWidth + params->y_0();
					float* xLine = &zSlice[iy * params->numX];
					for (int ix = 0; ix < params->numX; ix++)
					{
						const float x = ix * params->voxelWidth + params->x_0();

						float pos[3];
						pos[0] = x - sourcePosition[0];
						pos[1] = y - sourcePosition[1];
						pos[2] = z - sourcePosition[2];

						const float lineLength = (pos[0] * v_cross_u[0] + pos[1] * v_cross_u[1] + pos[2] * v_cross_u[2]) / (c_minus_p[0] * v_cross_u[0] + c_minus_p[1] * v_cross_u[1] + c_minus_p[2] * v_cross_u[2]);
						const float u_arg = (pos[0] * u_vec[0] + pos[1] * u_vec[1] + pos[2] * u_vec[2]) / lineLength - (c_minus_p[0] * u_vec[0] + c_minus_p[1] * u_vec[1] + c_minus_p[2] * u_vec[2]);
						const float v_arg = (pos[0] * v_vec[0] + pos[1] * v_vec[1] + pos[2] * v_vec[2]) / lineLength - (c_minus_p[0] * v_vec[0] + c_minus_p[1] * v_vec[1] + c_minus_p[2] * v_vec[2]);

						const int u_arg_mid = int(0.5 + (u_arg - u_0) / params->pixelWidth);
						const int v_arg_mid = int(0.5 + (v_arg - v_0) / params->pixelHeight);

						int iv_min = max(0, v_arg_mid - searchWidth_v);
						int iv_max = min(params->numRows - 1, v_arg_mid + searchWidth_v);
						int iu_min = max(0, u_arg_mid - searchWidth_u);
						int iu_max = min(params->numCols - 1, u_arg_mid + searchWidth_u);

						float val = 0.0f;
						for (int iv = iv_min; iv <= iv_max; iv++)
						{
							const float v = iv * params->pixelHeight + v_0;
							for (int iu = iu_min; iu <= iu_max; iu++)
							{
								const float u = iu * params->pixelWidth + u_0;

								float traj[3];
								traj[0] = c_minus_p[0] + u * u_vec[0] + v * v_vec[0];
								traj[1] = c_minus_p[1] + u * u_vec[1] + v * v_vec[1];
								traj[2] = c_minus_p[2] + u * u_vec[2] + v * v_vec[2];
								const float trajMag_inv = 1.0 / sqrt(traj[0] * traj[0] + traj[1] * traj[1] + traj[2] * traj[2]);
								traj[0] *= trajMag_inv;
								traj[1] *= trajMag_inv;
								traj[2] *= trajMag_inv;

								float t_max = float(1.0e16);
								float t_min = float(-1.0e16);
								if (traj[0] != 0.0f)
								{
									const float t_a = (pos[0] + x_width) / traj[0];
									const float t_b = (pos[0] - x_width) / traj[0];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[1] != 0.0f)
								{
									const float t_a = (pos[1] + y_width) / traj[1];
									const float t_b = (pos[1] - y_width) / traj[1];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}
								if (traj[2] != 0.0f)
								{
									const float t_a = (pos[2] + z_width) / traj[2];
									const float t_b = (pos[2] - z_width) / traj[2];
									t_max = min(t_max, max(t_b, t_a));
									t_min = max(t_min, min(t_b, t_a));
								}

								val += max(0.0f, t_max - t_min) * aProj[iv * params->numCols + iu];
							}
						}
						xLine[ix] += val;
					}
				}
			}
		}
	}
	return true;
}

float projectLine(float* f, parameters* params, float* pos, float* traj)
{
	double T_x = params->voxelWidth;
	double T_y = params->voxelWidth;
	double T_z = params->voxelHeight;
	int N_x = params->numX;
	int N_y = params->numY;
	int N_z = params->numZ;
	double x_0 = params->x_0();
	double y_0 = params->y_0();
	double z_0 = params->z_0();

	double val = 0.0;
	if (fabs(traj[0]) >= fabs(traj[1]) && fabs(traj[0]) >= fabs(traj[2]))
	{
		float x_bottom = x_0 - 0.5f*T_x;
		float t_bottom = (x_bottom - pos[0]) / traj[0];

		float y_low = (pos[1] + t_bottom * traj[1] - y_0) / T_y;
		float z_low = (pos[2] + t_bottom * traj[2] - z_0) / T_z;

		float y_inc = (traj[1] / traj[0]) * (T_y / T_x);
		float z_inc = (traj[2] / traj[0]) * (T_z / T_x);

		if (y_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy + 1, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy + 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy + 1, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy + 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc))*tex3D(f, iz, iy + 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else if (y_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy - 1, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy - 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy - 1, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy - 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc))*tex3D(f, iz, iy - 1, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else // y_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc))*tex3D(f, iz + 1, iy, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc))*tex3D(f, iz - 1, iy, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D(f, iz, iy, ix, params);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + y_inc * y_inc + z_inc * z_inc)*T_x;
	}
	else if (fabs(traj[1]) >= fabs(traj[0]) && fabs(traj[1]) >= fabs(traj[2]))
	{
		float y_bottom = y_0 - 0.5f*T_y;
		float t_bottom = (y_bottom - pos[1]) / traj[1];

		float x_low = (pos[0] + t_bottom * traj[0] - x_0) / T_x;
		float z_low = (pos[2] + t_bottom * traj[2] - z_0) / T_z;

		float x_inc = (traj[0] / traj[1]) * (T_x / T_y);
		float z_inc = (traj[2] / traj[1]) * (T_z / T_y);

		if (x_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix + 1, params)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix + 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix + 1, params)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix + 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix + 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix - 1, params)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D(f, iz + 1, iy, ix - 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D(f, iz, iy, ix - 1, params)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D(f, iz - 1, iy, ix - 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix - 1, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc))*tex3D(f, iz + 1, iy, ix, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc))*tex3D(f, iz - 1, iy, ix, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D(f, iz, iy, ix, params);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + z_inc * z_inc)*T_y;
	}
	else
	{
		float z_bottom = z_0 - 0.5f*T_z;
		float t_bottom = (z_bottom - pos[2]) / traj[2];

		float x_low = (pos[0] + t_bottom * traj[0] - x_0) / T_x;
		float y_low = (pos[1] + t_bottom * traj[1] - y_0) / T_y;

		float x_inc = (traj[0] / traj[2]) * (T_x / T_z);
		float y_inc = (traj[1] / traj[2]) * (T_y / T_z);

		if (x_inc > 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix + 1, params)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy + 1, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy + 1, ix + 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix + 1, params)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy - 1, ix, params)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy - 1, ix + 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else //if (y_inc == 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix + 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix - 1, params)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy + 1, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy + 1, ix - 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy, ix - 1, params)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy - 1, ix, params)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D(f, iz, iy - 1, ix - 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc))*tex3D(f, iz, iy, ix - 1, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_y; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc))*tex3D(f, iz, iy + 1, ix, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc))*tex3D(f, iz, iy, ix, params)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc))*tex3D(f, iz, iy - 1, ix, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += tex3D(f, iz, iy, ix, params);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + y_inc * y_inc)*T_z;
	}

	return float(val);
}
