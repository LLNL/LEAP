////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for cpu projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "projectors_cpu.h"

using namespace std;


bool CPUproject_cone(float* g, float* f, parameters* params)
{
	if (g == NULL || f == NULL || params == NULL)
		return false;
	if (params->isSymmetric())
		return CPUproject_AbelCone(g, f, params);
	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	params->windowFOV(f);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
	#pragma omp parallel for
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
		#pragma omp parallel for
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
	if (params->isSymmetric())
		return CPUbackproject_AbelCone(g, f, params);
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

	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
    #pragma omp parallel for
	for (int ix = 0; ix < params->numX; ix++)
	{
		const float x = ix * params->voxelWidth + params->x_0();
		float* xSlice = &f[ix*params->numY*params->numZ];

		for (int iphi = 0; iphi < params->numAngles; iphi++)
		{
			float* aProj = &g[iphi*params->numCols*params->numRows];
			const float cos_phi = cos(params->phis[iphi]);
			const float sin_phi = sin(params->phis[iphi]);

			for (int iy = 0; iy < params->numY; iy++)
			{
				const float y = iy * params->voxelWidth + params->y_0();
                
                if (x*x + y*y <= rFOVsq)
                {
                    float* zLine = &xSlice[iy*params->numZ];
                    for (int iz = 0; iz < params->numZ; iz++)
                    {
                        const float z = iz * params->voxelHeight + params->z_0();

                        float pos[3];
                        pos[0] = x - R * cos_phi;
                        pos[1] = y - R * sin_phi;
                        pos[2] = z;

                        const float v_denom = R - x * cos_phi - y * sin_phi;
                        const int u_arg_mid = int(0.5 + (D*(y*cos_phi - x * sin_phi) / v_denom - u_0) / params->pixelWidth);
                        const int v_arg_mid = int(0.5 + (D*z / v_denom - v_0) / params->pixelHeight);

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
                                const float trajLength_inv = 1.0 / sqrt(D*D + u * u + v * v);
                                traj[0] = (-D * cos_phi - u * sin_phi) * trajLength_inv;
                                traj[1] = (-D * sin_phi + u * cos_phi) * trajLength_inv;
                                traj[2] = v * trajLength_inv;

                                float t_max = 1.0e16;
                                float t_min = -1.0e16;
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

                                val += max(0.0f, t_max - t_min)*aProj[iv * params->numCols + iu];
                            }
                        }
                        zLine[iz] += val;
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
	if (params->isSymmetric())
		return CPUproject_AbelParallel(g, f, params);
	params->windowFOV(f);
	//params->setToZero(g, params->numAngles*params->numRows*params->numCols);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
    #pragma omp parallel for
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
        #pragma omp parallel for
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
	if (params->isSymmetric())
		return CPUbackproject_AbelParallel(g, f, params);
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

	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
    #pragma omp parallel for
	for (int ix = 0; ix < params->numX; ix++)
	{
		const float x = ix * params->voxelWidth + params->x_0();
		float* xSlice = &f[ix*params->numY*params->numZ];

		for (int iphi = 0; iphi < params->numAngles; iphi++)
		{
			float* aProj = &g[iphi*params->numCols*params->numRows];
			const float cos_phi = cos(params->phis[iphi]);
			const float sin_phi = sin(params->phis[iphi]);
			float traj[3];
			traj[0] = cos_phi;
			traj[1] = sin_phi;
			//traj[2] = 0.0;

			for (int iy = 0; iy < params->numY; iy++)
			{
				const float y = iy * params->voxelWidth + params->y_0();
                
                if (x*x + y*y <= rFOVsq)
                {
                    float* zLine = &xSlice[iy*params->numZ];
                    for (int iz = 0; iz < params->numZ; iz++)
                    {
                        const float z = iz * params->voxelHeight + params->z_0();

                        const int u_arg_mid = int(0.5 + (y*cos_phi - x * sin_phi - u_0) / params->pixelWidth);
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

                            float t_max = 1.0e16;
                            float t_min = -1.0e16;
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

                            val += max(0.0f, t_max - t_min)*aProj[iv * params->numCols + iu];
                        }
                        zLine[iz] += val;
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

	params->windowFOV(f);
	int num_threads = omp_get_num_procs();
	int num_thread_phi = 1;
	int num_thread_row = 1;
	if (params->numAngles > params->numRows || params->numAngles > num_threads)
		num_thread_phi = num_threads;
	else
		num_thread_row = num_threads;

	omp_set_num_threads(num_thread_phi);
    #pragma omp parallel for
	for (int i = 0; i < params->numAngles; i++) {
		float cos_phi = cos(params->phis[i]);
		float sin_phi = sin(params->phis[i]);
		float* aProj = &g[i*params->numRows*params->numCols];

		omp_set_num_threads(num_thread_row);
        #pragma omp parallel for
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
    #pragma omp parallel for
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
        #pragma omp parallel for
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

	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	#pragma omp parallel for
	for (int ix = 0; ix < params->numX; ix++)
	{
		const float x = ix * params->voxelWidth + params->x_0();
		float* xSlice = &f[ix*params->numY*params->numZ];

		for (int iphi = 0; iphi < params->numAngles; iphi++)
		{
			float* aProj = &g[iphi*params->numCols*params->numRows];
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
				float* zLine = &xSlice[iy*params->numZ];
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

							float t_max = 1.0e16;
							float t_min = -1.0e16;
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

							val += max(0.0f, t_max - t_min)*aProj[iv * params->numCols + iu];
						}
					}
					zLine[iz] += val;
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

float tex3D(float* f, int iz, int iy, int ix, parameters* params)
{
	if (0 <= ix && ix < params->numX && 0 <= iy && iy < params->numY && 0 <= iz && iz < params->numZ)
		return f[ix*params->numZ*params->numY + iy * params->numZ + iz];
	else
		return 0.0;
}


//########################################################################################################################################################################
//########################################################################################################################################################################
//### Projectors for Symmetric Objects
//########################################################################################################################################################################
//########################################################################################################################################################################
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

	int N_r = int(0.5 + 0.5*params->numY);
	double r_max = (params->numY - 1)*params->voxelWidth + y_0;

	double Rcos_sq_plus_tau_sq = params->sod*params->sod*cos_beta*cos_beta + params->tau*params->tau;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
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

			int rInd_floor, rInd_max;
			if (u_unbounded < 0.0)
			{
				rInd_floor = int(0.5 - y_0 / params->voxelWidth);
				rInd_max = N_r;
			}
			else
			{
				rInd_floor = N_r;
				//rInd_max = params->numY;
				rInd_max = N_r;
			}
			double r_min = fabs((float)(rInd_floor)*params->voxelWidth + y_0);

			double sec_sq_plus_u_sq = X * X + u * u;
			double b_ti = X * params->sod*cos_beta + u * params->tau;
			double a_ti_inv = 1.0 / sec_sq_plus_u_sq;
			double disc_ti_shift = b_ti * b_ti - sec_sq_plus_u_sq * Rcos_sq_plus_tau_sq; // new

			if (fabs(sec_sq_plus_u_sq) < 1.0e-8 || disc_ti_shift > 0.0)
				continue;

			int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / params->voxelWidth);
			double r_prev = double(rInd_min)*params->voxelWidth;
			double disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq);

			double curVal = 0.0;

			// Go back one sample and check
			if (rInd_min >= 1)
			{
				double r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
				int rInd_min_minus = max(0, min(N_r - 1, int(ceil((r_absoluteMinimum) / params->voxelWidth - 1.0))));

				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r - 1 - rInd_min_minus;
				else
					ir_shifted_or_flipped = N_r + rInd_min_minus;

				if (r_absoluteMinimum < r_max && disc_sqrt_prev > 0.0)
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
					ir_shifted_or_flipped = N_r - 1 - ir;
				else
					ir_shifted_or_flipped = N_r + ir;

				double r_next = r_prev + params->voxelWidth;
				double disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next*sec_sq_plus_u_sq);

				// Negative t interval
				// low:  (b_ti - disc_sqrt_next) * a_ti_inv
				// high: (b_ti - disc_sqrt_prev) * a_ti_inv

				// Positive t interval
				// low:  (b_ti + disc_sqrt_prev) * a_ti_inv
				// high: (b_ti + disc_sqrt_next) * a_ti_inv

				//(b_ti - disc_sqrt_next) * a_ti_inv + (b_ti - disc_sqrt_prev) * a_ti_inv
				double iz_arg_low = (b_ti - 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
				if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
				{
					int iz_arg_low_floor = int(iz_arg_low);
					double dz = iz_arg_low - double(iz_arg_low_floor);
					int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);

					curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_low_ceil]);
				}

				double iz_arg_high = (b_ti + 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
				if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
				{
					int iz_arg_high_floor = int(iz_arg_high);
					double dz = iz_arg_high - double(iz_arg_high_floor);
					int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);

					curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_high_ceil]);
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
	#pragma omp parallel for
	for (int j = 0; j < params->numY; j++)
	{
		float* zLine = &f[j*params->numZ];
		double r_unbounded = j * T_y + y_0;
		double r = fabs(r_unbounded);
		double r_inner = r - 0.5*T_y;
		double r_outer = r + 0.5*T_y;

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

	int N_r = int(0.5 + 0.5*params->numY);
	double T_r = params->voxelWidth;

	double r_max = (params->numY - 1)*params->voxelWidth + y_0;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
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

			int rInd_floor, rInd_max;
			if (u_unbounded < 0.0)
			{
				rInd_floor = int(0.5 - y_0 / params->voxelWidth);
				rInd_max = N_r;
			}
			else
			{
				rInd_floor = N_r;
				//rInd_max = params->numY;
				rInd_max = N_r;
			}
			double r_min = fabs((float)(rInd_floor)*params->voxelWidth + y_0);

			double b_ti = v * sin_beta;
			double a_ti_inv = sec_beta;
			double disc_ti_shift = -u * u;

			int rInd_min = (int)ceil(u / T_r);
			double r_prev = double(rInd_min)*T_r;
			double disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev);

			double curVal = 0.0;

			//*
			//####################################################################################
			// Go back one sample and check
			if (rInd_min >= 1)
			{
				double r_absoluteMinimum = u;
				//double disc_sqrt_check = sqrt(disc_ti_shift + r_absoluteMinimum*r_absoluteMinimum*sec_sq_plus_u_sq);
				//int rInd_min_minus = max(0, int(floor(0.5+r_absoluteMinimum/T_r)));
				int rInd_min_minus = max(0, min(N_r - 1, int(ceil(r_absoluteMinimum / T_r - 1.0))));

				int ir_shifted_or_flipped;
				if (u_unbounded < 0.0)
					ir_shifted_or_flipped = N_r - 1 - rInd_min_minus;
				else
					ir_shifted_or_flipped = N_r + rInd_min_minus;

				if (r_absoluteMinimum < r_max)
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
					ir_shifted_or_flipped = N_r - 1 - ir;
				else
					ir_shifted_or_flipped = N_r + ir;

				double r_next = r_prev + T_r;
				double disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next);

				// Negative t interval
				// low:  (b_ti - disc_sqrt_next) * a_ti_inv
				// high: (b_ti - disc_sqrt_prev) * a_ti_inv

				// Positive t interval
				// low:  (b_ti + disc_sqrt_prev) * a_ti_inv
				// high: (b_ti + disc_sqrt_next) * a_ti_inv

				//(b_ti - disc_sqrt_next) * a_ti_inv + (b_ti - disc_sqrt_prev) * a_ti_inv
				double iz_arg_low = (b_ti - 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
				if (0.0 <= iz_arg_low && iz_arg_low <= params->numZ - 1)
				{
					int iz_arg_low_floor = int(iz_arg_low);
					double dz = iz_arg_low - double(iz_arg_low_floor);
					int iz_arg_low_ceil = min(iz_arg_low_floor + 1, params->numZ - 1);

					curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_low_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_low_ceil]);
				}

				double iz_arg_high = (b_ti + 0.5*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift;
				if (0.0 <= iz_arg_high && iz_arg_high <= params->numZ - 1)
				{
					int iz_arg_high_floor = int(iz_arg_high);
					double dz = iz_arg_high - double(iz_arg_high_floor);
					int iz_arg_high_ceil = min(iz_arg_high_floor + 1, params->numZ - 1);

					curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*((1.0 - dz)*f[ir_shifted_or_flipped*params->numZ + iz_arg_high_floor] + dz * f[ir_shifted_or_flipped*params->numZ + iz_arg_high_ceil]);
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
	#pragma omp parallel for
	for (int j = 0; j < params->numY; j++)
	{
		float* zLine = &f[j*params->numZ];
		double r_unbounded = j * T_y + y_0;
		double r = fabs(r_unbounded);
		double r_inner = r - 0.5*T_y;
		double r_outer = r + 0.5*T_y;

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


//########################################################################################################################################################################
//########################################################################################################################################################################
//### Separable Footprint (SF) Projectors
//########################################################################################################################################################################
//########################################################################################################################################################################
bool CPUproject_SF_parallel(float* g, float* f, parameters* params)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (params->isSymmetric())
        return CPUproject_AbelParallel(g, f, params);
	params->setToZero(g, params->numAngles*params->numRows*params->numCols);
    double u_0 = params->u_0();
    
    float rFOVsq = params->rFOV()*params->rFOV();

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[iphi*params->numCols*params->numRows];
        const float cos_phi = cos(params->phis[iphi]);
        const float sin_phi = sin(params->phis[iphi]);
        
        float cos_phi_over_T_s = cos_phi / params->pixelWidth;
        float sin_phi_over_T_s = sin_phi / params->pixelWidth;
        float T_y_mult_cos_phi_over_T_s = params->voxelWidth*cos_phi_over_T_s;
        float N_s_minus_one = float(params->numCols-1);

        float l_phi = 1.0 / max(fabs(cos_phi), fabs(sin_phi));

        float maxWeight = params->voxelWidth*params->voxelWidth / params->pixelWidth; // cm
        float A = 0.5*(1.0 - params->voxelWidth/(params->pixelWidth*l_phi));
        float one_minus_A = 1.0 - A;
        float T_x_mult_l_phi = params->voxelWidth * l_phi;

        float s_arg, ds, ds_conj;
        int s_low, s_high;
        
        float s_0_over_T_s = u_0 / params->pixelWidth;
        for (int ix = 0; ix < params->numX; ix++)
        {
            const float x = ix * params->voxelWidth + params->x_0();
            float* xSlice = &f[ix*params->numY*params->numZ];
        
            s_arg = params->y_0()*cos_phi_over_T_s - x*sin_phi_over_T_s - s_0_over_T_s;
            for (int iy = 0; iy < params->numY; iy++)
            {
                const float y = iy * params->voxelWidth + params->y_0();
                if (x*x + y*y <= rFOVsq && s_arg > -1.0 && s_arg < double(params->numCols))
                {
                    float* zLine = &xSlice[iy*params->numZ];
                    
                    s_low = int(s_arg);
                    if (s_arg < 0.0 || s_arg > N_s_minus_one)
                    {
                        ds_conj = maxWeight;
                        for (int k = 0; k < params->numZ; k++)
                            aProj[k*params->numCols+s_low] += ds_conj*zLine[k];
                    }
                    else
                    {
                        s_high = s_low+1;
                        ds = s_arg - double(s_low);
                        if (A > ds)
                        {
                            ds_conj = maxWeight;
                            for (int k = 0; k < params->numZ; k++)
                                aProj[k*params->numCols+s_low] += ds_conj*zLine[k];
                        }
                        else if (ds > one_minus_A)
                        {
                            ds = maxWeight;
                            for (int k = 0; k < params->numZ; k++)
                                aProj[k*params->numCols+s_high] += ds*zLine[k];
                        }
                        else
                        {
                            ds_conj = T_x_mult_l_phi*(one_minus_A - ds);
                            ds = maxWeight - ds_conj;
                            for (int k = 0; k < params->numZ; k++)
                            {
                                aProj[k*params->numCols+s_low] += ds_conj*zLine[k];
                                aProj[k*params->numCols+s_high] += ds*zLine[k];
                            }
                        }
                    }
                }
                s_arg += T_y_mult_cos_phi_over_T_s;
            }
        }
    }
    return true;
}

bool CPUbackproject_SF_parallel(float* g , float* f, parameters* params)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (params->isSymmetric())
        return CPUbackproject_AbelParallel(g, f, params);
	params->setToZero(f, params->numX*params->numY*params->numZ);
    float u_0 = params->u_0();
    
    float rFOVsq = params->rFOV()*params->rFOV();

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int ix = 0; ix < params->numX; ix++)
    {
        const float x = ix * params->voxelWidth + params->x_0();
        float* xSlice = &f[ix*params->numY*params->numZ];

        for (int iphi = 0; iphi < params->numAngles; iphi++)
        {
            float* aProj = &g[iphi*params->numCols*params->numRows];
            const float cos_phi = cos(params->phis[iphi]);
            const float sin_phi = sin(params->phis[iphi]);
            
            float cos_phi_over_T_s = cos_phi / params->pixelWidth;
            float sin_phi_over_T_s = sin_phi / params->pixelWidth;
            float T_y_mult_cos_phi_over_T_s = params->voxelWidth*cos_phi_over_T_s;
            float N_s_minus_one = float(params->numCols-1);

            float l_phi = 1.0 / max(fabs(cos_phi), fabs(sin_phi));

            float maxWeight = params->voxelWidth*params->voxelWidth / params->pixelWidth; // cm
            float A = 0.5*(1.0 - params->voxelWidth/(params->pixelWidth*l_phi));
            float one_minus_A = 1.0 - A;
            float T_x_mult_l_phi = params->voxelWidth * l_phi;

            float s_arg, ds, ds_conj;
            int s_low, s_high;
            
            float s_0_over_T_s = u_0 / params->pixelWidth;
            s_arg = params->y_0()*cos_phi_over_T_s - x*sin_phi_over_T_s - s_0_over_T_s;
            for (int iy = 0; iy < params->numY; iy++)
            {
                const float y = iy * params->voxelWidth + params->y_0();
                if (x*x + y*y <= rFOVsq && s_arg > -1.0 && s_arg < double(params->numCols))
                {
                    float* zLine = &xSlice[iy*params->numZ];
                    
                    s_low = int(s_arg);
                    if (s_arg < 0.0 || s_arg > N_s_minus_one)
                    {
                        ds_conj = maxWeight;
                        for (int k = 0; k < params->numZ; k++)
                            zLine[k] += ds_conj*aProj[k*params->numCols+s_low];
                    }
                    else
                    {
                        s_high = s_low+1;
                        ds = s_arg - double(s_low);
                        if (A > ds)
                        {
                            ds_conj = maxWeight;
                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds_conj*aProj[k*params->numCols+s_low];
                        }
                        else if (ds > one_minus_A)
                        {
                            ds = maxWeight;
                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds*aProj[k*params->numCols+s_high];
                        }
                        else
                        {
                            ds_conj = T_x_mult_l_phi*(one_minus_A - ds);
                            ds = maxWeight - ds_conj;
                            for (int k = 0; k < params->numZ; k++)
                                zLine[k] += ds_conj*aProj[k*params->numCols+s_low] + ds*aProj[k*params->numCols+s_high];
                        }
                    }
                }
                s_arg += T_y_mult_cos_phi_over_T_s;
            }
        }
    }
    return true;
}

bool CPUproject_SF_cone(float* g, float* f, parameters* params)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (params->isSymmetric())
        return CPUproject_AbelCone(g, f, params);
	params->setToZero(g, params->numAngles*params->numRows*params->numCols);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        float* aProj = &g[iphi*params->numCols*params->numRows];
        
        for (int ix = 0; ix < params->numX; ix++)
        {
            float* xSlice = &f[ix*params->numY*params->numZ];
            CPUproject_SF_cone_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
	applyInversePolarWeight(g, params);
    return true;
}

bool CPUbackproject_SF_cone(float* g, float* f, parameters* params)
{
    if (g == NULL || f == NULL || params == NULL)
        return false;
    if (params->isSymmetric())
        return CPUbackproject_AbelCone(g, f, params);
	applyInversePolarWeight(g, params);
	params->setToZero(f, params->numX*params->numY*params->numZ);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int ix = 0; ix < params->numX; ix++)
    {
        float* xSlice = &f[ix*params->numY*params->numZ];
        for (int iphi = 0; iphi < params->numAngles; iphi++)
        {
            float* aProj = &g[iphi*params->numCols*params->numRows];
            CPUbackproject_SF_cone_kernel(aProj, xSlice, params, ix, iphi);
        }
    }
	applyPolarWeight(g, params); // do this so projection data is not altered by backprojection operation
    return true;
}

bool CPUproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float u_0 = params->u_0() / params->sdd;
    float v_0 = params->v_0() / params->sdd;
    float T_u = params->pixelWidth / params->sdd;
    float T_v = params->pixelHeight / params->sdd;
    
    const float cos_phi = cos(params->phis[iphi]);
    const float sin_phi = sin(params->phis[iphi]);
    const float x = params->voxelWidth*ix + params->x_0();
    
	float A_x, B_x, A_y, B_y;
	float T_x_over_2 = params->voxelWidth / 2.0;
    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }
    
    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;
    
    float v_denom;//, v_phi_x;
    float pitch_mult_phi_plus_startZ = 0.0;

    float t_neg, t_pos, t_1, t_2;
    int t_ind_min, t_ind_max;
    float A_ind;
    float T_z_over_2 = params->voxelHeight/2.0;

    float T_z_over_2T_v_v_denom;
    
    float rFOVsq = params->rFOV()*params->rFOV();

	int vBounds[2];
	vBounds[0] = 0;
	vBounds[1] = params->numRows - 1;
	dist_from_source_components[0] = fabs(params->sod*cos_phi + params->tau*sin_phi - x);
    for (int iy = 0; iy < params->numY; iy++)
    {
        const float y = iy * params->voxelWidth + params->y_0();
        if (x*x + y*y <= rFOVsq)
        {
            float* zLine = &xSlice[iy*params->numZ];
            x_dot_theta = x*cos_phi + y*sin_phi;
            x_dot_theta_perp = -sin_phi*x + cos_phi*y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod*sin_phi - params->tau*cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0]*dist_from_source_components[0] + dist_from_source_components[1]*dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);
            
            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg*cos_phi-sin_phi) > fabs(u_arg*sin_phi+cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low+0.5); // first detector index
            ind_last = int(tauInd_high+0.5); // last detector index
            
            if (tauInd_low >= double(params->numCols)-0.5 || tauInd_high <= -0.5)
                break;
            
            ind_diff = ind_last - ind_first;

            T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);
            
            double v_phi_x_step = 2.0*T_z_over_2T_v_v_denom;
            t_neg = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;

            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;

                for (int i = 0; i < params->numZ; i++)
                {
                    if (zLine[i] != 0.0)
                    {
                        t_ind_min = int(t_neg + 0.5);
                        t_ind_max = int(t_pos + 0.5);
                        if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                        if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                        for (int L=t_ind_min; L <= t_ind_max; L++)
                        {
                            // [t_1   t_2  ]
                            // [t_neg t_pos]
                            t_1 = double(L) - 0.5;
                            t_2 = t_1 + 1.0;
                            if (t_pos < t_2) {t_2 = t_pos;}
                            if (t_neg > t_1) {t_1 = t_neg;}
                            if (t_2 > t_1)
                            {
                                A_ind = (t_2 - t_1) * theWeight;
                                aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                            }
                        } // L
                    }
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                } // z
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                    }
                                } // L
                            }
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else
                    {
                        // do first
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                    }
                                } // L
                            }
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        }
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                        aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                        aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                    else
                    {
                        // do first only
                        for (int i = 0; i < params->numZ; i++)
                        {
                            if (zLine[i] != 0.0)
                            {
                                t_ind_min = int(t_neg + 0.5);
                                t_ind_max = int(t_pos + 0.5);
                                if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                                if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                                for (int L=t_ind_min; L <= t_ind_max; L++)
                                {
                                    t_1 = double(L) - 0.5;
                                    t_2 = t_1 + 1.0;
                                    if (t_pos < t_2) {t_2 = t_pos;}
                                    if (t_neg > t_1) {t_1 = t_neg;}
                                    if (t_2 > t_1)
                                    {
                                        A_ind = (t_2 - t_1) * theWeight;
                                        aProj[L*params->numCols+ind_first] += zLine[i] * firstWeight * A_ind;
                                    }
                                } // L
                            } // support check
                            t_neg = t_pos;
                            t_pos += v_phi_x_step;
                        } // z
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_middle] += zLine[i] * middleWeight * A_ind;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        } // support check
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
                else
                {
                    // do last only
                    for (int i = 0; i < params->numZ; i++)
                    {
                        if (zLine[i] != 0.0)
                        {
                            t_ind_min = int(t_neg + 0.5);
                            t_ind_max = int(t_pos + 0.5);
                            if (t_ind_min < vBounds[0]) {t_ind_min = vBounds[0];}
                            if (t_ind_max > vBounds[1]) {t_ind_max = vBounds[1];}
                            for (int L=t_ind_min; L <= t_ind_max; L++)
                            {
                                t_1 = double(L) - 0.5;
                                t_2 = t_1 + 1.0;
                                if (t_pos < t_2) {t_2 = t_pos;}
                                if (t_neg > t_1) {t_1 = t_neg;}
                                if (t_2 > t_1)
                                {
                                    A_ind = (t_2 - t_1) * theWeight;
                                    aProj[L*params->numCols+ind_last] += zLine[i] * lastWeight * A_ind;
                                }
                            } // L
                        } // support check
                        t_neg = t_pos;
                        t_pos += v_phi_x_step;
                    } // z
                }
            } // number of contributions (1, 2, or 3)
        }
    }
    return true;
}

bool CPUbackproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi)
{
    float u_0 = params->u_0() / params->sdd;
    float v_0 = params->v_0() / params->sdd;
    float T_u = params->pixelWidth / params->sdd;
    float T_v = params->pixelHeight / params->sdd;
    
    float rFOVsq = params->rFOV()*params->rFOV();
    
    float tau[2];
    int ind_first, ind_middle, ind_last, ind_diff;
    float firstWeight, middleWeight, lastWeight;
    float tauInd_low, tauInd_high;

    float sampleConstant = params->voxelWidth;
    float theWeight = sampleConstant;

    float dist_from_source;
    float dist_from_source_components[2];
    float l_phi = 1.0;

    float R_minus_x_dot_theta, u_arg;

    float x_dot_theta, x_dot_theta_perp;
    
    float phi = params->phis[iphi];
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float v_denom, v_phi_x;
    float pitch_mult_phi_plus_startZ = 0.0;

    float t_neg, t_pos; //, t_1, t_2;
    //int t_ind_min, t_ind_max;
    float T_z_over_2 = params->voxelHeight/2.0;

    float T_x_over_2 = params->voxelWidth / 2.0;
    float A_x, B_x;
    float A_y, B_y;
    
    if (sin_phi < 0.0)
    {
        A_x = -sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = -cos_phi * T_x_over_2;
    }
    else
    {
        A_x = sin_phi * T_x_over_2; // fabs(sin_phi)*T_x/2
        B_x = cos_phi * T_x_over_2;
    }
    if (cos_phi < 0.0)
    {
        A_y = -cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = sin_phi * T_x_over_2;
    }
    else
    {
        A_y = cos_phi * T_x_over_2; // fabs(cos_phi)*T_x/2
        B_y = -sin_phi * T_x_over_2;
    }

    int i, L;

    double T_z_over_2T_v_v_denom;

    int v_arg_bounds[2];
	v_arg_bounds[0] = 0;
	v_arg_bounds[1] = params->numRows - 1;
    float* interpolatedLineOut = (float*) calloc(size_t(params->numRows), sizeof(float));

    const float x = ix*params->voxelWidth + params->x_0();
    dist_from_source_components[0] = fabs(params->sod*cos_phi + params->tau*sin_phi - x);
    for (int iy=0; iy < params->numY; iy++)
    {
        const float y = iy*params->voxelWidth + params->y_0();
        float* zLine = &xSlice[iy*params->numZ];
        if (x*x + y*y <= rFOVsq)
        {
            x_dot_theta = x*cos_phi + y*sin_phi;
            x_dot_theta_perp = -sin_phi*x + cos_phi*y + params->tau; // note: shifted by tau
            R_minus_x_dot_theta = params->sod - x_dot_theta;

            dist_from_source_components[1] = fabs(params->sod*sin_phi - params->tau*cos_phi - y);
            dist_from_source = sqrt(dist_from_source_components[0]*dist_from_source_components[0] + dist_from_source_components[1]*dist_from_source_components[1]);
            l_phi = dist_from_source / max(dist_from_source_components[0], dist_from_source_components[1]);

            u_arg = x_dot_theta_perp / R_minus_x_dot_theta;

            if (fabs(u_arg*cos_phi-sin_phi) > fabs(u_arg*sin_phi+cos_phi))
            {
                tau[0] = (x_dot_theta_perp - A_x) / (R_minus_x_dot_theta - B_x);
                tau[1] = (x_dot_theta_perp + A_x) / (R_minus_x_dot_theta + B_x);
            }
            else
            {
                tau[0] = (x_dot_theta_perp - A_y) / (R_minus_x_dot_theta - B_y);
                tau[1] = (x_dot_theta_perp + A_y) / (R_minus_x_dot_theta + B_y);
            }

            v_denom = R_minus_x_dot_theta;

            theWeight = sampleConstant * l_phi;

            tauInd_low = (tau[0] - u_0) / T_u;
            tauInd_high = (tau[1] - u_0) / T_u;

            ind_first = int(tauInd_low+0.5); // first detector index
            ind_last = int(tauInd_high+0.5); // last detector index
            
            //if (ind_first > params->numCols-1 || ind_last < 0)
            //    break;
            if (tauInd_low >= double(params->numCols)-0.5 || tauInd_high <= -0.5)
                break;
            
            ind_diff = ind_last - ind_first;

            T_z_over_2T_v_v_denom = T_z_over_2 / (T_v * v_denom);

            //g->v_arg_bounds(f, pitch_mult_phi_plus_startZ, v_denom, &v_arg_bounds[0]);
            //v_arg_bounds[0] = max(v_arg_bounds[0], vBounds[0]);
            //v_arg_bounds[1] = min(v_arg_bounds[1], vBounds[1]);

            
            if (ind_diff == 0)
            {
                // distributed over 1 bin
                firstWeight = tauInd_high - tauInd_low;
                
                for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                    interpolatedLineOut[i] = aProj[i*params->numCols + ind_first] * firstWeight * theWeight;
            }
            else if (ind_diff == 1)
            {
                // distributed over 2 bins
                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= params->numCols-1
                    {
                        // do first and last
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i*params->numCols+ind_first] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                    }
                    else
                    {
                        // do first
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i*params->numCols+ind_first] * theWeight;
                    }
                }
                else //if (ind_last < params->numCols)
                {
                    // do last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i*params->numCols+ind_last] * theWeight;
                }
            }
            else //if (ind_diff == 2)
            {
                // distributed over 3 bins
                ind_middle = ind_first + 1;

                firstWeight = double(ind_first)+0.5 - tauInd_low; // double(ind_first) - tauInd_low + 0.5;
                lastWeight = tauInd_high - (double(ind_last)-0.5); // tauInd_high - double(ind_last) + 0.5;
                middleWeight = 1.0;

                if (ind_first >= 0)
                {
                    if (ind_last < params->numCols) // ind_last <= N_lateral-1
                    {
                        // do all 3
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight*aProj[i*params->numCols+ind_first] + middleWeight*aProj[i*params->numCols+ind_middle] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                    }
                    else if (ind_last == params->numCols) // ind_middle == N_lateral-1
                    {
                        // do first and middle
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = (firstWeight * aProj[i*params->numCols+ind_first] + middleWeight*aProj[i*params->numCols+ind_middle]) * theWeight;
                    }
                    else
                    {
                        // do first only
                        for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                            interpolatedLineOut[i] = firstWeight * aProj[i*params->numCols+ind_first] * theWeight;
                    }
                }
                else if (ind_middle == 0)
                {
                    // do middle and last
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = (middleWeight * aProj[i*params->numCols+ind_middle] + lastWeight*aProj[i*params->numCols+ind_last]) * theWeight;
                }
                else
                {
                    // do last only
                    for (i = v_arg_bounds[0]; i <= v_arg_bounds[1]; i++)
                        interpolatedLineOut[i] = lastWeight * aProj[i*params->numCols+ind_last] * theWeight;
                }
            } // number of contributions (1, 2, or 3)

            v_phi_x = ((params->z_0() - pitch_mult_phi_plus_startZ) / v_denom - v_0) / T_v;
            double v_phi_x_step = 2.0*T_z_over_2T_v_v_denom;
            t_neg = v_phi_x - T_z_over_2T_v_v_denom;
            t_pos = t_neg + v_phi_x_step;

            i = 0;
            L = int(ceil(t_neg - 0.5)); // enforce: t_neg <= L+0.5
            if (L < v_arg_bounds[0])
                L = v_arg_bounds[0];
            double L_plus_half = double(L) + 0.5;
            double previousBoundary = double(L) - 0.5;

            // Extrapolation off bottom of detector
            while (t_pos < previousBoundary && i < params->numZ)
            {
                //if ((supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //    zLine[i] += interpolatedLineOut[L]*v_phi_x_step;
                t_neg = t_pos;
                t_pos += v_phi_x_step;
                i += 1;
            }
            if (t_neg < previousBoundary)
            {
                // known: t_neg < previousBoundary <= t_pos
                //if (i < params->numZ && (supportMapArray == NULL || supportMapArray[i] == 1) && doExtrapolation == true)
                //if (i < params->numZ)
                //    zLine[i] += interpolatedLineOut[L]*(previousBoundary - t_neg);
            }
            else
                previousBoundary = t_neg;

            while (i < params->numZ && L < params->numRows)
            {
                if (t_pos <= L_plus_half)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(t_pos - previousBoundary);
                    previousBoundary = t_pos;
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                    i += 1;
                }
                else // L_plus_half < t_pos
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(L_plus_half - previousBoundary);
                    previousBoundary = L_plus_half;
                    L_plus_half += 1.0;
                    L += 1;
                }
            }

            // now either: i == params->numZ || L == g->N_v
            // Extrapolation off top of detector
			/*
            if (i < params->numZ && doExtrapolation == true)
            {
                L = params->numRows-1;
                L_plus_half = double(L) + 0.5;
                if (t_neg < L_plus_half)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*(t_pos - L_plus_half);
                    t_neg = t_pos;
                    t_pos += v_phi_x_step;
                    i += 1;
                }
                // now: L_plus_half < t_neg < t_pos
                while (i < params->numZ)
                {
                    //if (supportMapArray == NULL || supportMapArray[i] == 1)
                        zLine[i] += interpolatedLineOut[L]*v_phi_x_step;
                    i += 1;
                }
            }
			//*/
        } // ROI check
    } // y
    free(interpolatedLineOut);

    return true;
}

bool applyPolarWeight(float* g, parameters* params)
{
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for private (i)
	for (int i = 0; i < params->numAngles; i++)
	{
		float* aProj = &g[i*params->numRows*params->numCols];
		for (int j = 0; j < params->numRows; j++)
		{
			float v = (j*params->pixelHeight + params->v_0()) / params->sdd;
			float temp = 1.0 / sqrt(1.0 + v*v);
			float* zLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
				zLine[k] *= temp;
		}
	}
	return true;
}

bool applyInversePolarWeight(float* g, parameters* params)
{
	omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for private (i)
	for (int i = 0; i < params->numAngles; i++)
	{
		float* aProj = &g[i*params->numRows*params->numCols];
		for (int j = 0; j < params->numRows; j++)
		{
			float v = (j*params->pixelHeight + params->v_0()) / params->sdd;
			float temp = sqrt(1.0 + v*v);
			float* zLine = &aProj[j*params->numCols];
			for (int k = 0; k < params->numCols; k++)
				zLine[k] *= temp;
		}
	}
	return true;
}