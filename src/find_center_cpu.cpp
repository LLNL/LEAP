////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based routines to find the center detector pixel
////////////////////////////////////////////////////////////////////////////////
#include <omp.h>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include "find_center_cpu.h"

using namespace std;

bool findCenter_cpu(float* g, parameters* params, int iRow)
{
	if (params->offsetScan == true)
		printf("Warning: findCenter may not work for offsetScan\n");
    if (params->geometry == parameters::PARALLEL)
        return findCenter_parallel_cpu(g, params, iRow);
    else if (params->geometry == parameters::FAN)
        return findCenter_fan_cpu(g, params, iRow);
	else if (params->geometry == parameters::CONE)
	{
		if (params->helicalPitch != 0.0)
			printf("Warning: findCenter will likely not work for helical data\n");
		return findCenter_cone_cpu(g, params, iRow);
	}
    else
    {
        printf("Error: currently findCenter only works for parallel-, fan-, or cone-beam data\n");
        return false;
    }
}

bool findCenter_fan_cpu(float* g, parameters* params, int iRow)
{
	return findCenter_cone_cpu(g, params, iRow);
}

bool findCenter_parallel_cpu(float* g, parameters* params, int iRow)
{
    if (iRow < 0 || iRow > params->numRows - 1)
        iRow = max(0, min(params->numRows-1, int(floor(0.5 + params->centerRow))));

	int rowStart = 0;
	int rowEnd = params->numRows - 1;

	if (params->angularRange + 2.0*params->T_phi() * 180.0 / PI < 180.0)
	{
		printf("Error: angular range insufficient to estimate centerCol\n");
		return false;
	}
	else if (params->angularRange > 225.0)
	{
		rowStart = iRow;
		rowEnd = iRow;
	}

	int conj_ind = 0;
	if (params->T_phi() > 0.0)
		conj_ind = int(floor(0.5 + params->phi_inv(params->phis[0] + PI)));
	else
		conj_ind = int(floor(0.5 + params->phi_inv(params->phis[0] - PI)));

	int centerCol_low, centerCol_high;

	setDefaultRange_centerCol(params->numCols, centerCol_low, centerCol_high);

	double* shiftCosts = (double*)malloc(sizeof(double) * params->numCols);

	float phi_0 = params->phis[0];
	float phi_end = params->phis[params->numAngles - 1];
	float phi_min = min(phi_0, phi_end);
	float phi_max = max(phi_0, phi_end);

	float u_0 = params->u(0);
	float u_end = params->u(params->numCols - 1);

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int n = centerCol_low; n <= centerCol_high; n++)
	{
		shiftCosts[n] = 0.0;
		//double denom = 0.0;

		float u_0 = -(float(n) + params->colShiftFromFilter) * params->pixelWidth;

		double num = 0.0;
		for (int i = 0; i <= conj_ind - 1; i++)
		{
			float phi = params->phis[i];
			float* projA = &g[uint64(i) * uint64(params->numRows * params->numCols)];

			int i_conj = i + conj_ind;

			if (i_conj < params->numAngles || i == 0)
			{
				i_conj = min(i_conj, params->numAngles - 1);

				float* projB = &g[uint64(i_conj) * uint64(params->numRows * params->numCols)];
				for (int j = rowStart; j <= rowEnd; j++)
				{
					float* lineA = &projA[j * params->numCols];
					float* lineB = &projB[j * params->numCols];
					for (int k = 0; k < params->numCols; k++)
					{
						//float u = params->u(k);
						float u = k * params->pixelWidth + u_0;

						float u_conj = -u;
						if (u_0 <= u_conj && u_conj <= u_end)
						{
							int u_conj_ind = int(0.5 + (u_conj - u_0) / params->pixelWidth);

							float val = lineA[k];
							float val_conj = lineB[u_conj_ind];
							//if (val != 0.0 || val_conj != 0.0)
							//	printf("%f and %f\n", val, val_conj);

							num += (val - val_conj) * (val - val_conj);
						}
					}
				}
			}
		}
		//printf("%f ", num);
		shiftCosts[n] = num;
	}

	for (int i = centerCol_low; i <= centerCol_high; i++)
	{
		//printf("%f\n", shiftCosts[i]);
		if (shiftCosts[i] == 0.0)
			shiftCosts[i] = 1e12;
	}

	params->centerCol = findMinimum(shiftCosts, centerCol_low, centerCol_high);
	free(shiftCosts);

	return true;
}

bool findCenter_cone_cpu(float* g, parameters* params, int iRow)
{
    if (iRow < 0 || iRow > params->numRows - 1)
        iRow = max(0, min(params->numRows - 1, int(floor(0.5 + params->centerRow))));
	int rowStart = 0;
	int rowEnd = params->numRows - 1;

	rowStart = iRow;
	rowEnd = iRow;

	int conj_ind = 0;
	if (params->T_phi() > 0.0)
		conj_ind = int(floor(0.5 + params->phi_inv(params->phis[0] + PI)));
	else
		conj_ind = int(floor(0.5 + params->phi_inv(params->phis[0] - PI)));

	int centerCol_low, centerCol_high;

	setDefaultRange_centerCol(params->numCols, centerCol_low, centerCol_high);
	
	double* shiftCosts = (double*)malloc(sizeof(double) * params->numCols);

	float A = 2.0 * params->tau * params->sod / (params->sod * params->sod - params->tau * params->tau);
	float atanA = atan(A);
	bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
	params->normalizeConeAndFanCoordinateFunctions = true;

	float phi_0 = params->phis[0];
	float phi_end = params->phis[params->numAngles - 1];
	float phi_min = min(phi_0, phi_end);
	float phi_max = max(phi_0, phi_end);

	float u_0 = params->u(0);
	float u_end = params->u(params->numCols-1);

	float atanTu = atan(params->pixelWidth / params->sdd);

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int n = centerCol_low; n <= centerCol_high; n++)
	{
		shiftCosts[n] = 0.0;
		//double denom = 0.0;

		float u_0 = -(float(n) + params->colShiftFromFilter) * params->pixelWidth;
		if (params->detectorType == parameters::CURVED)
			u_0 = -(float(n) + params->colShiftFromFilter) * atanTu;

		double num = 0.0;
		for (int i = 0; i <= conj_ind - 1; i++)
		{
			float phi = params->phis[i];
			float* projA = &g[uint64(i) * uint64(params->numRows * params->numCols)];
			for (int j = rowStart; j <= rowEnd; j++)
			{
				float* lineA = &projA[j * params->numCols];
				for (int k = 0; k < params->numCols; k++)
				{
					//float u = params->u(k);
					float u = (k * params->pixelWidth + u_0) / params->sdd;
					if (params->detectorType == parameters::CURVED)
						u = atan((k * atanTu + u_0));

					float u_conj = (-u + A) / (1.0 + u*A);
					float phi_conj = phi - 2.0 * atan(u) + atanA + PI;
					if (phi_conj > phi_max)
						phi_conj -= 2.0 * PI;
					if (phi_min <= phi_conj && phi_conj <= phi_max && u_0 <= u_conj && u_conj <= u_end)
					{
						int phi_conj_ind = int(0.5 + params->phi_inv(phi_conj));

						int u_conj_ind = int(0.5 + ((u_conj * params->sdd) - u_0) / params->pixelWidth);
						if (params->detectorType == parameters::CURVED)
							u_conj_ind = int(0.5 + (tan(u_conj) - u_0) / atanTu);

						float val = lineA[k];
						float val_conj = g[uint64(phi_conj_ind) * uint64(params->numRows * params->numCols) + uint64(j * params->numCols + u_conj_ind)];
						//if (val != 0.0 || val_conj != 0.0)
						//	printf("%f and %f\n", val, val_conj);

						num += (val - val_conj) * (val - val_conj);
					}
				}
			}
		}
		//printf("%f ", num);
		shiftCosts[n] = num;
	}

	for (int i = centerCol_low; i <= centerCol_high; i++)
	{
		//printf("%f\n", shiftCosts[i]);
		if (shiftCosts[i] == 0.0)
			shiftCosts[i] = 1e12;
	}

	params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

	params->centerCol = findMinimum(shiftCosts, centerCol_low, centerCol_high);
	free(shiftCosts);

    return true;
}

float findMinimum(double* costVec, int startInd, int endInd, bool findOnlyLocalMin/* = true*/)
{
	int localMin_ind = -1;
	double localMin_value = 1e12;

	int minCost_ind = startInd;
	double minCost = costVec[startInd];
	for (int i = startInd + 1; i <= endInd; i++)
	{
		if (costVec[i] < minCost)
		{
			minCost = costVec[i];
			minCost_ind = i;
		}
		if (i < endInd && costVec[i] < costVec[i - 1] && costVec[i] < costVec[i + 1])
		{
			if (costVec[i] < localMin_value)
			{
				localMin_value = costVec[i];
				localMin_ind = i;
			}
		}
	}

	if ((minCost_ind == startInd || minCost_ind == endInd) && localMin_ind != -1)
	{
		// min cost is at the end of estimation region and these does exist a local minimum
		// so min cost is likely just an edge effect and thus the local minimum should be used instead
		minCost_ind = localMin_ind;
	}

	float retVal = float(minCost_ind);
	if (minCost_ind > startInd && minCost_ind < endInd && costVec[minCost_ind - 1] != 1.0e12 && costVec[minCost_ind + 1] != 1.0e12 && costVec[minCost_ind - 1] > 0.0 && costVec[minCost_ind + 1] > 0.0)
		retVal += 0.5 * (costVec[minCost_ind - 1] - costVec[minCost_ind + 1]) / (costVec[minCost_ind - 1] + costVec[minCost_ind + 1] - 2.0 * costVec[minCost_ind]);
	return retVal;
}

bool setDefaultRange_centerCol(int numCols, int& centerCol_low, int& centerCol_high)
{
	double c = 0.23;
	int N_trim = 50;
	if (numCols < 200)
		N_trim = 5;

	centerCol_low = c * numCols;
	centerCol_high = numCols - c * numCols;

	/*
	if (left_center_right == -1)
		centerCol_low = N_trim; // assumes left-side offsetScan
	else if (left_center_right == 1)
		centerCol_low = numCols / 2 + 1 + N_trim; // assumes left-side offsetScan
	if (left_center_right == -1)
		centerCol_high = numCols / 2 - 1 - N_trim; // assumes left-side halfscan
	else if (left_center_right == 1)
		centerCol_high = numCols - 1 - N_trim;
	//*/

	centerCol_low = max(0, min(numCols - 1, centerCol_low));
	centerCol_high = max(0, min(numCols - 1, centerCol_high));
	if (centerCol_low > centerCol_high)
	{
		centerCol_low = 5;
		centerCol_high = numCols - 1 - 5;
	}
	return true;
}
