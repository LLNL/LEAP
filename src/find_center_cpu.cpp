////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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
#include <algorithm>
#include "find_center_cpu.h"
#include "log.h"
#include "rebin.h"

using namespace std;
#define USE_MEAN_DIFFERENCE_METRIC

bool getConjugateProjections(float* g, parameters* params, float*& proj_A, float*& proj_B)
{
	proj_A = NULL;
	proj_B = NULL;
	if (g == NULL || params == NULL)
		return false;
	if (params->geometry == parameters::MODULAR)
	{
		LOG(logERROR, "", "estimateTilt") << "Error: this algorithm does not work with modular-beam geometry\n" << endl;
		return false;
	}
	if (params->angularRange < min(179.0, 180.0 - fabs(params->T_phi()) * 180.0 / PI))
	{
		LOG(logERROR, "", "estimateTilt") << "Error: this algorithm requires an angular range of at least 180 degrees\n" << endl;
		return false;
	}

	// Now get two projections separated by 180 degrees
	float leastAngle = min(params->phis[params->numAngles - 1], params->phis[0]);
	float maxAngle = max(params->phis[params->numAngles - 1], params->phis[0]);
	float midAngle = 0.5 * (params->phis[params->numAngles - 1] + params->phis[0]);
	float angle_A = midAngle - 0.5 * PI;
	float angle_B = midAngle + 0.5 * PI;

	if (angle_A < leastAngle)
	{
		angle_A = leastAngle;
		angle_B = angle_A + PI;
	}
	else if (angle_B > maxAngle)
	{
		angle_B = maxAngle;
		angle_A = angle_B - PI;
	}
	//printf("angles = %f, %f\n", angle_A, angle_B);

	if (params->geometry == parameters::FAN || params->geometry == parameters::CONE)
	{
		rebin rebinRoutines;
		proj_A = rebinRoutines.rebin_parallel_singleProjection(g, params, 6, angle_A);
		proj_B = rebinRoutines.rebin_parallel_singleProjection(g, params, 6, angle_B);
	}
	else
	{
		proj_A = new float[params->numRows * params->numCols];
		proj_B = new float[params->numRows * params->numCols];

		float ind_A = params->phi_inv(angle_A);
		int ind_A_low = int(floor(ind_A));
		int ind_A_high = int(ceil(ind_A));
		float d_A = ind_A - float(ind_A_low);
		float* proj_A_low = &g[uint64(ind_A_low) * uint64(params->numRows * params->numCols)];
		float* proj_A_high = &g[uint64(ind_A_high) * uint64(params->numRows * params->numCols)];

		float ind_B = params->phi_inv(angle_B);
		int ind_B_low = int(floor(ind_B));
		int ind_B_high = int(ceil(ind_B));
		float d_B = ind_B - float(ind_B_low);
		float* proj_B_low = &g[uint64(ind_B_low) * uint64(params->numRows * params->numCols)];
		float* proj_B_high = &g[uint64(ind_B_high) * uint64(params->numRows * params->numCols)];

		//printf("inds = %f, %f\n", ind_A, ind_B);

		omp_set_num_threads(omp_get_num_procs());
		#pragma omp parallel for
		for (int iRow = 0; iRow < params->numRows; iRow++)
		{
			for (int iCol = 0; iCol < params->numCols; iCol++)
			{
				int ind = iRow * params->numCols + iCol;
				proj_A[ind] = (1.0 - d_A) * proj_A_low[ind] + d_A * proj_A_high[ind];
				proj_B[ind] = (1.0 - d_B) * proj_B_low[ind] + d_B * proj_B_high[ind];
			}
		}
	}
	if (proj_A != NULL && proj_B != NULL)
		return true;
	else
		return false;
}

bool getConjugateDifference(float* g, parameters* params, float alpha, float centerCol, float* diff)
{
	if (g == NULL || params == NULL || diff == NULL)
		return false;
	float* proj_A = NULL;
	float* proj_B = NULL;
	float centerCol_save = params->centerCol;
	params->centerCol = centerCol;
	if (getConjugateProjections(g, params, proj_A, proj_B) == false)
	{
		params->centerCol = centerCol_save;
		return false;
	}
	params->centerCol = centerCol_save;

	float row_0_centered = -0.5 * float(params->numRows - 1) * params->pixelHeight;
	float col_0_centered = -0.5 * float(params->numCols - 1) * params->pixelWidth;

	float row_0 = -params->centerRow * params->pixelHeight;
	float col_0 = -centerCol * params->pixelWidth;

	float cos_alpha = cos(PI / 180.0 * alpha);
	float sin_alpha = sin(PI / 180.0 * alpha);

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iRow = 0; iRow < params->numRows; iRow++)
	{
		float* diff_line = &diff[iRow * params->numCols];
		//float row = iRow * params->pixelHeight + row_0;
		float row = iRow * params->pixelHeight + row_0_centered;
		for (int iCol = 0; iCol < params->numCols; iCol++)
		{
			//float col = iCol * params->pixelWidth + col_0;
			float col = iCol * params->pixelWidth + col_0_centered;

			float col_A = cos_alpha * col + sin_alpha * row - col_0 + col_0_centered;
			float row_A = -sin_alpha * col + cos_alpha * row;
			float col_A_ind = (col_A - col_0) / params->pixelWidth;
			float row_A_ind = (row_A - row_0_centered) / params->pixelHeight;

			float col_B = -(cos_alpha * col - sin_alpha * row - col_0 + col_0_centered);
			float row_B = sin_alpha * col + cos_alpha * row;
			float col_B_ind = (col_B - col_0) / params->pixelWidth;
			float row_B_ind = (row_B - row_0_centered) / params->pixelHeight;

			float proj_A_cur = interpolate2D(proj_A, row_A_ind, col_A_ind, params->numRows, params->numCols);
			float proj_B_cur = interpolate2D(proj_B, row_B_ind, col_B_ind, params->numRows, params->numCols);

			if (0.0 <= row_A_ind && row_A_ind <= float(params->numRows - 1) && 0.0 <= row_B_ind && row_B_ind <= float(params->numRows - 1) &&
				0.0 <= col_A_ind && col_A_ind <= float(params->numCols - 1) && 0.0 <= col_B_ind && col_B_ind <= float(params->numCols - 1))
			{
				diff_line[iCol] = proj_A_cur - proj_B_cur;
			}
			else
			{
				diff_line[iCol] = 0.0;
			}
		}
	}

	delete[] proj_A;
	delete[] proj_B;

	return true;
}

float estimateTilt(float* g, parameters* params)
{
	if (g == NULL || params == NULL)
		return 0.0;
	//if (findCenter_cpu(g, params) == false)
	//	return 0.0;

	float* proj_A = NULL;
	float* proj_B = NULL;
	if (getConjugateProjections(g, params, proj_A, proj_B) == false)
		return 0.0;

	float tilt_max = 4.9;
	float tilt_0 = -1.0 * tilt_max;
	float T_tilt = 0.1;
	int N_tilt = 2 * int(floor(0.5 + tilt_max / T_tilt)) + 1;
	double* errors = new double[N_tilt];

	float row_0_centered = -0.5 * float(params->numRows - 1) * params->pixelHeight;
	float col_0_centered = -0.5 * float(params->numCols - 1) * params->pixelWidth;

	float row_0 = -params->centerRow * params->pixelHeight;
	float col_0 = -params->centerCol * params->pixelWidth;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int itilt = 0; itilt < N_tilt; itilt++)
	{
		float alpha = itilt * T_tilt + tilt_0;
		//printf("alpha[%d] = %f\n", itilt, alpha);
		float cos_alpha = cos(PI / 180.0 * alpha);
		float sin_alpha = sin(PI / 180.0 * alpha);
		int count = 0;
		double curError = 0.0;
		for (int iRow = 0; iRow < params->numRows; iRow++)
		{
			//float row = iRow * params->pixelHeight + row_0;
			float row = iRow * params->pixelHeight + row_0_centered;
			for (int iCol = 0; iCol < params->numCols; iCol++)
			{
				//float col = iCol * params->pixelWidth + col_0;
				float col = iCol * params->pixelWidth + col_0_centered;

				float col_A = cos_alpha * col + sin_alpha * row - col_0 + col_0_centered;
				float row_A = -sin_alpha * col + cos_alpha * row;
				float col_A_ind = (col_A - col_0) / params->pixelWidth;
				float row_A_ind = (row_A - row_0_centered) / params->pixelHeight;

				float col_B = -(cos_alpha * col - sin_alpha * row - col_0 + col_0_centered);
				float row_B = sin_alpha * col + cos_alpha * row;
				float col_B_ind = (col_B - col_0) / params->pixelWidth;
				float row_B_ind = (row_B - row_0_centered) / params->pixelHeight;

				float proj_A_cur = interpolate2D(proj_A, row_A_ind, col_A_ind, params->numRows, params->numCols);
				float proj_B_cur = interpolate2D(proj_B, row_B_ind, col_B_ind, params->numRows, params->numCols);

				//g[itilt * params->numRows * params->numCols + iRow * params->numCols + iCol] = proj_A_cur - proj_B_cur;

				if (0.0 <= row_A_ind && row_A_ind <= float(params->numRows - 1) && 0.0 <= row_B_ind && row_B_ind <= float(params->numRows - 1) &&
					0.0 <= col_A_ind && col_A_ind <= float(params->numCols - 1) && 0.0 <= col_B_ind && col_B_ind <= float(params->numCols - 1))
				{
					count += 1;
					double diff = proj_A_cur - proj_B_cur;
					curError += diff * diff;
				}
			}
		}
		if (count > 0)
			errors[itilt] = curError / float(count);
		else
			errors[itilt] = 1.0e30;
	}

	/*
	float minError = errors[0];
	int ind_min = 0;
	for (int itilt = 1; itilt < N_tilt; itilt++)
	{
		if (errors[itilt] < minError)
		{
			minError = errors[itilt];
			ind_min = itilt;
		}
		printf("error[%f] = %f\n", itilt * T_tilt + tilt_0, errors[itilt]);
	}
	float retVal = ind_min * T_tilt + tilt_0;
	//*/
	//for (int itilt = 0; itilt < N_tilt; itilt++)
	//	printf("error[%f] = %f\n", itilt * T_tilt + tilt_0, errors[itilt]);
	float retVal = T_tilt*findMinimum(errors, 0, N_tilt) + tilt_0;

	// for loop over [centerCol-?, centerCol+?] in 1 pixel steps
	//	for loop over [tiltAngle-4.9, tiltAngle+4.9] in 0.1 degree steps
	// free temporary memory

	delete[] errors;
	delete[] proj_A;
	delete[] proj_B;

	return retVal;
}

float interpolate2D(float* data, float ind_1, float ind_2, int N_1, int N_2)
{
	int ind_1_lo, ind_1_hi;
	float d_1;
	if (ind_1 <= 0.0)
	{
		ind_1_lo = 0;
		ind_1_hi = 0;
		d_1 = 0.0;
	}
	else if (ind_1 >= float(N_1 - 1))
	{
		ind_1_lo = N_1-1;
		ind_1_hi = N_1-1;
		d_1 = 0.0;
	}
	else
	{
		ind_1_lo = int(ind_1);
		ind_1_hi = ind_1_lo + 1;
		d_1 = ind_1 - float(ind_1_lo);
	}

	int ind_2_lo, ind_2_hi;
	float d_2;
	if (ind_2 <= 0.0)
	{
		ind_2_lo = 0;
		ind_2_hi = 0;
		d_2 = 0.0;
	}
	else if (ind_2 >= float(N_2 - 1))
	{
		ind_2_lo = N_2 - 1;
		ind_2_hi = N_2 - 1;
		d_2 = 0.0;
	}
	else
	{
		ind_2_lo = int(ind_2);
		ind_2_hi = ind_2_lo + 1;
		d_2 = ind_2 - float(ind_2_lo);
	}

	return (1.0 - d_1)* ((1.0 - d_2) * data[ind_1_lo * N_2 + ind_2_lo] + d_2 * data[ind_1_lo * N_2 + ind_2_hi]) +
		d_1 * ((1.0 - d_2) * data[ind_1_hi * N_2 + ind_2_lo] + d_2 * data[ind_1_hi * N_2 + ind_2_hi]);
}

bool findCenter_cpu(float* g, parameters* params, int iRow)
{
	if (g == NULL || params == NULL)
		return false;
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

	if (params->angularRange + 2.0*fabs(params->T_phi()) * 180.0 / PI < 180.0)
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

	setDefaultRange_centerCol(params, centerCol_low, centerCol_high);

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
		double count = 0.0;
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
							count += 1.0;
						}
					}
				}
			}
		}
		//printf("%f ", num);
#ifdef USE_MEAN_DIFFERENCE_METRIC
		if (count > 0.0)
			shiftCosts[n] = num / count;
		else
			shiftCosts[n] = 0.0;
#else
		shiftCosts[n] = num;
#endif
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

	setDefaultRange_centerCol(params, centerCol_low, centerCol_high);
	
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
		double count = 0.0;
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
						phi_conj -= float(2.0 * PI);
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
						count += 1.0;
					}
				}
			}
		}
		//printf("%f ", num);
#ifdef USE_MEAN_DIFFERENCE_METRIC
		if (count > 0.0)
			shiftCosts[n] = num / count;
		else
			shiftCosts[n] = 0.0;
#else
		shiftCosts[n] = num;
#endif
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

bool setDefaultRange_centerCol(parameters* params, int& centerCol_low, int& centerCol_high)
{
	double c = 0.23;
	if (params->offsetScan == true)
		c = 0.1;
	int N_trim = 50;
	if (params->numCols < 200)
		N_trim = 5;

	centerCol_low = int(floor(c * params->numCols));
	centerCol_high = int(ceil(params->numCols - c * params->numCols));

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

	centerCol_low = max(0, min(params->numCols - 1, centerCol_low));
	centerCol_high = max(0, min(params->numCols - 1, centerCol_high));
	if (centerCol_low > centerCol_high)
	{
		centerCol_low = 5;
		centerCol_high = params->numCols - 1 - 5;
	}
	return true;
}
