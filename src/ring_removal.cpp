////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for ring removal algorithms
////////////////////////////////////////////////////////////////////////////////
#include "ring_removal.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include "log.h"
#include "leap_defines.h"
#include "cpu_utils.h"

using namespace std;

ringRemoval::ringRemoval()
{
}

ringRemoval::~ringRemoval()
{
}

bool ringRemoval::execute(float* projectionData, int N_1, int N_2, int N_3, float delta_in, float beta_in, int numIter, float maxChange, int angle_downsampling_factor)
{
	if (projectionData == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;
	delta = delta_in;
	beta = beta_in;
	if (delta < 1.0e-8)
		delta = float(1.0e-8);

	bool doLowPass = false;
	//bool doLowPass = true;

	angle_downsampling_factor = max(1, min(angle_downsampling_factor, N_1));
	int numAngles = N_1/angle_downsampling_factor;

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for schedule(dynamic)
	for (int iRow = 0; iRow < N_2; iRow++)
	{
		float* temp_data = new float[(numAngles+2) * N_3];
		float* g_0 = temp_data;
		float* d = &temp_data[numAngles*N_3];
		float* gainMap = &temp_data[(numAngles + 1) * N_3];

		/*
		float* g_0 = new float[N_1 * N_3];
		float* d = new float[N_3];
		float* gainMap = new float[N_3];
		//*/

		if (numAngles == N_1)
		{
			for (int iAngle = 0; iAngle < N_1; iAngle++)
			{
				float* aLine = &projectionData[uint64(iAngle) * uint64(N_2 * N_3) + uint64(iRow * N_3)];
				float* g_iAngle = &g_0[iAngle * N_3];
				memcpy(g_iAngle, aLine, sizeof(float) * N_3);
			}
		}
		else
		{
			for (int i = 0; i < numAngles; i++)
			{
				float* g_i = &g_0[i * N_3];
				for (int ii = 0; ii < angle_downsampling_factor; ii++)
				{
					int iAngle = i*angle_downsampling_factor + ii;
					float* aLine = &projectionData[uint64(iAngle) * uint64(N_2 * N_3) + uint64(iRow * N_3)];
					for (int k = 0; k < N_3; k++)
					{
						if (ii == 0)
							g_i[k] = aLine[k]/float(angle_downsampling_factor);
						else
							g_i[k] += aLine[k]/float(angle_downsampling_factor);
					}
				}
			}
		}
		
		// Algorithm goes here
		memset(gainMap, 0, N_3 * sizeof(float));
		for (int n = 0; n < numIter; n++)
		{
			memset(d, 0, N_3 * sizeof(float));
			int count = 0;
			for (int i = 0; i < numAngles; i++)
			{
				float* aLine_0 = &g_0[i * N_3];

				float backDiff = 0.0;
				float foreDiff = 0.0;
				for (int j = 0; j < N_3; j++)
				{
					backDiff = -foreDiff;

					float curVal = aLine_0[j] + gainMap[j];
					float nextVal = curVal;
					if (j + 1 <= N_3 - 1)
						nextVal = aLine_0[j + 1] + gainMap[j + 1];

					if (j + 1 <= N_3 - 1)
						foreDiff = h1(curVal - nextVal);
					else
						foreDiff = 0.0;

					float Sf1_cur = beta * (foreDiff + backDiff);

					/*
					if (i == 0)
						d[j] = Sf1_cur;// / float(numAngles);
					else
						d[j] += Sf1_cur / float(numAngles);
					d[j] += gainMap[j];
					//*/
					if (doLowPass)
					{
						//double val = 3.0/9.0*gainMap[j] + 2.0/9.0*(gainMap[max(0,j-1)] + gainMap[min(N_3-1,j+1)]) + 1.0/9.0*(gainMap[max(0,j-2)] + gainMap[min(N_3-1,j+2)]);
						double val = 5.0/25.0*gainMap[j] + 4.0/25.0*(gainMap[max(0,j-1)] + gainMap[min(N_3-1,j+1)]) + 3.0/25.0*(gainMap[max(0,j-2)] + gainMap[min(N_3-1,j+2)]) + 2.0/25.0*(gainMap[max(0,j-3)] + gainMap[min(N_3-1,j+3)]) + 1.0/25.0*(gainMap[max(0,j-4)] + gainMap[min(N_3-1,j+4)]);
						d[j] += val + Sf1_cur;
					}
					else
						d[j] += (gainMap[j] + Sf1_cur);// / float(numAngles) * float(angle_downsampling_factor);
				}
				count += 1;
			}

			for (int j = 0; j < N_3; j++)
				d[j] = d[j] / float(count);

			double num = 0.0;
			double denomA = 0.0;
			double denomB = 0.0;
			for (int i = 0; i < numAngles; i++)
			{
				float* aLine_0 = &g_0[i * N_3];

				float backDiff_2 = 0.0;
				float foreDiff_2 = 0.0;

				float backDiff_d = 0.0;
				float foreDiff_d = 0.0;
				for (int j = 0; j < N_3; j++)
				{
					float curVal = aLine_0[j] + gainMap[j];
					float nextVal = 0.0;
					if (j + 1 <= N_3 - 1)
						nextVal = aLine_0[j + 1] + gainMap[j + 1];
					else
						nextVal = curVal;

					backDiff_d = -foreDiff_d;
					if (j + 1 <= N_3 - 1)
						foreDiff_d = (d[j] - d[j + 1]);
					else
						foreDiff_d = 0.0;

					backDiff_2 = foreDiff_2;
					foreDiff_2 = h2(curVal - nextVal);

					num += d[j] * d[j];
					if (doLowPass)
					{
						//double val = 1.0/3.0*(d[j] + d[max(0,j-1)] + d[min(N_3-1,j+1)]);
						double val = 1.0/5.0*(d[j] + d[max(0,j-1)] + d[min(N_3-1,j+1)] + d[max(0,j-2)] + d[min(N_3-1,j+2)]);
						denomA += val*val;
					}
					else
					{
						denomA += d[j] * d[j];
					}
					denomB += beta * (foreDiff_d * foreDiff_d * foreDiff_2 + backDiff_d * backDiff_d * backDiff_2);
					//denom += beta * (foreDiff_d * foreDiff_d * foreDiff_2);
				}
			}
			double denom = denomA + denomB;
			//num *= numAngles;
			if (fabs(denom) < 1.0e-16)
				break;
			float lambda = num / denom;
			for (int j = 0; j < N_3; j++)
			{
				gainMap[j] -= lambda * d[j];
				gainMap[j] = max(-maxChange, min(gainMap[j], maxChange));
			}
			/*
			if (iRow == N_2 / 2)
			{
				printf("lambda = %f = %f / %f\n", lambda, num, denom);
				float minGain = gainMap[0];
				float maxGain = gainMap[0];
				for (int j = 0; j < N_3; j++)
				{
					if (n == 0)
						printf(" %f", gainMap[j]);
					minGain = min(minGain, gainMap[j]);
					maxGain = max(maxGain, gainMap[j]);
				}
				printf("gain range: %f to %f\n", minGain, maxGain);
			}
			//*/
		}

		// Apply estimated gain map
		for (int iAngle = 0; iAngle < N_1; iAngle++)
		{
			float* aLine = &projectionData[uint64(iAngle) * uint64(N_2 * N_3) + uint64(iRow * N_3)];
			for (int iCol = 0; iCol < N_3; iCol++)
				aLine[iCol] += max(-maxChange, min(gainMap[iCol], maxChange));
		}

		/*
		delete[] g_0;
		delete[] d;
		delete[] gainMap;
		//*/
		delete[] temp_data;
	}
	return true;
}

float ringRemoval::h1(float t)
{
	if (fabs(t) <= delta)
		return t;
	else
		return (t > 0.0f) ? delta : -delta;
}

float ringRemoval::h2(float t)
{
	return delta / max(delta, fabs(t));
}
