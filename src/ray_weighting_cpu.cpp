////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for ray weighting
////////////////////////////////////////////////////////////////////////////////
#include "ray_weighting_cpu.h"
#ifndef __USE_CPU
#include "ray_weighting.cuh"
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include "log.h"

using namespace std;

float FBPscalar(parameters* params)
{
	float magFactor = params->sod / params->sdd;
	if (params->geometry == parameters::CONE || params->modularbeamIsAxiallyAligned())
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth * magFactor * params->pixelHeight * magFactor / (params->voxelWidth * params->voxelWidth * params->voxelHeight));
	else if (params->geometry == parameters::FAN)
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth * magFactor * params->pixelHeight / (params->voxelWidth * params->voxelWidth * params->voxelHeight));
	else if (params->geometry == parameters::CONE_PARALLEL)
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth * params->pixelHeight * magFactor / (params->voxelWidth * params->voxelWidth * params->voxelHeight));
	else
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth / (params->voxelWidth * params->voxelWidth));
}

bool applyPreRampFilterWeights(float* g, parameters* params, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params->whichGPU < 0)
		return applyPreRampFilterWeights_CPU(g, params);
	else
		return applyPreRampFilterWeights_GPU(g, params, data_on_cpu);
#else
	return applyPreRampFilterWeights_CPU(g, params);
#endif
}

bool applyPostRampFilterWeights(float* g, parameters* params, bool data_on_cpu)
{
#ifndef __USE_CPU
	if (params->whichGPU < 0)
		return applyPostRampFilterWeights_CPU(g, params);
	else
		return applyPostRampFilterWeights_GPU(g, params, data_on_cpu);
#else
	return applyPostRampFilterWeights_CPU(g, params);
#endif
}

float* setViewWeights(parameters* params)
{
	float* w = setParkerWeights(params);
	return setRedundantAndNonEquispacedViewWeights(params, w);
}

float* setParkerWeights(parameters* params)
{
	if (params->numAngles == 1)
		return NULL;
	else if (params->angularRange >= 359.9999)
		return NULL;

	if (params->geometry == parameters::CONE || params->geometry == parameters::FAN || params->modularbeamIsAxiallyAligned())
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;

		double beta_max = params->angularRange * PI / 180.0;

		double gamma = asin(params->tau / sqrt(params->sod * params->sod + params->tau * params->tau));
		double alpha_max;
		if (params->detectorType == parameters::FLAT)
		{
			alpha_max = min(fabs(atan(params->u(0)) - gamma), fabs(atan(params->u(params->numCols - 1)) - gamma));
			if (params->modularbeamIsAxiallyAligned())
			{
				for (int i = 0; i < params->numAngles; i++)
					alpha_max = max(alpha_max, min(fabs(atan(params->u(0,i)) - gamma), fabs(atan(params->u(params->numCols - 1,i)) - gamma)));
			}
		}
		else
			alpha_max = min(fabs(params->u(0) - gamma), fabs(params->u(params->numCols - 1) - gamma));

		double shortScanThreshold = PI + 2.0 * alpha_max;

		if (beta_max < shortScanThreshold)
		{
			LOG(logWARNING, "ray_weighting", "setParkerWeights") << "Not enough data (need at least " << shortScanThreshold * 180.0 / PI << " degrees)!" << std::endl;
			//printf("setParkerWeights: Not enough data (need at least %f degrees)!\n", shortScanThreshold*180.0/PI);
			return NULL;
		}

		float* retVal = (float*)malloc(sizeof(float) * params->numAngles * params->numCols);

		double thres = (beta_max - PI) / 2.0;
		if (thres < 0.0)
			thres = 0.0;
		double beta, alpha, theWeight;
		double alpha_c;

		double plus_or_minus = 1.0;
		if (params->T_phi() < 0.0)
			plus_or_minus = -1.0;
		//plus_or_minus = 1.0;
		for (int i = 0; i < params->numAngles; i++)
		{
			//beta = fabs(params->phis[i] - params->phis[0]);
			beta = fabs(params->get_phis_full(i+params->get_phi_full_ind_offset()) - params->get_phis_full(0));
			for (int j = 0; j < params->numCols; j++)
			{
				if (params->detectorType == parameters::FLAT)
					alpha = plus_or_minus * atan(params->u(j,i)) - gamma;
				else
					alpha = plus_or_minus * params->u(j,i) - gamma;

				alpha_c = -alpha;

				if (beta < 2.0 * (thres + alpha))
				{
					theWeight = sin(PI / 4.0 * beta / (thres + alpha));
					theWeight = theWeight * theWeight;
				}
				else if (beta < PI + alpha - alpha_c)
					theWeight = 1.0;
				else if (beta < PI + 2.0 * thres)
				{
					theWeight = cos(PI / 4.0 * (beta - alpha + alpha_c - PI) / (thres + alpha_c));
					theWeight = theWeight * theWeight;
				}
				else
					theWeight = 0.0;
				if (theWeight < 1e-8)
					theWeight = 1e-8;
				retVal[i * params->numCols + j] = theWeight;
			}
		}

		params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

		return retVal;
	}
	else if (params->geometry == parameters::PARALLEL || params->geometry == parameters::CONE_PARALLEL)
	{
		float T_phi = params->T_phi();
		float* retVal = (float*)malloc(sizeof(float) * params->numAngles * params->numCols);
		for (int i = 0; i < params->numAngles; i++)
		{
			double theWeight = 0.0;

			double view_1 = fabs(T_phi) * double(i);
			if (view_1 > PI)
				view_1 -= PI;

			for (int j = 0; j < params->numAngles; j++)
			{
				double view_2 = fabs(T_phi) * double(j);

				double viewOffset = min(fabs(view_1 - view_2), fabs(view_1 - (view_2 - PI)));
				viewOffset = min(viewOffset, fabs((view_1 - PI) - view_2));
				if (fabs(viewOffset) < fabs(T_phi))
					theWeight += min(0.5, viewOffset / fabs(T_phi) + 0.5) - max(-0.5, viewOffset / fabs(T_phi) - 0.5);
			}
			theWeight = 1.0 / theWeight;

			for (int j = 0; j < params->numCols; j++)
				retVal[i * params->numCols + j] = theWeight;
		}
		return retVal;
	}
	else
		return NULL;
}

float* setOffsetScanWeights(parameters* params)
{
	/*
	if (params->geometry == parameters::MODULAR)
	{
		printf("Error: offsetScan FBP not defined for modular-beam data!\n");
		return NULL;
	}
	//*/
	if (params->offsetScan == false)
		return NULL;
	else if (params->offsetScan_has_adequate_angular_range() == false)
	{
		printf("Error: offsetScan requires at least 360 degree scan!\n");
		return NULL;
	}
	else
	{
		bool doHardCut = false;
		//doHardCut = true; // JUST FOR TESTING

		//printf("applying offsetScan weights\n");
		if (params->geometry == parameters::CONE || params->geometry == parameters::FAN || params->geometry == parameters::MODULAR)
		{
			bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
			params->normalizeConeAndFanCoordinateFunctions = true;

			float alpha_min = params->u(0);
			float alpha_max = params->u(params->numCols - 1);

			if (params->detectorType == parameters::FLAT)
			{
				alpha_min = atan(alpha_min);
				alpha_max = atan(alpha_max);
			}
			float abs_minVal = fabs(params->sod * sin(alpha_min) - params->tau * cos(alpha_min));
			float abs_maxVal = fabs(params->sod * sin(alpha_max) - params->tau * cos(alpha_max));

			float delta = min(abs_minVal, abs_maxVal);
			float s_arg;

			float* retVal = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
			if (params->geometry == parameters::CONE && params->detectorType == parameters::FLAT && params->tiltAngle != 0.0)
			{
				float cos_tilt = cos(params->tiltAngle * PI / 180.0);
				float sin_tilt = sin(params->tiltAngle * PI / 180.0);
				for (int i = 0; i < params->numRows; i++)
				{
					float v = params->v(i);
					for (int j = 0; j < params->numCols; j++)
					{
						s_arg = params->u(j);
						//s_arg = cos_tilt * s_arg - sin_tilt * v;
						s_arg = (params->sod * s_arg - params->tau) / sqrt(1.0 + s_arg * s_arg);
						s_arg = cos_tilt * s_arg - sin_tilt * v;

						float theWeight = 1.0;
						if (fabs(s_arg) <= delta)
						{
							if (delta == 0.0 || doHardCut)
								theWeight = 0.5;
							else
							{
								theWeight = cos(PI / 4.0 * (s_arg - delta) / delta);
								theWeight = theWeight * theWeight;
							}
						}
						else if (s_arg < -delta)
							theWeight = 0.0;
						else
							theWeight = 1.0;

						if (abs_maxVal < abs_minVal)
							theWeight = 1.0 - theWeight;

						if (theWeight < 1e-12)
							theWeight = float(1e-12);

						retVal[i * params->numCols + j] = theWeight;
					}
				}
			}
			else
			{
				for (int j = 0; j < params->numCols; j++)
				{
					s_arg = params->u(j);
					if (params->detectorType == parameters::FLAT)
						s_arg = (params->sod * s_arg - params->tau) / sqrt(1.0 + s_arg * s_arg);
					else
						s_arg = params->sod * sin(s_arg) - params->tau * cos(s_arg);

					float theWeight = 1.0;
					if (fabs(s_arg) <= delta)
					{
						if (delta == 0.0 || doHardCut)
							theWeight = 0.5;
						else
						{
							theWeight = cos(PI / 4.0 * (s_arg - delta) / delta);
							theWeight = theWeight * theWeight;
						}
					}
					else if (s_arg < -delta)
						theWeight = 0.0;
					else
						theWeight = 1.0;

					if (abs_maxVal < abs_minVal)
						theWeight = 1.0 - theWeight;

					if (theWeight < 1e-12)
						theWeight = float(1e-12);

					for (int i = 0; i < params->numRows; i++)
						retVal[i * params->numCols + j] = theWeight;
				}
			}
			params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
			return retVal;
		}
		else if (params->geometry == parameters::PARALLEL || params->geometry == parameters::CONE_PARALLEL)
		{
			float abs_minVal = fabs(params->u(0));
			float abs_maxVal = fabs(params->u(params->numCols - 1));
			float delta = min(abs_minVal, abs_maxVal);

			float* retVal = (float*)malloc(sizeof(float) * params->numAngles * params->numCols);
			for (int j = 0; j < params->numCols; j++)
			{
				float theWeight = 1.0;
				if (fabs(params->u(j)) <= delta)
				{
					theWeight = cos(PI / 4.0 * (params->u(j) - delta) / delta);
					theWeight = theWeight * theWeight;
				}
				else if (params->u(j) < -delta)
					theWeight = 0.0;
				else
					theWeight = 1.0;

				if (abs_maxVal < abs_minVal)
					theWeight = 1.0 - theWeight;

				if (theWeight < 1e-12)
					theWeight = float(1e-12);

				for (int i = 0; i < params->numAngles; i++)
					retVal[i * params->numCols + j] = theWeight;
			}
			return retVal;
		}
		else
		{
			return NULL;
		}
	}
}

float* setRedundantAndNonEquispacedViewWeights(parameters* params, float* w)
{
	float* retVal = w;
	if (retVal == NULL)
	{
		retVal = (float*)malloc(sizeof(float) * params->numAngles * params->numCols);
		for (int i = 0; i < params->numAngles * params->numCols; i++) retVal[i] = 1.0;
	}
	if (params->numAngles < 2)
		return retVal;

	// First modify weight in cases where we have non-equispaced angles
	float T_phi = fabs(params->T_phi());
	for (int i = 0; i < params->numAngles; i++)
	{
		float theWeight = 1.0;
		if (params->is_partial_view_data())
		{
			int i_offs = i + params->get_phi_full_ind_offset();
			if (i_offs == 0)
				theWeight = fabs(params->get_phis_full(1) - params->get_phis_full(0)) / T_phi;
			else if (i_offs == params->get_numAngles_full() - 1)
				theWeight = fabs(params->get_phis_full(params->get_numAngles_full() - 1) - params->get_phis_full(params->get_numAngles_full() - 2)) / T_phi;
			else
				theWeight = 0.5 * (fabs(params->get_phis_full(i_offs + 1) - params->get_phis_full(i_offs)) + fabs(params->get_phis_full(i_offs) - params->get_phis_full(i_offs - 1))) / T_phi;
		}
		else
		{
			if (i == 0)
				theWeight = fabs(params->phis[1] - params->phis[0]) / T_phi;
			else if (i == params->numAngles - 1)
				theWeight = fabs(params->phis[params->numAngles - 1] - params->phis[params->numAngles - 2]) / T_phi;
			else
				theWeight = 0.5 * (fabs(params->phis[i + 1] - params->phis[i]) + fabs(params->phis[i] - params->phis[i - 1])) / T_phi;
		}
		for (int j = 0; j < params->numCols; j++)
			retVal[i * params->numCols + j] *= theWeight;
	}

	// Now apply weights for cases where we have redundant measurements
	if (params->angularRange >= 359.9999 && params->helicalPitch == 0.0)
	{
		float c = 0.5;
		float T = fabs(params->T_phi());
		if (params->is_partial_view_data())
		{
			for (int i = 0; i < params->numAngles; i++)
			{
				int i_offs = i + params->get_phi_full_ind_offset();
				float viewWeight = 0.0;
				for (int j = 0; j < params->get_numAngles_full(); j++)
				{
					double viewOffset = atan2(sin(double(j - i_offs) * T), cos(double(j - i_offs) * T)); // signed angular distance
					if (fabs(viewOffset) < T)
						viewWeight += min(0.5, viewOffset / T + 0.5) - max(-0.5, viewOffset / T - 0.5);
				}
				//printf("%f\n", viewWeight);
				viewWeight = 1.0 / viewWeight;
				for (int j = 0; j < params->numCols; j++)
					retVal[i * params->numCols + j] *= c * viewWeight;
			}
		}
		else
		{
			for (int i = 0; i < params->numAngles; i++)
			{
				float viewWeight = 0.0;
				for (int j = 0; j < params->numAngles; j++)
				{
					double viewOffset = atan2(sin(double(j - i) * T), cos(double(j - i) * T)); // signed angular distance
					if (fabs(viewOffset) < T)
						viewWeight += min(0.5, viewOffset / T + 0.5) - max(-0.5, viewOffset / T - 0.5);
				}
				//printf("%f\n", viewWeight);
				viewWeight = 1.0 / viewWeight;
				for (int j = 0; j < params->numCols; j++)
					retVal[i * params->numCols + j] *= c * viewWeight;
			}
		}
	}
	return retVal;
}

float* setInverseConeWeight(parameters* params)
{
	if (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL || params->modularbeamIsAxiallyAligned())
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;

		float* retVal = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
		for (int iv = 0; iv < params->numRows; iv++)
		{
			float v = params->v(iv,0) + params->v_offset();
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu,0);
				if (params->geometry == parameters::MODULAR || (params->detectorType == parameters::FLAT && params->geometry != parameters::CONE_PARALLEL))
					retVal[iv * params->numCols + iu] = 1.0 / sqrt(1.0 + u * u + v * v);
				else
					retVal[iv * params->numCols + iu] = 1.0 / sqrt(1.0 + v * v);
			}
		}
		params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

		return retVal;
	}
	else if (params->geometry == parameters::FAN)
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;

		float* retVal = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
		for (int iv = 0; iv < params->numRows; iv++)
		{
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu);
				if (params->detectorType == parameters::FLAT)
					retVal[iv * params->numCols + iu] = 1.0 / sqrt(1.0 + u * u);
				else
					retVal[iv * params->numCols + iu] = 1.0;
			}
		}
		params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

		return retVal;
	}
	else
		return NULL;
}

float* setViewDependentPolarWeights(parameters* params)
{
	bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
	params->normalizeConeAndFanCoordinateFunctions = true;

	float* retVal = (float*)malloc(sizeof(float) * params->numRows * params->numAngles);
	for (int iphi = 0; iphi < params->numAngles; iphi++)
	{
		for (int iv = 0; iv < params->numRows; iv++)
		{
			float v = params->v(iv, iphi) + params->v_offset(iphi);
			retVal[iphi * params->numRows + iv] = sqrt(1.0 + v * v);
		}
	}
	params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

	return retVal;
}

float* setPreRampFilterWeights(parameters* params)
{
	float* w = setInverseConeWeight(params); // numRows X numCols
	if (w != NULL && (params->geometry == parameters::CONE || params->geometry == parameters::FAN || params->modularbeamIsAxiallyAligned()))
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;
		for (int iv = 0; iv < params->numRows; iv++)
		{
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu,0);
				if (params->detectorType == parameters::FLAT)
					w[iv * params->numCols + iu] *= (1.0 + params->tau / params->sod * u);
				else
					w[iv * params->numCols + iu] *= (cos(u) + params->tau / params->sod * sin(u));
			}
		}
		params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
	}
	return w;
}

bool applyPreRampFilterWeights_CPU(float* g, parameters* params)
{
	float* w = setPreRampFilterWeights(params);
	float* w_view = setViewWeights(params); // numAngles X numCols

	if (w == NULL && w_view == NULL)
		return true;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = 0; iphi < params->numAngles; iphi++)
	{
		for (int iv = 0; iv < params->numRows; iv++)
		{
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float theWeight = 1.0;
				if (w != NULL)
					theWeight *= w[iv * params->numCols + iu];
				if (w_view != NULL)
					theWeight *= w_view[iphi * params->numCols + iu];
				g[uint64(iphi) * uint64(params->numRows * params->numCols) + uint64(iv * params->numCols + iu)] *= theWeight;
			}
		}
	}
	if (w != NULL)
		free(w);
	if (w_view != NULL)
		free(w_view);
	return true;
}

bool applyPostRampFilterWeights_CPU(float* g, parameters* params)
{
	float* w = setInverseConeWeight(params); // numRows X numCols
	if (w == NULL)
		return true;
	else
	{
		omp_set_num_threads(omp_get_num_procs());
		#pragma omp parallel for
		for (int iphi = 0; iphi < params->numAngles; iphi++)
		{
			for (int iv = 0; iv < params->numRows; iv++)
			{
				for (int iu = 0; iu < params->numCols; iu++)
				{
					float theWeight = 1.0;
					if (w != NULL)
						theWeight *= w[iv * params->numCols + iu];
					g[uint64(iphi) * uint64(params->numRows * params->numCols) + uint64(iv * params->numCols + iu)] *= theWeight;
				}
			}
		}
		free(w);
		return true;
	}
}

bool convertARTtoERT_CPU(float* g, parameters* params, bool doInverse)
{
	float muCoeff = params->muCoeff;
	if (doInverse)
		muCoeff *= -1.0;

	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for
	for (int iphi = 0; iphi < params->numAngles; iphi++)
	{
		for (int iv = 0; iv < params->numRows; iv++)
		{
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu);

				if (fabs(u) < params->muRadius)
					g[uint64(iphi) * uint64(params->numRows * params->numCols) + uint64(iv * params->numCols + iu)] *= exp(muCoeff * sqrt(params->muRadius * params->muRadius - u * u));
			}
		}
	}
	return true;
}
