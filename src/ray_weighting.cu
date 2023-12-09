#include "ray_weighting.cuh"

#include "cuda_runtime.h"
#include "cuda_utils.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

__global__ void convertARTtoERTkernel(float* g, const float muCoeff, const float muRadius, const float T_u, const float u_0, int3 N)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N.x || j >= N.y || k >= N.z)
		return;

	//weight = np.sqrt(np.clip(150.0 * *2 - u * *2, 0.0, 150.0 * *2))
	//weight = np.exp(muCoeff * weight)
	const float u = T_u * k + u_0;
	if (fabs(u) < muRadius)
		g[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] *= expf(muCoeff * sqrt(muRadius*muRadius - u*u));
}

__global__ void applyWeightsKernel(float* g, const float* w_view, const float* w_ray, int3 N)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N.x || j >= N.y || k >= N.z)
		return;

	float theWeight = 1.0f;
	if (w_ray != NULL)
		theWeight *= w_ray[j * N.z + k];
	if (w_view != NULL)
		theWeight *= w_view[i * N.z + k];
	g[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] *= theWeight;
}

float FBPscalar(parameters* params)
{
	float magFactor = params->sod / params->sdd;
	if (params->geometry == parameters::CONE)
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth * magFactor * params->pixelHeight * magFactor / (params->voxelWidth * params->voxelWidth * params->voxelHeight));
	else if (params->geometry == parameters::FAN)
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth * magFactor * params->pixelHeight / (params->voxelWidth * params->voxelWidth * params->voxelHeight));
	else
		return 1.0 / (2.0 * PI) * fabs(params->T_phi() * params->pixelWidth / (params->voxelWidth * params->voxelWidth));
}

float* setViewWeights(parameters* params)
{
	float* w = setParkerWeights(params);
	return setRedundantAndNonEquispacedViewWeights(params, w);
}

float* setParkerWeights(parameters* params)
{
	if (params->angularRange >= 359.9999 || params->numAngles == 1)
		return NULL;

	if (params->geometry == parameters::CONE || params->geometry == parameters::FAN)
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;
		float* retVal = (float*)malloc(sizeof(float) * params->numAngles * params->numCols);

		double beta_max = params->angularRange*PI / 180.0;

		double gamma = asin(params->tau / sqrt(params->sod * params->sod + params->tau * params->tau));
		double alpha_max;
		if (params->detectorType == parameters::FLAT)
			alpha_max = min(fabs(atan(params->u(0)) - gamma), fabs(atan(params->u(params->numCols - 1)) - gamma));
		else
			alpha_max = min(fabs(params->u(0) - gamma), fabs(params->u(params->numCols - 1) - gamma));

		double shortScanThreshold = PI + 2.0 * alpha_max;

		if (beta_max < shortScanThreshold)
			printf("FDK::calcParkerWeights: Not enough data!\n");

		double thres = (beta_max - PI) / 2.0;
		if (thres < 0.0)
			thres = 0.0;
		double beta, alpha, theWeight;
		double alpha_c;

		double plus_or_minus = 1.0;
		if (params->T_phi() < 0.0)
			plus_or_minus = -1.0;
		//plus_or_minus = 1.0;
		for (int j = 0; j < params->numCols; j++)
		{
			if (params->detectorType == parameters::FLAT)
				alpha = plus_or_minus * atan(params->u(j)) - gamma;
			else
				alpha = plus_or_minus * params->u(j) - gamma;

			alpha_c = -alpha;

			for (int i = 0; i < params->numAngles; i++)
			{
				beta = fabs(params->phis[i] - params->phis[0]);
				//beta = fabs(g->T_phi)*double(i); //g->phi(i) - g->phi_0;
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
				retVal[i*params->numCols+j] = theWeight;
			}
		}

		params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

		return retVal;
	}
	else if (params->geometry == parameters::PARALLEL)
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
				retVal[i*params->numCols+j] = theWeight;
		}
		return retVal;
	}
	else
		return NULL;
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
		if (i == 0)
		{
			theWeight = fabs(params->phis[1] - params->phis[0]) / T_phi;
		}
		else if (i == params->numAngles - 1)
		{
			theWeight = fabs(params->phis[params->numAngles - 1] - params->phis[params->numAngles - 2]) / T_phi;
		}
		else
		{
			theWeight = 0.5 * (fabs(params->phis[i + 1] - params->phis[i]) + fabs(params->phis[i] - params->phis[i - 1])) / T_phi;
		}
		for (int j = 0; j < params->numCols; j++)
			retVal[i * params->numCols + j] *= theWeight;
	}

	// Now apply weights for cases where we have redundant measurements
	if ((params->geometry == parameters::PARALLEL && params->angularRange >= 359.9999) || 
		(params->geometry == parameters::FAN && params->angularRange >= 359.9999) ||
		(params->geometry == parameters::CONE && params->angularRange >= 359.9999))
	{
		float c = 0.5;
		//float c = 1.0;
		//if (params->geometry == parameters::FAN || params->geometry == parameters::CONE)
		//	c = 0.5;
		float T = fabs(params->T_phi());
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
	return retVal;
}

float* setInverseConeWeight(parameters* params)
{
	if (params->geometry == parameters::CONE)
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;

		float* retVal = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
		for (int iv = 0; iv < params->numRows; iv++)
		{
			float v = params->v(iv);
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu);
				if (params->detectorType == parameters::FLAT)
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

float* setPreRampFilterWeights(parameters* params)
{
	float* w = setInverseConeWeight(params); // numRows X numCols
	if (w != NULL && (params->geometry == parameters::CONE || params->geometry == parameters::FAN))
	{
		bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
		params->normalizeConeAndFanCoordinateFunctions = true;
		for (int iv = 0; iv < params->numRows; iv++)
		{
			for (int iu = 0; iu < params->numCols; iu++)
			{
				float u = params->u(iu);
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

bool applyPreRampFilterWeights_GPU(float* g, parameters* params, bool cpu_to_gpu)
{
	float* w_ray = setPreRampFilterWeights(params);
	float* w_view = setViewWeights(params); // numAngles X numCols

	if (w_ray == NULL && w_view == NULL)
		return true;
	else
	{
		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (cpu_to_gpu)
		{
			dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		}
		else
		{
			dev_g = g;
		}

		float* dev_w_view = 0;
		if (w_view != NULL)
		{
			if (cudaSuccess != cudaMalloc((void**)&dev_w_view, params->numAngles * params->numCols * sizeof(float)))
				fprintf(stderr, "cudaMalloc failed!\n");
			if (cudaMemcpy(dev_w_view, w_view, params->numAngles * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "cudaMemcpy failed!\n");
		}
		float* dev_w_ray = 0;
		if (w_ray != NULL)
		{
			if (cudaSuccess != cudaMalloc((void**)&dev_w_ray, params->numRows * params->numCols * sizeof(float)))
				fprintf(stderr, "cudaMalloc failed!\n");
			if (cudaMemcpy(dev_w_ray, w_ray, params->numRows * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "cudaMemcpy failed!\n");
		}

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		applyWeightsKernel <<< dimGrid, dimBlock >>> (dev_g, dev_w_view, dev_w_ray, N);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (cpu_to_gpu)
		{
			//printf("pulling projections off GPU...\n");
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
		}

		if (dev_w_view != 0)
			cudaFree(dev_w_view);
		if (dev_w_ray != 0)
			cudaFree(dev_w_ray);
		if (w_ray != NULL)
			free(w_ray);
		if (w_view != NULL)
			free(w_view);
		if (cpu_to_gpu == true && dev_g != 0)
			cudaFree(dev_g);

		return true;
	}
}

bool applyPostRampFilterWeights_GPU(float* g, parameters* params, bool cpu_to_gpu)
{
	float* w_ray = setInverseConeWeight(params); // numRows X numCols
	if (w_ray == NULL)
		return true;
	else
	{
		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (cpu_to_gpu)
		{
			dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		}
		else
		{
			dev_g = g;
		}

		float* dev_w_view = 0;
		float* dev_w_ray = 0;
		if (w_ray != NULL)
		{
			if (cudaSuccess != cudaMalloc((void**)&dev_w_ray, params->numRows * params->numCols * sizeof(float)))
				fprintf(stderr, "cudaMalloc failed!\n");
			if (cudaMemcpy(dev_w_ray, w_ray, params->numRows * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "cudaMemcpy failed!\n");
		}

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		applyWeightsKernel <<< dimGrid, dimBlock >>> (dev_g, dev_w_view, dev_w_ray, N);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (cpu_to_gpu)
		{
			//printf("pulling projections off GPU...\n");
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
		}

		if (dev_w_ray != 0)
			cudaFree(dev_w_ray);
		if (w_ray != NULL)
			free(w_ray);
		if (cpu_to_gpu == true && dev_g != 0)
			cudaFree(dev_g);

		return true;
	}
}

bool applyPreRampFilterWeights(float* g, parameters* params, bool cpu_to_gpu)
{
	if (params->whichGPU < 0)
		return applyPreRampFilterWeights_CPU(g, params);
	else
		return applyPreRampFilterWeights_GPU(g, params, cpu_to_gpu);
}

bool applyPostRampFilterWeights(float* g, parameters* params, bool cpu_to_gpu)
{
	if (params->whichGPU < 0)
		return applyPostRampFilterWeights_CPU(g, params);
	else
		return applyPostRampFilterWeights_GPU(g, params, cpu_to_gpu);
}

bool convertARTtoERT(float* g, parameters* params, bool cpu_to_gpu, bool doInverse)
{
	if (params->whichGPU < 0)
		return convertARTtoERT_CPU(g, params, doInverse);
	else
	{
		cudaSetDevice(params->whichGPU);
		cudaError_t cudaStatus;

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (cpu_to_gpu)
			dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		else
			dev_g = g;

		float muCoeff = params->muCoeff;
		if (doInverse)
			muCoeff *= -1.0;

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		convertARTtoERTkernel <<< dimGrid, dimBlock >>> (dev_g, muCoeff, params->muRadius, params->pixelWidth, params->u_0(), N);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (cpu_to_gpu)
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

		if (cpu_to_gpu == true && dev_g != 0)
			cudaFree(dev_g);

		return true;
	}
}

bool convertARTtoERT_CPU(float* g, parameters* params, bool doInverse)
{
	float muCoeff = params->muCoeff;
	if (doInverse)
		muCoeff *= -1.0;
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
