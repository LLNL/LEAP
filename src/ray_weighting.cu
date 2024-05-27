////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for ray weighting
////////////////////////////////////////////////////////////////////////////////
#include "ray_weighting.cuh"
#include "ray_weighting_cpu.h"
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "log.h"

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

__global__ void applyViewDependentPolarWeightsKernel(float* g, const float* w_polar, const int3 N, const bool doInverse)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N.x || j >= N.y || k >= N.z)
		return;

	if (doInverse)
		g[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] /= w_polar[i * N.y + j];
	else
		g[uint64(i) * uint64(N.y * N.z) + uint64(j * N.z + k)] *= w_polar[i * N.y + j];
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

bool applyPreRampFilterWeights_GPU(float* g, parameters* params, bool data_on_cpu)
{
	float* w_ray = setPreRampFilterWeights(params);
	float* w_view = setViewWeights(params); // numAngles X numCols

	if (w_ray == NULL && w_view == NULL)
		return true;
	else
	{
		LOG(logDEBUG, "ray_weighting", "applyPreRampFilterWeights_GPU") << "GPU " << params->whichGPU << ": start" << std::endl;
		cudaError_t cudaStatus;
		cudaSetDevice(params->whichGPU);

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (data_on_cpu)
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
				fprintf(stderr, "Error: applyPreRampFilterWeights_GPU: cudaMalloc failed!\n");
			if (cudaSuccess != cudaMemcpy(dev_w_view, w_view, params->numAngles * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "Error: applyPreRampFilterWeights_GPU: cudaMemcpy failed!\n");
		}
		float* dev_w_ray = 0;
		if (w_ray != NULL)
		{
			if (cudaSuccess != cudaMalloc((void**)&dev_w_ray, params->numRows * params->numCols * sizeof(float)))
				fprintf(stderr, "Error: applyPreRampFilterWeights_GPU: cudaMalloc failed!\n");
			if (cudaSuccess != cudaMemcpy(dev_w_ray, w_ray, params->numRows * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "Error: applyPreRampFilterWeights_GPU: cudaMemcpy failed!\n");
		}

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		applyWeightsKernel <<< dimGrid, dimBlock >>> (dev_g, dev_w_view, dev_w_ray, N);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Error: applyWeightsKernel: kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (data_on_cpu)
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
		if (data_on_cpu == true && dev_g != 0)
			cudaFree(dev_g);

		LOG(logDEBUG, "ray_weighting", "applyPreRampFilterWeights_GPU") << "GPU " << params->whichGPU << ": completed successfully" << std::endl;

		return true;
	}
}

bool applyPostRampFilterWeights_GPU(float* g, parameters* params, bool data_on_cpu)
{
	float* w_ray = setInverseConeWeight(params); // numRows X numCols
	if (w_ray == NULL)
		return true;
	else
	{
		LOG(logDEBUG, "ray_weighting", "applyPostRampFilterWeights_GPU") << "GPU " << params->whichGPU << ": start" << std::endl;
		cudaError_t cudaStatus;
		cudaSetDevice(params->whichGPU);

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (data_on_cpu)
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
				fprintf(stderr, "Error: applyPostRampFilterWeights_GPU: cudaMalloc failed!\n");
			if (cudaSuccess != cudaMemcpy(dev_w_ray, w_ray, params->numRows * params->numCols * sizeof(float), cudaMemcpyHostToDevice))
				fprintf(stderr, "Error: applyPostRampFilterWeights_GPU: cudaMemcpy failed!\n");
		}

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		applyWeightsKernel <<< dimGrid, dimBlock >>> (dev_g, dev_w_view, dev_w_ray, N);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Error: applyWeightsKernel: kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (data_on_cpu)
		{
			//printf("pulling projections off GPU...\n");
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
		}

		if (dev_w_ray != 0)
			cudaFree(dev_w_ray);
		if (w_ray != NULL)
			free(w_ray);
		if (data_on_cpu == true && dev_g != 0)
			cudaFree(dev_g);

		LOG(logDEBUG, "ray_weighting", "applyPostRampFilterWeights_GPU") << "GPU " << params->whichGPU << ": completed successfully" << std::endl;

		return true;
	}
}

bool convertARTtoERT(float* g, parameters* params, bool data_on_cpu, bool doInverse)
{
	if (params->whichGPU < 0)
		return convertARTtoERT_CPU(g, params, doInverse);
	else
	{
		cudaError_t cudaStatus;
		cudaSetDevice(params->whichGPU);

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (data_on_cpu)
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
			fprintf(stderr, "Error: convertARTtoERT: kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (data_on_cpu)
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

		if (data_on_cpu == true && dev_g != 0)
			cudaFree(dev_g);

		return true;
	}
}

bool applyViewDependentPolarWeights_gpu(float* g, parameters* params, float* w_in, bool data_on_cpu, bool doInverse)
{
	if (params->whichGPU < 0)
		return false;
	else
	{
		float* w = NULL;
		if (w_in == NULL)
			w = setViewDependentPolarWeights(params);
		else
			w = w_in;
		cudaError_t cudaStatus;
		cudaSetDevice(params->whichGPU);

		int3 N = make_int3(params->numAngles, params->numRows, params->numCols);
		float* dev_g = 0;
		if (data_on_cpu)
			dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
		else
			dev_g = g;

		float* dev_w = 0;
		if (cudaSuccess != cudaMalloc((void**)&dev_w, params->numRows * params->numAngles * sizeof(float)))
			fprintf(stderr, "Error: applyViewDependentPolarWeights_gpu: cudaMalloc failed!\n");
		if (cudaSuccess != cudaMemcpy(dev_w, w, params->numRows * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
			fprintf(stderr, "Error: applyViewDependentPolarWeights_gpu: cudaMemcpy failed!\n");

		dim3 dimBlock = setBlockSize(N);
		dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
			int(ceil(double(N.z) / double(dimBlock.z))));
		applyViewDependentPolarWeightsKernel <<< dimGrid, dimBlock >>> (dev_g, dev_w, N, doInverse);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Error: applyViewDependentPolarWeightsKernel: kernel failed!\n");
			fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
			fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		}

		if (data_on_cpu)
			pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);

		if (data_on_cpu == true && dev_g != 0)
			cudaFree(dev_g);
		if (dev_w != 0)
			cudaFree(dev_w);
		if (w_in == NULL)
			free(w);

		return true;
	}
}
