////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// basic CUDA operations
////////////////////////////////////////////////////////////////////////////////
#include "cuda_utils.h"
#include "cuda_runtime.h"
#include <string.h>

__global__ void windowFOVKernel(float* f, const int4 N, const float4 T, const float4 startVal, const float rFOVsq, const int volumeDimensionOrder)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= N.x || iy >= N.y || iz >= N.z)
        return;

    const float x = ix * T.x + startVal.x;
    const float y = iy * T.y + startVal.y;
    const float z = iz * T.z + startVal.z;

    if (x * x + y * y > rFOVsq)
    {
        uint64 ind;
        if (volumeDimensionOrder == 0)
            ind = uint64(ix) * uint64(N.y * N.z) + uint64(iy * N.z + iz);
        else
            ind = uint64(iz) * uint64(N.y * N.x) + uint64(iy * N.x + ix);

        f[ind] = 0.0f;
    }
}

__global__ void replaceZerosKernel(float* lhs, const int3 dim, const float newVal)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (lhs[ind] == 0.0f)
        lhs[ind] = newVal;
}

__global__ void clipKernel(float* lhs, const int3 dim, const float clipVal)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    if (lhs[ind] < clipVal)
        lhs[ind] = clipVal;
}

__global__ void cosKernel(float* lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = cos(lhs[ind]);
}

__global__ void sinKernel(float* lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = sin(lhs[ind]);
}

__global__ void expKernel(float* lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = expf(lhs[ind]);
}

__global__ void negExpKernel(float* lhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = expf(-lhs[ind]);
}

__global__ void setToConstantKernel(float* lhs, const float c, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = c;
}

__global__ void equalKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] = rhs[ind];
}

__global__ void multiplyKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] *= rhs[ind];
}

__global__ void divideKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;
    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    const float rhs_val = rhs[ind];

    if (rhs_val == 0.0f)
        lhs[ind] = 1.0f;
    else
        lhs[ind] /= rhs_val;
}

__global__ void addKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += rhs[ind];
}

__global__ void addKernel(float* lhs, const float rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += rhs;
}

__global__ void subKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] -= rhs[ind];
}

__global__ void scaleKernel(float* lhs, const float c, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] *= c;
}

__global__ void scalarAddKernel(float* lhs, const float c, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    const uint64 ind = uint64(iz) * uint64(dim.x * dim.y) + uint64(iy * dim.x + ix);
    lhs[ind] += c*rhs[ind];
}

__global__ void sumKernel(const float* x, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[uint64(j * N.z + k)];
        }
    }
}

__global__ void innerProductKernel(const float* x, const float* y, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[uint64(j * N.z + k)] * y_slice[uint64(j * N.z + k)];
        }
    }
}

__global__ void sum_2D(const float* x, float* sum_x, int3 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N.x)
        return;

    const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
    float accum = 0.0f;
    for (int j = 0; j < N.y; j++)
    {
        for (int k = 0; k < N.z; k++)
            accum += x_slice[j * N.z + k];
    }
    sum_x[i] = accum;
}

__global__ void innerProductKernel_2D(const float* x, const float* y, float* sum_x, const int3 N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N.x)
        return;

    const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
    const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
    float accum = 0.0f;
    for (int j = 0; j < N.y; j++)
    {
        for (int k = 0; k < N.z; k++)
            accum += x_slice[j * N.z + k] * y_slice[j * N.z + k];
    }
    sum_x[i] = accum;
}

__global__ void sum_1D(const float* x, float* sum_x, int N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N; i++)
        *sum_x += x[i];
}

__global__ void weightedInnerProductKernel(const float* x, const float* w, const float* y, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        const float* x_slice = &x[uint64(i) * uint64(N.y * N.z)];
        const float* y_slice = &y[uint64(i) * uint64(N.y * N.z)];
        const float* w_slice = &w[uint64(i) * uint64(N.y * N.z)];
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x_slice[j * N.z + k] * y_slice[j * N.z + k] * w_slice[j * N.z + k];
        }
    }
}

int numberOfGPUs()
{
    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err == cudaSuccess)
        return num_gpus;
    else
        return 0;
}

float getAvailableGPUmemory(std::vector<int> whichGPUs)
{
    if (whichGPUs.size() == 0)
        return 0.0;
    else if (whichGPUs.size() == 1)
        return getAvailableGPUmemory(whichGPUs[0]);
    else
    {
        float retVal = getAvailableGPUmemory(whichGPUs[0]);
        for (int i = 1; i < int(whichGPUs.size()); i++)
            retVal = std::min(retVal, getAvailableGPUmemory(whichGPUs[i]));
        return retVal;
    }
}

float getAvailableGPUmemory(int whichGPU)
{
    if (whichGPU >= 0)
    {
        cudaSetDevice(whichGPU);
        std::size_t free_byte;
        std::size_t total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        return float(double(free_byte) / pow(2.0, 30.0));
    }
    else
        return 0.0;
}

cudaError_t setToConstant(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    setToConstantKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t equal(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    equalKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t multiply(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    multiplyKernel<<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t divide(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    divideKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t add(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    addKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t add(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    addKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t sub(float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    subKernel <<< dimGrid, dimBlock >>> (dev_lhs, dev_rhs, N);
    return cudaPeekAtLastError();
}

cudaError_t scale(float* dev_lhs, const float c, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    scaleKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, N);
    return cudaPeekAtLastError();
}

cudaError_t scalarAdd(float* dev_lhs, const float c, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    scalarAddKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, dev_rhs, N);
    return cudaPeekAtLastError();
}

extern cudaError_t replaceZeros(float* dev_lhs, const int3 N, int whichGPU, float newVal)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    replaceZerosKernel <<< dimGrid, dimBlock >>> (dev_lhs, N, newVal);
    return cudaPeekAtLastError();
}

extern cudaError_t clip(float* dev_lhs, const int3 N, int whichGPU, float clipVal)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    clipKernel <<< dimGrid, dimBlock >>> (dev_lhs, N, clipVal);
    return cudaPeekAtLastError();
}

cudaError_t cosFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    cosKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t sinFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    sinKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t expFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    expKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

cudaError_t negExpFcn(float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    negExpKernel <<< dimGrid, dimBlock >>> (dev_lhs, N);
    return cudaPeekAtLastError();
}

float sum(const float* dev_lhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 0.0;
    }
    //sumKernel<<<1,1>>>(dev_lhs, dev_sum, N);

    //*
    float* dev_sum_1D = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum_1D, N.x * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 0.0;
    }
    int blockSize = 8;
    int gridSize = int(ceil(double(N.x) / double(blockSize)));
    sum_2D <<< gridSize, blockSize >>> (dev_lhs, dev_sum_1D, N);
    sum_1D <<< 1, 1 >>> (dev_sum_1D, dev_sum, N.x);
    cudaFree(dev_sum_1D);
    //*/

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

float innerProduct(const float* dev_lhs, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 0.0;
    }
    //innerProductKernel <<<1, 1 >>> (dev_lhs, dev_rhs, dev_sum, N);

    //*
    float* dev_sum_1D = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum_1D, N.x * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 0.0;
    }
    int blockSize = 8;
    int gridSize = int(ceil(double(N.x) / double(blockSize)));
    innerProductKernel_2D <<< gridSize, blockSize >>> (dev_lhs, dev_rhs, dev_sum_1D, N);
    sum_1D <<< 1, 1 >>> (dev_sum_1D, dev_sum, N.x);
    cudaFree(dev_sum_1D);
    //*/

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

float weightedInnerProduct(const float* dev_lhs, const float* dev_w, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    cudaError_t cudaStatus;
    float* dev_sum = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_sum, 1 * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 0.0;
    }
    weightedInnerProductKernel <<<1, 1 >>> (dev_lhs, dev_w, dev_rhs, dev_sum, N);

    float retVal = 0.0;
    cudaMemcpy(&retVal, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (dev_sum != 0)
    {
        cudaFree(dev_sum);
    }
    return retVal;
}

dim3 setBlockSize(int3 N)
{
	dim3 dimBlock(8, 8, 8);  // needs to be optimized
	if (N.z < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 16;
		dimBlock.z = 1;
	}
	else if (N.y < 8)
	{
		dimBlock.x = 16;
		dimBlock.y = 1;
		dimBlock.z = 16;
	}
	else if (N.x < 8)
	{
		dimBlock.x = 1;
		dimBlock.y = 16;
		dimBlock.z = 16;
	}
	return dimBlock;
}

dim3 setGridSize(int3 N, dim3 dimBlock)
{
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))), int(ceil(double(N.z) / double(dimBlock.z))));
    return dimGrid;
}

dim3 setBlockSize(int4 N)
{
    return setBlockSize(make_int3(N.x, N.y, N.z));
}

dim3 setGridSize(int4 N, dim3 dimBlock)
{
    return setGridSize(make_int3(N.x, N.y, N.z), dimBlock);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation, bool swapFirstAndLastDimensions)
{
    int3 N = make_int3(N_txt.x, N_txt.y, N_txt.z);
    if (swapFirstAndLastDimensions)
    {
        N.x = N_txt.z;
        N.z = N_txt.x;
    }
    return loadTexture(tex_object, dev_data, N, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int4 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
    int3 N3 = make_int3(N_txt.x, N_txt.y, N_txt.z);
    return loadTexture(tex_object, dev_data, N3, useExtrapolation, useLinearInterpolation);
}

cudaArray* loadTexture(cudaTextureObject_t& tex_object, float* dev_data, const int3 N_txt, bool useExtrapolation, bool useLinearInterpolation)
{
  if (dev_data == nullptr)
    return nullptr;
  cudaArray* d_data_array = nullptr;

  // Allocate 3D array memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&d_data_array, &channelDesc, make_cudaExtent(N_txt.z, N_txt.y, N_txt.x));
 
  // Bind 3D array to texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = (cudaArray_t)d_data_array;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = false;  // Texture coordinates normalization

  if (useExtrapolation)
  {
    texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeClamp;
    texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeClamp;
    texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeClamp;
  }
  else
  {
    texDesc.addressMode[0] = (cudaTextureAddressMode)cudaAddressModeBorder;
    texDesc.addressMode[1] = (cudaTextureAddressMode)cudaAddressModeBorder;
    texDesc.addressMode[2] = (cudaTextureAddressMode)cudaAddressModeBorder;
  }

  if (useLinearInterpolation)
  {
      texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModeLinear;
  }
  else
  {
      texDesc.filterMode = (cudaTextureFilterMode)cudaFilterModePoint;
  }
  cudaCreateTextureObject(&tex_object, &resDesc, &texDesc, nullptr);

  // Update the texture memory
  cudaMemcpy3DParms cudaparams = {0};
  cudaparams.extent = make_cudaExtent(N_txt.z, N_txt.y, N_txt.x);
  cudaparams.kind = cudaMemcpyDeviceToDevice;
  cudaparams.srcPos = make_cudaPos(0, 0, 0);
  cudaparams.srcPtr = make_cudaPitchedPtr(dev_data, N_txt.z * sizeof(float), N_txt.z, N_txt.y);
  cudaparams.dstPos = make_cudaPos(0, 0, 0);
  cudaparams.dstArray = (cudaArray_t)d_data_array;
  cudaMemcpy3D(&cudaparams);
  return d_data_array;
}

float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU)
{
	cudaSetDevice(whichGPU);

    uint64 N = params->projectionData_numberOfElements();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
	if (cudaMalloc((void**)&dev_g, N * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(projection) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_g, g, N * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(projection) failed!\n");
		return NULL;
	}

	return dev_g;
}

bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;

    uint64 N = params->projectionData_numberOfElements();

	cudaStatus = cudaMemcpy(g, dev_g, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy projection data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU)
{
	cudaSetDevice(whichGPU);

    uint64 N = params->volumeData_numberOfElements();

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_f = 0;
	if (cudaMalloc((void**)&dev_f, N * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_f, f, N * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(volume) failed!\n");
		return NULL;
	}

	return dev_f;
}

bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;
    uint64 N = params->volumeData_numberOfElements();
	cudaStatus = cudaMemcpy(f, dev_f, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy volume data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copy3DdataToGPU(float* g, int3 N, int whichGPU)
{
	cudaSetDevice(whichGPU);

	uint64 N_prod = uint64(N.x) * uint64(N.y) * uint64(N.z);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy volume data to GPU
	float* dev_g = 0;
	if (cudaMalloc((void**)&dev_g, N_prod * sizeof(float)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc(volume) failed!\n");
		return NULL;
	}
	if (cudaMemcpy(dev_g, g, N_prod * sizeof(float), cudaMemcpyHostToDevice))
	{
		fprintf(stderr, "cudaMemcpy(volume) failed!\n");
		return NULL;
	}

	return dev_g;
}

bool pull3DdataFromGPU(float* g, int3 N, float* dev_g, int whichGPU)
{
	cudaSetDevice(whichGPU);
	cudaError_t cudaStatus;
    uint64 N_prod = uint64(N.x) * uint64(N.y) * uint64(N.z);
	cudaStatus = cudaMemcpy(g, dev_g, N_prod * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "failed to copy volume data back to host!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}
	else
		return true;
}

float* copyAngleArrayToGPU(parameters* params)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;
    float* dev_phis = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(phis) failed!\n");
    return dev_phis;
}

bool setProjectionGPUparams(parameters* params, int4& N, float4& T, float4& startVals, bool doNormalize)
{
    if (params == NULL)
        return false;
    else
    {
        N.x = params->numAngles; N.y = params->numRows; N.z = params->numCols;
        T.x = params->T_phi(); T.y = params->pixelHeight; T.z = params->pixelWidth;
        startVals.x = params->phi_0(); startVals.y = params->v_0(); startVals.z = params->u_0();

        if (params->geometry == parameters::CONE)
        {
            N.w = params->numAngles;
            T.w = params->helicalPitch;
            startVals.w = params->z_source_offset;
        }
        else
        {
            N.w = params->numAngles;
            T.w = 0.0;
            startVals.w = 0.0;
        }

        if (params->geometry == parameters::CONE && params->detectorType == parameters::CURVED)
        {
            T.z = atan(T.z / params->sdd);
            if (doNormalize)
            {
                T.y = T.y / params->sdd;
                startVals.y = startVals.y / params->sdd;
            }
        }
        else if (doNormalize)
        {
            if (params->geometry == parameters::CONE)
            {
                T.y = T.y / params->sdd;
                startVals.y = startVals.y / params->sdd;
                
                T.z = T.z / params->sdd;
                startVals.z = startVals.z / params->sdd;
            }
            else if (params->geometry == parameters::FAN)
            {
                T.z = T.z / params->sdd;
                startVals.z = startVals.z / params->sdd;
            }
        }

        return true;
    }
}

bool setVolumeGPUparams(parameters* params, int4& N, float4& T, float4& startVals)
{
    if (params == NULL)
        return false;
    else
    {
        N.x = params->numX; N.y = params->numY; N.z = params->numZ;
        T.x = params->voxelWidth; T.y = params->voxelWidth; T.z = params->voxelHeight;
        startVals.x = params->x_0(); startVals.y = params->y_0(); startVals.z = params->z_0();
        return true;
    }
}

bool windowFOV_gpu(float* f, parameters* params)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float rFOVsq = params->rFOV() * params->rFOV();

    int4 N; float4 T; float4 startVal;
    setVolumeGPUparams(params, N, T, startVal);

    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid = setGridSize(N, dimBlock);
    windowFOVKernel <<< dimGrid, dimBlock >>> (f, N, T, startVal, rFOVsq, params->volumeDimensionOrder);
    cudaStatus = cudaDeviceSynchronize();

    return true;
}
