#include "cuda_utils.h"
#include "cuda_runtime.h"

__global__ void setToConstantKernel(float* lhs, const float c, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] = c;
}

__global__ void multiplyKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] *= rhs[iz * dim.x * dim.y + iy * dim.x + ix];
}

__global__ void divideKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;
    const float rhs_val = rhs[iz * dim.x * dim.y + iy * dim.x + ix];

    if (rhs_val == 0.0f)
        lhs[iz * dim.x * dim.y + iy * dim.x + ix] = 1.0f;
    else
        lhs[iz * dim.x * dim.y + iy * dim.x + ix] *= rhs_val;
}

__global__ void addKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] += rhs[iz * dim.x * dim.y + iy * dim.x + ix];
}

__global__ void addKernel(float* lhs, const float rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] += rhs;
}

__global__ void subKernel(float* lhs, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] -= rhs[iz * dim.x * dim.y + iy * dim.x + ix];
}

__global__ void scalarAddKernel(float* lhs, const float c, const float* rhs, const int3 dim)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    lhs[iz * dim.x * dim.y + iy * dim.x + ix] += c*rhs[iz * dim.x * dim.y + iy * dim.x + ix];
}

__global__ void sumKernel(const float* x, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x[i * N.y * N.z + j * N.z + k];
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
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x[i * N.y * N.z + j * N.z + k] * y[i * N.y * N.z + j * N.z + k];
        }
    }
}

__global__ void weightedInnerProductKernel(const float* x, const float* w, const float* y, float* sum_x, const int3 N)
{
    if (threadIdx.x > 0)
        return;

    *sum_x = 0.0f;
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) *sum_x += x[i * N.y * N.z + j * N.z + k] * y[i * N.y * N.z + j * N.z + k] * w[i * N.y * N.z + j * N.z + k];
        }
    }
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

cudaError_t scalarAdd(float* dev_lhs, const float c, const float* dev_rhs, const int3 N, int whichGPU)
{
    cudaSetDevice(whichGPU);
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    scalarAddKernel <<< dimGrid, dimBlock >>> (dev_lhs, c, dev_rhs, N);
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
    sumKernel<<<1,1>>>(dev_lhs, dev_sum, N);

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
    innerProductKernel <<<1, 1 >>> (dev_lhs, dev_rhs, dev_sum, N);

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
