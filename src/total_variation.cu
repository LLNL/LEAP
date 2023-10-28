#include "total_variation.cuh"

#include <math.h>

#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USE_P_HUBER // currently using p=1.2

__forceinline__ __device__ float square(const float x)
{
    return x * x;
}

__forceinline__ __device__ float Huber(const float x, const float delta)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= delta)
        return 0.5f * x * x;
    else
        return delta * delta * (5.0f * pow(fabs(x / delta), 1.2f) - 2.0f) / 6.0f;
#else
    if (fabs(x) <= delta)
        return 0.5f * x * x;
    else
        return delta * (fabs(x) - 0.5f * delta);
#endif
}

__forceinline__ __device__ float DHuber(const float x, const float delta)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= delta)
        return x;
    else
        return x * pow(fabs(x / delta), -0.8f);
    //return delta * pow(fabs(x/delta),0.2f);
#else
    if (fabs(x) <= delta)
        return x;
    else
        return (x > 0.0f) ? delta : -delta;
#endif
}

__forceinline__ __device__ float DDHuber(const float x, const float delta)
{
#ifdef USE_P_HUBER
    if (fabs(x) <= delta)
        return  1.0;
    else
        return pow(fabs(x / delta), -0.8f);
#else
    return delta / max(delta, fabs(x));
#endif
}

__device__ float aTV_Huber_costTerm(float* f, const int i, const int j, const int k, int3 N, float delta, float beta)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];

    return (Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) +
        Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) +
        Huber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) +
        Huber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta)) *
        dist_1 +
        (Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.z + j_plus * N.z + k], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_2 +
        (Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            Huber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_3;
}

__global__ void aTV_Huber_cost(float* f, float* d, int3 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    d[i * N.y * N.z + j * N.z + k] = aTV_Huber_costTerm(f, i, j, k, N, delta, beta);
}

__device__ float aTV_Huber_quadFormTerm(float* f, float* d, const int i, const int j, const int k, int3 N, float delta,
    float beta)
{
    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];
    const float curVal_d = d[i * N.y * N.z + j * N.z + k];

    return (DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) *
        square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k]) +
        DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) *
        square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) *
        square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) *
        square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k]) +
        DDHuber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) *
        square(curVal_d - d[i * N.y * N.z + j * N.z + k_plus]) +
        DDHuber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta) *
        square(curVal_d - d[i * N.y * N.z + j * N.z + k_minus])) *
        dist_1 +
        (DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j * N.z + k_minus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i * N.y * N.z + j_minus * N.z + k_minus])) *
        dist_2 +
        (DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i_plus * N.y * N.z + j_minus * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_plus * N.z + k_minus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k_plus]) +
            DDHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta) *
            square(curVal_d - d[i_minus * N.y * N.z + j_minus * N.z + k_minus])) *
        dist_3;
}

__global__ void aTV_Huber_quadForm(float* f, float* d, float* quad, int3 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    quad[i * N.y * N.z + j * N.z + k] = aTV_Huber_quadFormTerm(f, d, i, j, k, N, delta, beta);
}

__global__ void aTV_Huber_gradient(float* f, float* Df, int3 N, float delta, float beta)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int i_minus = max(0, i - 1);
    const int i_plus = min(N.x - 1, i + 1);
    const int j_minus = max(0, j - 1);
    const int j_plus = min(N.y - 1, j + 1);
    const int k_minus = max(0, k - 1);
    const int k_plus = min(N.z - 1, k + 1);

    const float dist_1 = 1.0f * beta;                 // 1/sqrt(1)
    const float dist_2 = 0.7071067811865475f * beta;  // 1/sqrt(2)
    const float dist_3 = 0.5773502691896258f * beta;  // 1/sqrt(3)

    const float curVal = f[i * N.y * N.z + j * N.z + k];

    // dist 1: 6
    // dist 2: 12
    // dist 3: 8
    Df[i * N.y * N.z + j * N.z + k] = (DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k], delta) +
        DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k], delta) +
        DHuber(curVal - f[i * N.y * N.z + j * N.z + k_plus], delta) +
        DHuber(curVal - f[i * N.y * N.z + j * N.z + k_minus], delta)) *
        dist_1 +
        (DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j * N.z + k_minus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_2 +
        (DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_plus * N.y * N.z + j_minus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_plus * N.z + k_minus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_plus], delta) +
            DHuber(curVal - f[i_minus * N.y * N.z + j_minus * N.z + k_minus], delta)) *
        dist_3;
}

bool anisotropicTotalVariation_gradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta, int whichGPU)
{
    if (f == NULL) return false;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = 1.0e-8;

    cudaSetDevice(whichGPU);

    int3 N = make_int3(N_1, N_2, N_3);

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, N, whichGPU);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_gradient <<< dimGrid, dimBlock >>> (dev_f, dev_Df, N, delta, beta);
    cudaDeviceSynchronize();

    // pull result off GPU
    if (Df == NULL) Df = f;
    pull3DdataFromGPU(Df, N, dev_Df, whichGPU);

    // Clean up
    if (dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }
    return true;
}

float anisotropicTotalVariation_quadraticForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta, int whichGPU)
{
    if (f == NULL || d == NULL) return -1.0;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = 1.0e-8;

    cudaSetDevice(whichGPU);

    int3 N = make_int3(N_1, N_2, N_3);

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, N, whichGPU);

    // Copy step direction to GPU
    float* dev_d = 0;
    dev_d = copy3DdataToGPU(d, N, whichGPU);

    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_quad = 0;
    if (cudaMalloc((void**)&dev_quad, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_quadForm <<< dimGrid, dimBlock >>> (dev_f, dev_d, dev_quad, N, delta, beta);
    cudaDeviceSynchronize();

    // pull result off GPU
    float* quadTerms = (float*)malloc(sizeof(float) * N.x * N.y * N.z);
    pull3DdataFromGPU(quadTerms, N, dev_quad, whichGPU);
    float retVal = 0.0;
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) retVal += quadTerms[i * N.y * N.z + j * N.z + k];
        }
    }
    free(quadTerms);

    // Clean up
    if (dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (dev_d != 0)
    {
        cudaFree(dev_d);
    }
    if (dev_quad != 0)
    {
        cudaFree(dev_quad);
    }

    return retVal;
}

float anisotropicTotalVariation_cost(float* f, int N_1, int N_2, int N_3, float delta, float beta, int whichGPU)
{
    if (f == NULL) return -1.0;
    if (beta <= 0.0)
        beta = 1.0;
    if (delta < 1.0e-8)
        delta = 1.0e-8;

    cudaSetDevice(whichGPU);

    int3 N = make_int3(N_1, N_2, N_3);

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, N, whichGPU);

    // Allocate space on GPU for the un-collapsed quadratic form
    float* dev_d = 0;
    if (cudaMalloc((void**)&dev_d, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
        return -1.0;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
        int(ceil(double(N.z) / double(dimBlock.z))));
    aTV_Huber_cost <<< dimGrid, dimBlock >>> (dev_f, dev_d, N, delta, beta);
    cudaDeviceSynchronize();

    // pull result off GPU
    float* costTerms = (float*)malloc(sizeof(float) * N.x * N.y * N.z);
    pull3DdataFromGPU(costTerms, N, dev_d, whichGPU);
    float retVal = 0.0;
    for (int i = 0; i < N.x; i++)
    {
        for (int j = 0; j < N.y; j++)
        {
            for (int k = 0; k < N.z; k++) retVal += costTerms[i * N.y * N.z + j * N.z + k];
        }
    }
    free(costTerms);

    // Clean up
    if (dev_f != 0)
    {
        cudaFree(dev_f);
    }
    if (dev_d != 0)
    {
        cudaFree(dev_d);
    }

    return retVal;
}