#include "noise_filters.cuh"

#include <math.h>

#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 8

__global__ void medianFilterKernel(float* f, float* f_filtered, int3 N, float threshold)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    float v[27];
    int ind = 0;
    for (int di = -1; di <= 1; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -1; dj <= 1; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -1; dk <= 1; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));
                v[ind] = f[i_shift * N.y * N.z + j_shift * N.z + k_shift];
                ind += 1;
            }
        }
    }
    const float curVal = v[13];

    // bubble-sort for first 14 samples
    for (int i = 0; i < 14; i++)
    {
        for (int j = i + 1; j < 27; j++)
        {
            if (v[i] > v[j])
            {  // swap?
                const float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // fabs(curVal-v[13])/fabs(curVal) > threshold
    if (fabs(curVal - v[13]) >= threshold * fabs(v[13]))
        f_filtered[i * N.y * N.z + j * N.z + k] = v[13];
    else
        f_filtered[i * N.y * N.z + j * N.z + k] = curVal;
}

__global__ void BlurFilterKernel(float* f, float* f_filtered, int3 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const int pixelRadius = int(floor(FWHM));
    const float denom = 1.0f / FWHM;

    float val = 0.0;
    float sum = 0.0;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));
        for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
        {
            const int j_shift = max(0, min(j + dj, N.y - 1));
            for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
            {
                const int k_shift = max(0, min(k + dk, N.z - 1));

                //const float arg = min(sqrtf(float(di*di + dj*dj + dk*dk)) * denom, FWHM);
                //const float theWeight = 0.5f + 0.5f * cospif(arg);
                const float theWeight = 0.5f +
                    0.5f * cosf(3.141592653589793f* min(sqrtf(float(di * di + dj * dj + dk * dk)) * denom, 1.0f));

                if (theWeight > 0.0001f)
                {
                    val += theWeight * f[i_shift * N.y * N.z + j_shift * N.z + k_shift];
                    sum += theWeight;
                }
            }
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void BlurFilter2DKernel(float* f, float* f_filtered, int3 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    const float denom = 1.0f / (2.0f * sigma * sigma);

    float val = 0.0f;
    float sum = 0.0f;

    for (int dj = -pixelRadius; dj <= pixelRadius; dj++)
    {
        const int j_shift = max(0, min(j + dj, N.y - 1));
        const float j_dist_sq = float((j - j_shift) * (j - j_shift));
        for (int dk = -pixelRadius; dk <= pixelRadius; dk++)
        {
            const int k_shift = max(0, min(k + dk, N.z - 1));
            const float k_dist_sq = float((k - k_shift) * (k - k_shift));

            const float theWeight = exp(-denom * (j_dist_sq + k_dist_sq));

            val += theWeight * f[i * N.y * N.z + j_shift * N.z + k_shift];
            sum += theWeight;
        }
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

__global__ void BlurFilter1DKernel(float* f, float* f_filtered, int3 N, float FWHM)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || j >= N.y || k >= N.z) return;

    const float sigma = FWHM / (2.0f * sqrt(2.0f * log(2.0f)));
    // FWHM = 2*sqrt(2*log(2))*sigma
    const int pixelRadius = int(ceil(sqrt(2.0f * log(10.0f)) * sigma));
    const float denom = 1.0f / (2.0f * sigma * sigma);

    float val = 0.0;
    float sum = 0.0;
    for (int di = -pixelRadius; di <= pixelRadius; di++)
    {
        const int i_shift = max(0, min(i + di, N.x - 1));

        const float theWeight = exp(-denom * float((i - i_shift) * (i - i_shift)));

        val += theWeight * f[i_shift * N.y * N.z + j * N.z + k];
        sum += theWeight;
    }

    f_filtered[i * N.y * N.z + j * N.z + k] = val / sum;
}

bool blurFilter(float* f, int N_1, int N_2, int N_3, float FWHM, int numDims, int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    int3 N = make_int3(N_1, N_2, N_3);

    // Copy volume to GPU
    float* dev_f = 0;
    dev_f = copy3DdataToGPU(f, N, whichGPU);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    if (numDims == 1)
        BlurFilter1DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else if (numDims == 2)
        BlurFilter2DKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);
    else
        BlurFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, FWHM);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // pull result off GPU
    // if (f_filtered == NULL)
    //    f_filtered = f;
    // pullVolumeDataFromGPU(f_filtered, N, dev_Df, whichGPU);
    pull3DdataFromGPU(f, N, dev_Df, whichGPU);

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

bool medianFilter(float* f, int N_1, int N_2, int N_3, float threshold, int whichGPU)
{
    if (f == NULL) return false;

    cudaSetDevice(whichGPU);

    int3 N = make_int3(N_1, N_2, N_3);

    // Copy volume to GPU
    float* dev_f = copy3DdataToGPU(f, N, whichGPU);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, N.x * N.y * N.z * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }

    // Call kernel
    dim3 dimBlock = setBlockSize(N);
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))),
                 int(ceil(double(N.z) / double(dimBlock.z))));
    medianFilterKernel<<<dimGrid, dimBlock>>>(dev_f, dev_Df, N, threshold);
    // medianFilterKernel(float* f, float* f_filtered, int4 N, float threshold)

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // pull result off GPU
    pull3DdataFromGPU(f, N, dev_Df, whichGPU);

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

dim3 setBlockSize(int4 N)
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
