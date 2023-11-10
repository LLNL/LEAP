
#include "ramp_filter.cuh"
#include "ramp_filter_cpu.h"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_utils.h"

#define INCLUDE_CUFFT
#ifndef PI
#define PI 3.141592653589793
#endif

#ifdef INCLUDE_CUFFT
#include <cufft.h>


__global__ void multiply2DRampFilterKernel(cufftComplex* F, const float* H, int3 N)
{
    // int k = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > 0) return;
    // if (k > N.z - 1)
    //	return;
    for (int j = 0; j < N.y; j++)
    {
        for (int i = 0; i < N.x; i++)
        {
            F[k * N.x * N.y + j * N.x + i].x *= H[j * N.x + i];
            F[k * N.x * N.y + j * N.x + i].y *= H[j * N.x + i];
        }
    }
}

__global__ void setPaddedDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView, int numExtrapolate)
{
    int j = threadIdx.x;
    int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[(i - startView) * N_pad * N.y + j * N_pad];
    float* data_block = &data[i * N.z * N.y + j * N.z];
    for (int k = 0; k < N.z; k++)
        data_padded_block[k] = data_block[k];

    for (int k = N.z; k < N_pad; k++)
        data_padded_block[k] = 0.0;

    if (numExtrapolate > 0)
    {
        const float leftVal = data_block[0];
        const float rightVal = data_block[N.z - 1];
        for (int k = N.z; k < N.z + numExtrapolate; k++)
            data_padded_block[k] = rightVal;
        for (int k = N_pad - numExtrapolate; k < N_pad; k++)
            data_padded_block[k] = leftVal;
    }
}

__global__ void multiplyRampFilterKernel(cufftComplex* G, const float* H, int3 N)
{
    int j = threadIdx.x;
    int i = blockIdx.x;
    if (i > N.x - 1 || j > N.y - 1)
        return;
    for (int k = 0; k < N.z; k++)
    {
        G[i * N.y * N.z + j * N.z + k].x *= H[k];
        G[i * N.y * N.z + j * N.z + k].y *= H[k];
    }
}

__global__ void setFilteredDataKernel(float* data_padded, float* data, int3 N, int N_pad, int startView, int endView)
{
    int j = threadIdx.x;
    int i = blockIdx.x + startView;
    if (i > endView || j > N.y - 1)
        return;
    float* data_padded_block = &data_padded[(i - startView) * N_pad * N.y + j * N_pad];
    float* data_block = &data[i * N.z * N.y + j * N.z];
    for (int k = 0; k < N.z; k++)
        data_block[k] = data_padded_block[k];
}

float* rampFilterFrequencyResponseMagnitude(int N, parameters* params)
{
    float T = params->pixelWidth;
    bool isCurved = false;
    if (params->geometry == parameters::FAN || params->geometry == parameters::CONE)
    {
        T *= params->sod / params->sdd;
        if (params->detectorType == parameters::CURVED)
            isCurved = true;
    }

    int rampID = 2;

    cudaError_t cudaStatus;
    double* h_d = rampImpulseResponse(N, T, rampID);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = h_d[i];

        if (i != 0 && isCurved == true)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= temp * temp;
        }
    }
    delete[] h_d;

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N, CUFFT_R2C, 1))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft");
        return NULL;
    }

    float* dev_h = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_h, N * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        return NULL;
    }
    cudaStatus = cudaMemcpy(dev_h, h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Make data for the result of the FFT
    int N_over2 = N / 2 + 1;
    cufftComplex* dev_H = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_H, N_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of ramp filter) failed!\n");
        return NULL;
    }

    // FFT
    result = cufftExecR2C(forward_plan, (cufftReal*)dev_h, dev_H);
    cudaDeviceSynchronize();

    // get result
    cufftComplex* H_ramp = new cufftComplex[N_over2];
    cudaStatus = cudaMemcpy(H_ramp, dev_H, N_over2 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    float* H_real = new float[N_over2];
    for (int i = 0; i < N_over2; i++)
    {
        H_real[i] = H_ramp[i].x / float(N);
    }

    // Clean up
    cufftDestroy(forward_plan);
    cudaFree(dev_h);
    cudaFree(dev_H);
    delete[] h;
    delete[] H_ramp;

    return H_real;
}

bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu, float scalar)
{
    bool retVal = true;
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    if (cpu_to_gpu)
    {
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    }
    else
    {
        dev_g = g;
    }

    // PUT CODE HERE
    int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    int N_H_over2 = N_H / 2 + 1;
    float* H_real = rampFilterFrequencyResponseMagnitude(N_H, params);
    if (scalar != 1.0)
    {
        for (int i = 0; i < N_H_over2; i++)
            H_real[i] *= scalar;
    }

    //int N_viewChunk = params->numAngles;
    int N_viewChunk = params->numAngles / 40; // number of views in a chunk (needs to be optimized)
    int numChunks = int(ceil(double(params->numAngles) / double(N_viewChunk)));

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&forward_plan, N_H, CUFFT_R2C, N_viewChunk * params->numRows))
    {
        fprintf(stderr, "Failed to plan 1d r2c fft (size %d)\n", N_H);
        return false;
    }
    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan1d(&backward_plan, N_H, CUFFT_C2R, N_viewChunk * params->numRows)) // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 1d c2r ifft\n");
        return false;
    }
    //return true;

    float* dev_g_pad = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_g_pad, N_viewChunk * params->numRows * N_H * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded projection data) failed!\n");
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_G = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_G, N_viewChunk * params->numRows * N_H_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded projection data) failed!\n");
        retVal = false;
    }

    // Copy filter to device
    float* dev_H = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_H, N_H_over2 * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (H_real == NULL)
        printf("H_real is NULL!!!\n");
    cudaStatus = cudaMemcpy(dev_H, H_real, N_H_over2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "cudaMemcpy(H) failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }
    int3 dataSize; dataSize.x = N_viewChunk; dataSize.y = params->numRows; dataSize.z = N_H_over2;
    int3 origSize; origSize.x = params->numAngles; origSize.y = params->numRows; origSize.z = params->numCols;

    int numExtrapolate = 0;

    if (retVal == true)
    {
        for (int iChunk = 0; iChunk < numChunks; iChunk++)
        {
            int startView = iChunk * N_viewChunk;
            int endView = min(params->numAngles - 1, startView + N_viewChunk - 1);

            setPaddedDataKernel <<< endView - startView + 1, params->numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView, numExtrapolate);
            cudaDeviceSynchronize();

            // FFT
            result = cufftExecR2C(forward_plan, (cufftReal*)dev_g_pad, dev_G);

            // Multiply Filter
            multiplyRampFilterKernel <<< N_viewChunk, params->numRows >>> (dev_G, dev_H, dataSize);
            cudaDeviceSynchronize();

            // IFFT
            result = cufftExecC2R(backward_plan, (cufftComplex*)dev_G, (cufftReal*)dev_g_pad);

            setFilteredDataKernel <<< endView - startView + 1, params->numRows >>> (dev_g_pad, dev_g, origSize, N_H, startView, endView);
            cudaDeviceSynchronize();
        }

        if (cpu_to_gpu)
        {
            // Copy result back to host
            cudaStatus = cudaMemcpy(g, dev_g, params->numAngles * params->numRows * params->numCols * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            }
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_g_pad);
    if (cpu_to_gpu)
        cudaFree(dev_g);
    cudaFree(dev_H);
    cudaFree(dev_G);
    delete[] H_real;

    return retVal;
}

bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu)
{
    if (cpu_to_gpu == false)
    {
        printf("Error: current implementation of rampFilter2D requires that data reside on the CPU\n");
        return false;
    }

    int N_x = params->numX;
    int N_y = params->numY;
    int N_z = params->numZ;

    // Pad and then find next power of 2
    int N_H1 = int(pow(2.0, ceil(log2(2 * max(N_y, N_x)))));
    int N_H2 = N_H1;
    int N_H2_over2 = N_H2 / 2 + 1;

    cudaSetDevice(params->whichGPU);
    bool retVal = true;

    int smoothingLevel = 0;

    // Make cuFFT Plans
    cufftResult result;
    cufftHandle forward_plan;
    if (CUFFT_SUCCESS != cufftPlan2d(&forward_plan, N_H1, N_H2, CUFFT_R2C))
    {
        fprintf(stderr, "Failed to plan 2d r2c fft");
        return false;
    }
    cufftHandle backward_plan;
    if (CUFFT_SUCCESS != cufftPlan2d(&backward_plan, N_H1, N_H2, CUFFT_C2R))  // do I use N_H_over2?
    {
        fprintf(stderr, "Failed to plan 2d c2r ifft");
        return false;
    }

    float* paddedSlice = (float*)malloc(sizeof(float) * N_H1 * N_H2);
    // Make zero-padded array, copy data to 1st half of array and set remaining slots to zero
    cudaError_t cudaStatus;
    float* dev_f_pad = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_f_pad, N_H1 * N_H2 * sizeof(float)))
    {
        fprintf(stderr, "cudaMalloc(padded volume data) failed!\n");
        retVal = false;
    }

    // Make data for the result of the FFT
    cufftComplex* dev_F = 0;
    if (cudaStatus = cudaMalloc((void**)&dev_F, N_H1 * N_H2_over2 * sizeof(cufftComplex)))
    {
        fprintf(stderr, "cudaMalloc(Fourier transform of padded volume data) failed!\n");
        retVal = false;
    }

    // Copy filter to device
    float* H = rampFrequencyResponse2D(N_H1, 1.0, 1.0, smoothingLevel);  // FIXME?
    float* dev_H = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_H, N_H1 * N_H2_over2 * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    cudaStatus = cudaMemcpy(dev_H, H, N_H1 * N_H2_over2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
    {
        fprintf(stderr, "cudaMemcpy(H) failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    for (int k = 0; k < N_z; k++)
    {
        if (params->volumeDimensionOrder == parameters::XYZ)
        {
            for (int j = 0; j < N_H1; j++)
            {
                int j_source = j;
                if (j >= N_y)
                {
                    if (j - N_y < N_H1 - j)
                        j_source = N_y - 1;
                    else
                        j_source = 0;
                }
                for (int i = 0; i < N_H2; i++)
                {
                    int i_source = i;
                    if (i >= N_x)
                    {
                        if (i - N_x < N_H2 - i)
                            i_source = N_x - 1;
                        else
                            i_source = 0;
                    }
                    paddedSlice[j * N_H2 + i] = f[i_source * N_y * N_z + j_source*N_z + k];
                }
            }
        }
        else //if (params->volumeDimensionOrder == parameters::ZYX)
        {
            float* f_slice = &f[k * N_x * N_y];
            for (int j = 0; j < N_H1; j++)
            {
                int j_source = j;
                if (j >= N_y)
                {
                    if (j - N_y < N_H1 - j)
                        j_source = N_y - 1;
                    else
                        j_source = 0;
                }
                for (int i = 0; i < N_H2; i++)
                {
                    int i_source = i;
                    if (i >= N_x)
                    {
                        if (i - N_x < N_H2 - i)
                            i_source = N_x - 1;
                        else
                            i_source = 0;
                    }
                    paddedSlice[j * N_H2 + i] = f_slice[j_source * N_x + i_source];
                }
            }
        }
        if (cudaMemcpy(dev_f_pad, paddedSlice, N_H1 * N_H2 * sizeof(float), cudaMemcpyHostToDevice))
        {
            fprintf(stderr, "cudaMemcpy(padded volume data) failed!\n");
            retVal = false;
        }

        // FFT
        result = cufftExecR2C(forward_plan, (cufftReal*)dev_f_pad, dev_F);

        // Multiply Filter
        int3 dataSize;
        dataSize.z = N_z;
        dataSize.y = N_H1;
        dataSize.x = N_H2_over2;
        multiply2DRampFilterKernel<<<1, 1>>>(dev_F, dev_H, dataSize);

        // IFFT
        result = cufftExecC2R(backward_plan, (cufftComplex*)dev_F, (cufftReal*)dev_f_pad);

        // Copy result back to host
        if (retVal)
        {
            cudaStatus = cudaMemcpy(paddedSlice, dev_f_pad, N_H1 * N_H2 * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStatus)
            {
                fprintf(stderr, "failed to copy result back to host!\n");
                fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
                fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
                retVal = false;
            }
            if (params->volumeDimensionOrder == parameters::XYZ)
            {
                for (int j = 0; j < N_y; j++)
                {
                    for (int i = 0; i < N_x; i++)
                    {
                        f[i * N_y * N_z + j * N_z + k] = paddedSlice[j * N_H2 + i] / float(N_H1 * N_H2);
                    }
                }
            }
            else
            {
                float* f_slice = &f[k * N_x * N_y];
                for (int j = 0; j < N_y; j++)
                {
                    for (int i = 0; i < N_x; i++)
                    {
                        f_slice[j * N_x + i] = paddedSlice[j * N_H2 + i] / float(N_H1 * N_H2);
                    }
                }
            }
        }
    }

    // Clean up
    cufftDestroy(forward_plan);
    cufftDestroy(backward_plan);
    cudaFree(dev_f_pad);
    cudaFree(dev_H);
    cudaFree(dev_F);
    free(H);
    free(paddedSlice);

    return retVal;
}
#else
bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu)
{
    //printf("CUFFT libraries not available!\n");
    //return false;
    return rampFilter1D_cpu(g, params);
}

bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu)
{
    printf("CUFFT libraries not available!\n");
    return false;
}
#endif
