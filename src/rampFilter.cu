
#include "rampFilter.cuh"

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define INCLUDE_CUFFT
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

float* rampFilterFrequencyResponseMagnitude(int N, double T, int rampID)
{
    cudaError_t cudaStatus;
    double* h_d = rampImpulseResponse(N, T, rampID);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
        h[i] = h_d[i];
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

bool rampFilter1D(float*& g, parameters* params, bool cpu_to_gpu)
{
    bool retVal = true;
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->numAngles * params->numRows * params->numCols * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
            return false;
        }
        if ((cudaStatus = cudaMemcpy(dev_g, g, params->numAngles * params->numRows * params->numCols * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy(projection) failed!\n");
            return false;
        }
    }
    else
    {
        dev_g = g;
    }

    // PUT CODE HERE
    int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    float* H_real = rampFilterFrequencyResponseMagnitude(N_H, params->pixelWidth);

    //digitalFilters filterLib;
    //float* H_real = filterLib.rampFilterFrequencyResponseMagnitude(N_H, g->getParallelRayWidth(), g->cfg->rampID.value);

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
    int N_H_over2 = N_H / 2 + 1;
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
bool rampFilter1D(float*& f, parameters* params, bool cpu_to_gpu)
{
    printf("CUFFT libraries not available!\n");
    return false;
}

bool rampFilter2D(float*& f, parameters* params, bool cpu_to_gpu)
{
    printf("CUFFT libraries not available!\n");
    return false;
}
#endif

float* rampFrequencyResponse2D(int N, double T, double scalingFactor, int smoothingLevel)
{
    int N_over2 = N / 2 + 1;
    float* H_2D = (float*)malloc(sizeof(float) * N * N_over2);
    float* H = rampFrequencyResponse(N, T);

    double freqResponseAtNyquist = 2.0 * sin(0.5 * PI);
    double c = T / (freqResponseAtNyquist);
    double T_X = 2.0 * PI / double(N);

    for (int i = 0; i < N; i++)
    {
        double Hx_squared = H[i] * H[i];
        double X = frequencySamples(i, N, T_X);
        for (int j = 0; j < N_over2; j++)
        {
            double Y = frequencySamples(j, N, T_X);
            double Hy_squared = H[j] * H[j];
            double temp = Hx_squared + Hy_squared - c * c * Hx_squared * Hy_squared;
            H_2D[i * N_over2 + j] = sqrt(std::max(0.0, temp)) * scalingFactor;
            if (smoothingLevel > 0)  // 0.5 + 0.5*cos(X) = cos(0.5*X)^2
                H_2D[i * N_over2 + j] *=
                    pow(cos(0.5 * X), 2.0 * float(smoothingLevel)) * pow(cos(0.5 * Y), 2 * float(smoothingLevel));
            // H_2D[i * N_over2 + j] *= (0.5 + 0.5 * cos(X)) * (0.5 + 0.5 * cos(Y));
        }
    }

    free(H);
    return H_2D;
}

float* rampFrequencyResponse(int N, double T)
{
    float* H = (float*)malloc(sizeof(float) * N);

    double T_X = 2.0 * PI / double(N);
    for (int i = 0; i < N; i++) H[i] = rampFrequencyResponse(frequencySamples(i, N, T_X), T);

    return H;
}

double rampFrequencyResponse(double X, double T)
{
    return 2.0 * sin(0.5 * fabs(X)) / T;
}

double frequencySamples(int i, int N, double T)
{
    // samples lie in [-pi, pi)
    if (i < N / 2)
        return double(i) * T;
    else
        return double(i - N) * T;
}

double timeSamples(int i, int N)
{
    if (i < N / 2)
        return double(i);
    else
        return double(i) - double(N);
}

double rampImpulseResponse(int N, double T, int n, int rampID)
{
    double retVal = 0.0;
    double s = timeSamples(n, N);

    double s_sq = s * s;
    switch (rampID)
    {
    case 0:  // Blurred Shepp-Logan, FWHM 2.1325 samples
        retVal = 1.0 / (PI * (0.25 - s_sq)) * (0.75 - s_sq) / (2.25 - s_sq);
        break;
    case 1: // Cosine Filter, not a very good impulse response, FWHM 1.8487 samples
        retVal = (PI * pow(-1.0, s) / (0.25 - s_sq) - (2.0 * s_sq + 0.5) / ((s_sq - 0.25) * (s_sq - 0.25))) / (2.0 * PI);
        break;
    case 2: // Shepp-Logan, FWHM 1.0949(1.2907) samples
        retVal = 1.0 / (PI * (0.25 - s_sq));
        break;
    case 3:
    case 4: // Shepp-Logan with 4th order finite difference, FWHM 1.0518(1.2550) samples
        retVal = 1.0 / (PI * (0.25 - s_sq)) * (2.5 - s_sq) / (2.25 - s_sq);
        break;
    case 5:
    case 6: // Shepp-Logan with 6th order finite difference, FWHM 1.0353(1.2406) samples
        retVal = 1.0 / (PI * (0.25 - s_sq)) * (s_sq * s_sq - 35.0 / 4.0 * s_sq + 259.0 / 16.0) / ((25.0 / 4.0 - s_sq) * (9.0 / 4.0 - s_sq));
        break;
    case 7:
    case 8: // Shepp-Logan with 8th order finite difference, FWHM 1.0266(1.2328) samples
        retVal = 1.0 / (PI * (0.25 - s_sq)) * (s_sq * s_sq * s_sq - 336.0 / 16.0 * s_sq * s_sq + 1974.0 / 16.0 * s_sq - 3229.0 / 16.0) / ((s_sq - 49.0 / 4.0) * (s_sq - 25.0 / 4.0) * (s_sq - 9.0 / 4.0));
        break;
    case 9:
    case 10: // Shepp-Logan with 10th order finite difference, FWHM 1.0214(1.2280) samples
        retVal = 1.0 / (PI * (0.25 - s_sq)) * (s_sq * s_sq * s_sq * s_sq - 165.0 / 4.0 * s_sq * s_sq * s_sq + 4389.0 / 8.0 * s_sq * s_sq - 86405.0 / 32.0 * s_sq + 1057221.0 / 256.0) / ((s_sq - 81.0 / 4.0) * (s_sq - 49.0 / 4.0) * (s_sq - 25.0 / 4.0) * (s_sq - 9.0 / 4.0));
        break;
    default: // Ram-Lak, the "exact" ramp filter, FWHM 1.0000(1.2067) samples
        if (s == 0.0)
            retVal = PI / 2.0;
        else
            retVal = (pow(-1.0, s) - 1.0) / (PI * s_sq);
    }
    retVal = retVal / T;
    return retVal;
}

double* rampImpulseResponse(int N, double T, int rampID)
{
    double* h = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++)
        h[i] = rampImpulseResponse(N, T, i, rampID);
    return h;
}
