////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for calculating system matrix values
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "system_matrix.cuh"
#include "log.h"

__global__ void systemMatrixKernel_parallel(float* A, short* indices, const int* iCols, const int numCols, const float cos_phi, const float sin_phi, const int4 N_g, const float4 T_g, const float4 startVals_g, const int4 N_f, const float4 T_f, const float4 startVals_f, const int N_max, const int iAngle)
{
    const int N_xy = max(N_f.x, N_f.y);
    const int iCol_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const int ivox = threadIdx.y + blockIdx.y * blockDim.y;
    if (iCol_thread >= numCols || ivox >= N_xy)
        return;

    const int iCol = iCols[iCol_thread];

    const float T_x_inv = 1.0f / T_f.x;
    const float T_u_inv = 1.0f / T_g.z;
    const float u_lo = float(iCol) - 0.5f;
    const float u_hi = float(iCol) + 0.5f;

    const float u = (float)iCol * T_g.z + startVals_g.z;

    const float l_phi = T_f.x / max(fabs(cos_phi), fabs(sin_phi));
    //N_max = N = N_xy * footprint_row * footprint_col
    //A[numCols][N_max] ==> A[numCols][N_xy][footprint_row * footprint_col]
    //indices[numCols][N_max][2]

    const uint64 ind_offset = uint64(iCol_thread) * uint64(N_max) + uint64(ivox*N_max/N_xy);

    if (fabs(sin_phi) > fabs(cos_phi))
    {
        // primary direction is y
        if (ivox >= N_f.y)
            return;
        const int iy = ivox;

        const float y = (float)iy * T_f.y + startVals_f.y;

        // u = cos_phi * y - sin_phi * x
        const float x_c = (cos_phi * y - u) / sin_phi;
        const int dix = max(1, int(ceil(0.5f * T_g.z / (T_f.x * fabs(sin_phi)))));
        const int ix_c = int(0.5f + (x_c - startVals_f.x) * T_x_inv);

        int count = 0;
        for (int ix = max(ix_c - dix, 0); ix <= min(ix_c + dix, N_f.x-1); ix++)
        {
            // calculate u index for x-0.5*T_f.x and x+0.5*T_f.x
            const float x = ix * T_f.x + startVals_f.x;

            const float x_A = (cos_phi * y - sin_phi * x - 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) * T_u_inv;
            const float x_B = (cos_phi * y - sin_phi * x + 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) * T_u_inv;
            const float uFootprint = max(0.0f, min(x_B, u_hi) - max(x_A, u_lo));

            if (uFootprint > 0.0f)
            {
                A[ind_offset + count] = l_phi * uFootprint;
                indices[2 * (ind_offset + count) + 0] = iy;
                indices[2 * (ind_offset + count) + 1] = ix;
                count += 1;
            }
        }
    }
    else
    {
        // primary direction is x
        if (ivox >= N_f.x)
            return;
        const int ix = ivox;

        const float x = (float)ix * T_f.x + startVals_f.x;

        // u = cos_phi * y - sin_phi * x
        const float y_c = (u + sin_phi * x) / cos_phi;
        const int diy = max(1, int(ceil(0.5f * T_g.z / (T_f.y * fabs(cos_phi)))));
        const int iy_c = int(0.5f + (y_c - startVals_f.y) * T_x_inv);

        int count = 0;
        for (int iy = max(iy_c - diy, 0); iy <= min(iy_c + diy, N_f.y-1); iy++)
        {
            // calculate u index for y-0.5*T_f.y and y+0.5*T_f.y
            const float y = iy * T_f.y + startVals_f.y;

            const float y_A = (cos_phi * y - sin_phi * x - 0.5f * T_f.y * fabs(cos_phi) - startVals_g.z) * T_u_inv;
            const float y_B = (cos_phi * y - sin_phi * x + 0.5f * T_f.y * fabs(cos_phi) - startVals_g.z) * T_u_inv;
            const float uFootprint = max(0.0f, min(y_B, u_hi) - max(y_A, u_lo));

            // system matrix value: l_phi * uFootprint
            if (uFootprint > 0.0f)
            {
                A[ind_offset + count] = l_phi * uFootprint;
                indices[2 * (ind_offset + count) + 0] = iy;
                indices[2 * (ind_offset + count) + 1] = ix;
                count += 1;
            }
        }
    }
}

bool systemMatrix(float*& A, short*& indices, int N_max, parameters* params, int iAngle, int iRow, int* iCols, int numCols, bool data_on_cpu)
{
    if (A == NULL || indices == NULL || params == NULL)
        return false;
    if (iRow < 0 || iRow >= params->numRows || iAngle < 0 || iAngle >= params->numAngles)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_A = 0;
    short* dev_ind = 0;
    int* dev_iCols = 0;

    uint64 N_A = params->numAngles * uint64(N_max);
    uint64 N_ind = N_A * uint64(3);

    // Allocate planogram data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    //float v = iRow * T_g.y + startVal_g.y;
    //float u = iCol * T_g.z + startVal_g.z;

    float rFOVsq = params->rFOV() * params->rFOV();

    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_A, N_A * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(system matrix) failed!\n");
            return false;
        }
        if ((cudaStatus = cudaMalloc((void**)&dev_ind, N_ind * sizeof(short))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(system matrix) failed!\n");
            if (dev_A != 0)
                cudaFree(dev_A);
            return false;
        }
        if ((cudaStatus = cudaMalloc((void**)&dev_iCols, numCols * sizeof(short))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(system matrix) failed!\n");
            if (dev_A != 0)
                cudaFree(dev_A);
            if (dev_ind != 0)
                cudaFree(dev_ind);
            return false;
        }
        if ((cudaStatus = cudaMemcpy(dev_iCols, iCols, numCols * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy(1D) failed!\n");
            printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
            if (dev_A != 0)
                cudaFree(dev_A);
            if (dev_ind != 0)
                cudaFree(dev_ind);
            cudaFree(dev_iCols);
            return false;
        }
    }
    else
    {
        dev_A = A;
        dev_ind = indices;
        dev_iCols = iCols;
    }

    float phi = params->phis[iAngle];
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    bool retVal = true;

    // Call Kernel
    int N_xy = max(params->numX, params->numY);
    //dim3 dimBlock = setBlockSize(N_g);
    //dim3 dimGrid = setGridSize(N_g, dimBlock);
    dim3 dimBlock(min(8, numCols), min(8, N_xy));
    dim3 dimGrid(int(ceil(double(numCols) / double(dimBlock.x))), int(ceil(double(N_xy) / double(dimBlock.y))));
    if (params->geometry == parameters::CONE)
    {
        LOG(logERROR, "system_matrix", "") << "Error: the function not yet implemented for cone-beam data" << std::endl;
        retVal = false;
    }
    else if (params->geometry == parameters::FAN)
    {
        LOG(logERROR, "system_matrix", "") << "Error: the function not yet implemented for fan-beam data" << std::endl;
        retVal = false;
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        systemMatrixKernel_parallel <<< dimGrid, dimBlock >>> (dev_A, dev_ind, dev_iCols, numCols, cos_phi, sin_phi, N_g, T_g, startVal_g, N_f, T_f, startVal_f, N_max, iAngle);
    }
    else if (params->geometry == parameters::CONE_PARALLEL)
    {
        LOG(logERROR, "system_matrix", "") << "Error: the function not yet implemented for cone-parallel data" << std::endl;
        retVal = false;
    }
    else if (params->geometry == parameters::MODULAR)
    {
        LOG(logERROR, "system_matrix", "") << "Error: the function not yet implemented for modular-beam data" << std::endl;
        retVal = false;
    }

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
        retVal = false;
    }

    if (data_on_cpu && retVal == true)
    {
        //pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
        if ((cudaStatus = cudaMemcpy(A, dev_A, N_A * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
        {
            fprintf(stderr, "error pulling result off GPU!\n");
            fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
            fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            retVal = false;
        }
        if ((cudaStatus = cudaMemcpy(indices, dev_ind, N_ind * sizeof(short), cudaMemcpyDeviceToHost)) != cudaSuccess)
        {
            fprintf(stderr, "error pulling result off GPU!\n");
            fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
            fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
            retVal = false;
        }
    }

    // Clean up
    //cudaFree(dev_phis);
    if (data_on_cpu)
    {
        if (dev_A != 0)
            cudaFree(dev_A);
        if (dev_ind != 0)
            cudaFree(dev_ind);
        if (dev_iCols != 0)
            cudaFree(dev_iCols);
    }

    return retVal;
}
