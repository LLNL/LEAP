////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// GPU-based geometric calibration routines
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "leap_defines.h"
#include "log.h"
#include "geometric_calibration.cuh"

#include <iostream>
#include <vector>

//d_data_txt, dev_cost, N, T, startVal, dev_phis, params->sod, params->sdd, params->tau, Delta_t, Delta_s, Delta_tilt
__global__ void consistencyCostKernel(cudaTextureObject_t g, float* cost, const int3 N, const float3 T, const float3 startVal, const float* phis, const float sod, const float sdd, const float tau, const float Delta_tilt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int iv = threadIdx.y + blockIdx.y * blockDim.y;
    //const int m = threadIdx.y + blockIdx.y * blockDim.y;
    //const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || iv >= N.y)
        return;

    uint64 ind = uint64(i) * uint64(N.y) + uint64(iv);

    const float cos_phi_i = cos(phis[i]);
    const float sin_phi_i = sin(phis[i]);
    const float2 s_i = make_float2(sod * cos_phi_i + tau * sin_phi_i, sod * sin_phi_i - tau * cos_phi_i);

    const float T_u_inv = sdd / T.z;
    const float T_v_inv = sdd / T.y;
    const float u_0 = startVal.z / sdd;
    const float u_end = T.z / sdd * (N.y-1) + u_0;
    const float v_0 = startVal.y / sdd;

    const float2 r_left_i = make_float2(-cos_phi_i - sin_phi_i * u_0, -sin_phi_i + cos_phi_i * u_0);
    const float2 r_right_i = make_float2(-cos_phi_i - sin_phi_i * u_end, -sin_phi_i + cos_phi_i * u_end);

    float cost_i = 0.0;
    const float maxAngleDiff = 150.0 * PI / 180.0;
    for (int j = 0; j < N.x; j++)
    {
        if (j == i || fabs(phis[i] - phis[j]) > maxAngleDiff)
            continue;

        bool doPrint = false;
        if (i == 0 && j == 1 && iv == N.y / 2)
            doPrint = true;

        const float cos_phi_j = cos(phis[j]);
        const float sin_phi_j = sin(phis[j]);

        const float2 s_j = make_float2(sod * cos_phi_j + tau * sin_phi_j, sod * sin_phi_j - tau * cos_phi_j);
        const float dist_inv = rsqrtf((s_j.x - s_i.x) * (s_j.x - s_i.x) + (s_j.y - s_i.y) * (s_j.y - s_i.y));

        const float2 r_left_j = make_float2(-cos_phi_j - sin_phi_j * u_0, -sin_phi_j + cos_phi_j * u_0);
        const float2 r_right_j = make_float2(-cos_phi_j - sin_phi_j * u_end, -sin_phi_j + cos_phi_j * u_end);

        const float2 virtual_colVec = make_float2((s_j.x - s_i.x) * dist_inv, (s_j.y - s_i.y) * dist_inv);
        const float2 virtual_normal = make_float2(virtual_colVec.y, -virtual_colVec.x);

        const float s_i_dot_n = s_i.x * virtual_normal.x + s_i.y * virtual_normal.y;
        const float s_j_dot_n = s_j.x * virtual_normal.x + s_j.y * virtual_normal.y;

        
        // calculate D_virt_i, D_virt_j
        const float D_virt_i = fabs(s_i_dot_n);
        const float D_virt_j = fabs(s_j_dot_n);

        const float T_u_virt = T.z * D_virt_i / sdd;
        const float T_u_virt_inv = 1.0f / T_u_virt;
        const float T_v_virt = T.y * D_virt_i / sdd;
        const float u_0_virt = T_u_virt / T.z * startVal.z;
        const float v_0_virt = T_v_virt / T.y * startVal.y;

        const float v_virt = iv * T_v_virt + v_0_virt;

        float t, u_arg;
        // Index range for view i
        t = -s_i_dot_n / (r_left_i.x * virtual_normal.x + r_left_i.y * virtual_normal.y);
        u_arg = (s_i.x + t * r_left_i.x) * virtual_colVec.x + (s_i.y + t * r_left_i.y) * virtual_colVec.y;
        const int iu_lo_i = int((u_arg - u_0_virt) * T_u_virt_inv) - 1;

        t = -s_i_dot_n / (r_right_i.x * virtual_normal.x + r_right_i.y * virtual_normal.y);
        u_arg = (s_i.x + t * r_right_i.x) * virtual_colVec.x + (s_i.y + t * r_right_i.y) * virtual_colVec.y;
        const int iu_hi_i = int(ceil((u_arg - u_0_virt) * T_u_virt_inv)) + 1;

        // Index range for view j
        t = -s_j_dot_n / (r_left_j.x * virtual_normal.x + r_left_j.y * virtual_normal.y);
        u_arg = (s_j.x + t * r_left_j.x) * virtual_colVec.x + (s_j.y + t * r_left_j.y) * virtual_colVec.y;
        const int iu_lo_j = int((u_arg - u_0_virt) * T_u_virt_inv) - 1;

        t = -s_j_dot_n / (r_right_j.x * virtual_normal.x + r_right_j.y * virtual_normal.y);
        u_arg = (s_j.x + t * r_right_j.x) * virtual_colVec.x + (s_j.y + t * r_right_j.y) * virtual_colVec.y;
        const int iu_hi_j = int(ceil((u_arg - u_0_virt) * T_u_virt_inv)) + 1;

        //const float coneWeight_i = rsqrtf(D_virt_i * D_virt_i + v * v);
        //const float coneWeight_j = rsqrtf(D_virt_j * D_virt_j + v * v);

        /*
        if (doPrint)
        {
            printf("virtual_colVec = %f, %f\n", virtual_colVec.x, virtual_colVec.y);
            printf("virtual_normal = %f, %f\n", virtual_normal.x, virtual_normal.y);
            printf("D_virt = %f, %f\n", D_virt_i, D_virt_j);
            printf("T_u_virt = %f\n", T_u_virt);
            printf("T_v_virt = %f\n", T_v_virt);
            printf("u_0_virt = %f\n", u_0_virt);
            printf("v_0_virt = %f\n", v_0_virt);
        }
        //*/

        float accum_i = 0.0f;
        for (int iu = iu_lo_i; iu <= iu_hi_i; iu++)
        {
            const float u_virt = iu * T_u_virt + u_0_virt;
            const float3 vox = make_float3(virtual_colVec.x * u_virt, virtual_colVec.y * u_virt, v_virt); // backproject to this point

            const float integrandWeight_i = rsqrtf(D_virt_i * D_virt_i + v_virt * v_virt + u_virt * u_virt);

            const float v_denom_inv_i = 1.0f / (sod - cos_phi_i * vox.x - sin_phi_i * vox.y);
            const float u_arg_i = (-sin_phi_i * vox.x + cos_phi_i * vox.y + tau) * v_denom_inv_i;
            const float v_arg_i = vox.z * v_denom_inv_i;

            accum_i += integrandWeight_i * tex3D<float>(g, (u_arg_i - u_0) * T_u_inv + 0.5f, (v_arg_i - v_0) * T_v_inv + 0.5f, i + 0.5f);
        }

        float accum_j = 0.0f;
        for (int iu = iu_lo_j; iu <= iu_hi_j; iu++)
        {
            const float u_virt = iu * T_u_virt + u_0_virt;
            const float3 vox = make_float3(virtual_colVec.x * u_virt, virtual_colVec.y * u_virt, v_virt); // backproject to this point

            const float integrandWeight_j = rsqrtf(D_virt_j * D_virt_j + v_virt * v_virt + u_virt * u_virt);

            const float v_denom_inv_j = 1.0f / (sod - cos_phi_j * vox.x - sin_phi_j * vox.y);
            const float u_arg_j = (-sin_phi_j * vox.x + cos_phi_j * vox.y + tau) * v_denom_inv_j;
            const float v_arg_j = vox.z * v_denom_inv_j;

            accum_j += integrandWeight_j * tex3D<float>(g, (u_arg_j - u_0) * T_u_inv + 0.5f, (v_arg_j - v_0) * T_v_inv + 0.5f, j + 0.5f);
        }
        /*
        for (int iu = -1000; iu < N.z+1000; iu++)
        {
            const float u_virt = iu * T_u_virt + u_0_virt;
            const float3 vox = make_float3(virtual_colVec.x * u_virt, virtual_colVec.y * u_virt, v_virt); // backproject to this point

            const float integrandWeight_i = rsqrtf(D_virt_i * D_virt_i + v_virt * v_virt + u_virt * u_virt);
            const float integrandWeight_j = rsqrtf(D_virt_j * D_virt_j + v_virt * v_virt + u_virt * u_virt);

            const float v_denom_inv_i = 1.0f / (sod - cos_phi_i * vox.x - sin_phi_i * vox.y);
            const float v_denom_inv_j = 1.0f / (sod - cos_phi_j * vox.x - sin_phi_j * vox.y);
            const float u_arg_i = (-sin_phi_i * vox.x + cos_phi_i * vox.y + tau) * v_denom_inv_i;
            const float u_arg_j = (-sin_phi_j * vox.x + cos_phi_j * vox.y + tau) * v_denom_inv_j;
            const float v_arg_i = vox.z * v_denom_inv_i;
            const float v_arg_j = vox.z * v_denom_inv_j;

            accum_i += integrandWeight_i * tex3D<float>(g, (u_arg_i - u_0) * T_u_inv + 0.5f, (v_arg_i - v_0) * T_v_inv + 0.5f, i + 0.5f);
            accum_j += integrandWeight_j * tex3D<float>(g, (u_arg_j - u_0) * T_u_inv + 0.5f, (v_arg_j - v_0) * T_v_inv + 0.5f, j + 0.5f);
        }
        //*/
        cost_i += (accum_i - accum_j) * (accum_i - accum_j);
    }

    cost[ind] = cost_i;
}

float consistencyCost(float* g, parameters* params, bool data_on_cpu, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt)
{
    if (g == NULL || params == NULL)
        return -1.0;
    if (data_on_cpu == false) // FIXME
    {
        LOG(logERROR, "geometric_calibration", "consistencyCost") << "Currently only implemented for data on the CPU!" << std::endl;
        return -1.0;
    }
    if (params->geometry != parameters::CONE || params->detectorType != parameters::FLAT || params->helicalPitch != 0.0)
    {
        LOG(logERROR, "geometric_calibration", "consistencyCost") << "Consistency metric only works for axial flat panel cone-beam geometries!" << std::endl;
        return -1.0;
    }

    // find projections spaced by 40 degrees
    float angularSeparation = 40.0 * PI / 180.0;
    std::vector<int> proj_inds;
    int ind = 0;
    float phi_cur = params->phis[ind];
    proj_inds.push_back(ind);
    for (int i = 1; i < params->numAngles; i++)
    {
        if (fabs(phi_cur - params->phis[i]) > angularSeparation)
        {
            proj_inds.push_back(i);
            phi_cur = params->phis[i];
        }
    }
    int numAngles_subset = int(proj_inds.size());
    if (numAngles_subset <= 1)
    {
        LOG(logERROR, "geometric_calibration", "consistencyCost") << "Insufficient angular coverage!" << std::endl;
        return -1.0;
    }

    cudaError_t cudaStatus;
    cudaSetDevice(params->whichGPU);

    uint64 projectionSize = uint64(params->numRows) * uint64(params->numCols);

    float* dev_g_subset = 0;
    if (cudaSuccess != (cudaStatus = cudaMalloc((void**)&dev_g_subset, uint64(numAngles_subset) * projectionSize * sizeof(float))))
    {
        fprintf(stderr, "cudaMalloc failed!\n");
        printf("cudaMalloc Error: %s\n", cudaGetErrorString(cudaStatus));
        return -1.0;
    }

    float* phis = new float[numAngles_subset];
    for (int ind = 0; ind < numAngles_subset; ind++)
    {
        float* dev_g_subset_ind = &dev_g_subset[uint64(ind)* projectionSize];
        float* g_ind = &g[uint64(proj_inds[ind]) * projectionSize];
        if ((cudaStatus = cudaMemcpy(dev_g_subset_ind, g_ind, projectionSize * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy(projection) failed!\n");
            printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
            cudaFree(dev_g_subset);
            delete[] phis;
            return -1.0;
        }
        phis[ind] = params->phis[proj_inds[ind]];
    }
    float* dev_phis = copy1DdataToGPU(phis, numAngles_subset, params->whichGPU);
    delete[] phis;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = false;
    int3 N = make_int3(numAngles_subset, params->numRows, params->numCols);
    float3 T = make_float3(params->T_phi(), params->pixelHeight, params->pixelWidth);
    float3 startVal = make_float3(params->phis[0], params->v(0) + Delta_centerRow * params->pixelHeight, params->u(0) + Delta_centerCol * params->pixelWidth);
    float tau = params->tau + Delta_tau;
    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

    // Copy to texture
    // FIXME: should copy directly from CPU to 3D array
    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g_subset, N, false, true);
    
    // Reuse dev_g_subset for cost values
    float* dev_cost = dev_g_subset;
    setToConstant(dev_g_subset, 0.0, N, params->whichGPU);
    
    dim3 dimBlock(min(8, N.x), min(8, N.y));
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))));
    //dim3 dimBlock = setBlockSize(N);
    //dim3 dimGrid = setGridSize(N, dimBlock);

    consistencyCostKernel <<< dimGrid, dimBlock >>> (d_data_txt, dev_cost, N, T, startVal, dev_phis, params->sod, params->sdd, tau, Delta_tilt);
    float retVal = sum(dev_g_subset, N, params->whichGPU);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_g_subset);
    cudaFree(dev_phis);

    return retVal;
}
