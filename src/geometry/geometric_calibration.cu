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
#include "file_io.h"

#include <iostream>
#include <vector>


__global__ void innerProduct_stack(float* f, const int4 N_f, float* shift_values, const int numShifts)
{
    const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= numShifts)
        return;

    float val = 0.0f;
    for (int i = 0; i < N_f.x; i++)
    {
        for (int j = 0; j < N_f.y; j++)
        {
            uint64 ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);
            val += f[ind]*f[ind];
        }
    }
    shift_values[k] = val / float(N_f.x*N_f.y);
}

__global__ void coneBeamBackprojectorKernel_sweep(TEX_DATA g, const int4 N_g, const float4 T_g, const float4 startVals_g, float* f, const int4 N_f, const float4 T_f, const float4 startVals_f, const float R, const float D, const float tau, const float tiltAngle, const float pitchAngle, const float rFOVsq, const float* phis, const float* shift_values, const int numShifts, const float* tilt_values, const int numTilts, const int which_param, float* metrics)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    const int shift_ind = k % numShifts;
    const int tilt_ind = (k-shift_ind) / numShifts;
    if (i >= N_f.x || j >= N_f.y || shift_ind >= numShifts || tilt_ind >= numTilts)
        return;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z_val = startVals_f.z;

    //uint64 ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);
    if (x * x + y * y > rFOVsq)
    {
        //f[ind] = 0.0f;
        return;
    }

    float sin_tilt, cos_tilt;
    __sincosf(tiltAngle + tilt_values[tilt_ind]*PI/180.0, &sin_tilt, &cos_tilt);

    const float Tv_inv = 1.0f / T_g.y;
    const float Tu_inv = 1.0f / T_g.z;

    const float v_min_ind = -10.0f;
    const float v_max_ind = N_g.y-1+10.0f;

    float tau_shift_temp = 0.0f;
    float u_shift_temp = startVals_g.z;
    if (which_param == 0)
    {
        u_shift_temp = startVals_g.z - shift_values[shift_ind]*T_g.z;
        tau_shift_temp = 0.0;
    }
    else
    {
        u_shift_temp = startVals_g.z;
        tau_shift_temp = shift_values[shift_ind];
    }
    const float u_shift = u_shift_temp;
    const float tau_shift = tau_shift_temp;

    float val = 0.0f;

    float sin_pitch, cos_pitch;
    __sincosf(pitchAngle, &sin_pitch, &cos_pitch);

    //const float3 n_vec_0 = make_float3(cos_pitch, 0.0f, -sin_pitch);
    const float3 u_vec_0 = make_float3(sin_pitch*sin_tilt, cos_tilt, cos_pitch*sin_tilt);
    const float3 v_vec_0 = make_float3(sin_pitch*cos_tilt, -sin_tilt, cos_pitch*cos_tilt);

    for (int l = 0; l < N_g.x; l++)
    {
        const float L = (float)l + 0.5f;

        float cos_phi, sin_phi;
        const float phi = phis[l];
        __sincosf(phi, &sin_phi, &cos_phi);

        const float u_0 = u_shift;
        const float v_0 = startVals_g.y;

        const float v_min = v_min_ind*T_g.y + v_0;
        const float v_max = v_max_ind*T_g.y + v_0;

        const float dist_from_source_0 = (R - x * cos_phi - y * sin_phi)*cos_pitch;

        //const float u_vec_dot_x = x * (cos_phi*u_vec_0.x - sin_phi*u_vec_0.y) + y * (sin_phi*u_vec_0.x + cos_phi*u_vec_0.y);
        const float v_vec_dot_x = x * (cos_phi*v_vec_0.x - sin_phi*v_vec_0.y) + y * (sin_phi*v_vec_0.x + cos_phi*v_vec_0.y);

        //const float u_val_num_shift = R*u_vec_0.x - tau*u_vec_0.y;
        const float v_val_num_shift = R*v_vec_0.x - (tau + tau_shift)*v_vec_0.y;

        const float dist_from_source_inv = 1.0f / (dist_from_source_0 + sin_pitch*z_val);

        const float v_val = (v_vec_dot_x + z_val * v_vec_0.z - v_val_num_shift) * dist_from_source_inv;

        if (v_min <= v_val && v_val <= v_max)
        {
            const float u_vec_dot_x = x * (cos_phi*u_vec_0.x - sin_phi*u_vec_0.y) + y * (sin_phi*u_vec_0.x + cos_phi*u_vec_0.y);
            const float u_val_num_shift = R*u_vec_0.x - (tau + tau_shift)*u_vec_0.y;
            const float u_val = (u_vec_dot_x + z_val * u_vec_0.z - u_val_num_shift) * dist_from_source_inv;
            val += TEX3D(g, (u_val - u_0) * Tu_inv + 0.5f, (v_val - v_0) * Tv_inv + 0.5f, L) * dist_from_source_inv * dist_from_source_inv * sqrtf(1.0f + u_val * u_val + v_val * v_val);
        }
    }
    
    //const float scalar = T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    //f[ind] = val * scalar;
    val *= T_f.x * T_f.y * T_f.z / (T_g.y * T_g.z);
    val *= val;
    val *= 1.0f / float(N_f.x * N_f.y);
    atomicAdd(&metrics[k], val);
}

//d_data_txt, dev_cost, N, T, startVal, dev_phis, params->sod, params->sdd, params->tau, Delta_t, Delta_s, Delta_tilt
__global__ void consistencyCostKernel(TEX_DATA g, float* cost, const int3 N, const float3 T, const float3 startVal, const float* phis, const float sod, const float sdd, const float tau, const float Delta_tilt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int iv = threadIdx.y + blockIdx.y * blockDim.y;
    //const int m = threadIdx.y + blockIdx.y * blockDim.y;
    //const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N.x || iv >= N.y)
        return;

    const float cos_psi = cos(Delta_tilt);
    const float sin_psi = sin(Delta_tilt);
    const float3 Rpsi_e3 = make_float3(0.0f, -sin_psi, cos_psi);

    const int padding = 1000*0;

    uint64 ind = uint64(i) * uint64(N.y) + uint64(iv);

    const float cos_phi_i = cos(phis[i]);
    const float sin_phi_i = sin(phis[i]);
    const float2 s_i = make_float2(sod * cos_phi_i + tau * sin_phi_i, sod * sin_phi_i - tau * cos_phi_i);

    // if psi == 0, (-sin_phi_i, cos_phi_i, 0)
    const float3 Rphi_Rpsi_e2_i = make_float3(-sin_phi_i * cos_psi, cos_phi_i * cos_psi, sin_psi);

    const float T_u_inv = sdd / T.z;
    const float T_v_inv = sdd / T.y;
    const float u_0 = startVal.z / sdd;
    const float u_end = T.z / sdd * (N.y-1) + u_0;
    const float u_0_edge = u_0 - 0.5f * T.z/sdd;
    const float u_end_edge = u_end + 0.5f * T.z/sdd;
    const float v_0 = startVal.y / sdd;

    const float2 r_left_i = make_float2(-cos_phi_i - sin_phi_i * u_0_edge, -sin_phi_i + cos_phi_i * u_0_edge);
    const float2 r_right_i = make_float2(-cos_phi_i - sin_phi_i * u_end_edge, -sin_phi_i + cos_phi_i * u_end_edge);
    //const float2 r_left_i = make_float2(-cos_phi_i + Rphi_Rpsi_e2_i.x * u_0_edge, -sin_phi_i + Rphi_Rpsi_e2_i.y * u_0_edge);
    //const float2 r_right_i = make_float2(-cos_phi_i + Rphi_Rpsi_e2_i.x * u_end_edge, -sin_phi_i + Rphi_Rpsi_e2_i.y * u_end_edge);

    //u_arg_i = (-sin_phi_i * vox.x + cos_phi_i * vox.y + tau) * v_denom_inv_i;
    //u_arg_i = ((vox.x - tau * sin_phi_i) * Rphi_Rpsi_e2_i.x + (vox.y + tau * cos_phi_i) * Rphi_Rpsi_e2_i.y) * v_denom_inv_i;

    float cost_i = 0.0;
    const float maxAngleDiff = 150.0f * PI / 180.0f;
    const float minAngleDiff = 35.0f * PI / 180.0f;
    for (int j = 0; j < N.x; j++)
    {
        float phi_diff = fabs(phis[i] - phis[j]);
        phi_diff -= floor(phi_diff / (2.0 * PI)) * phi_diff;
        if (j == i || phi_diff > maxAngleDiff || phi_diff < minAngleDiff)
            continue;

        const float cos_phi_j = cos(phis[j]);
        const float sin_phi_j = sin(phis[j]);

        const float3 Rphi_Rpsi_e2_j = make_float3(-sin_phi_j * cos_psi, cos_phi_j * cos_psi, sin_psi);

        const float2 s_j = make_float2(sod * cos_phi_j + tau * sin_phi_j, sod * sin_phi_j - tau * cos_phi_j);
        const float dist_inv = rsqrtf((s_j.x - s_i.x) * (s_j.x - s_i.x) + (s_j.y - s_i.y) * (s_j.y - s_i.y));

        const float2 r_left_j = make_float2(-cos_phi_j - sin_phi_j * u_0_edge, -sin_phi_j + cos_phi_j * u_0_edge);
        const float2 r_right_j = make_float2(-cos_phi_j - sin_phi_j * u_end_edge, -sin_phi_j + cos_phi_j * u_end_edge);
        //const float2 r_left_j = make_float2(-cos_phi_j + Rphi_Rpsi_e2_j.x * u_0_edge, -sin_phi_j + Rphi_Rpsi_e2_j.y * u_0_edge);
        //const float2 r_right_j = make_float2(-cos_phi_j + Rphi_Rpsi_e2_j.x * u_end_edge, -sin_phi_j + Rphi_Rpsi_e2_j.y * u_end_edge);

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

        float t, u_arg_A, u_arg_B;
        // Index range for view i
        t = -s_i_dot_n / (r_left_i.x * virtual_normal.x + r_left_i.y * virtual_normal.y);
        u_arg_A = ((s_i.x + t * r_left_i.x) * virtual_colVec.x + (s_i.y + t * r_left_i.y) * virtual_colVec.y - u_0_virt) * T_u_virt_inv;

        t = -s_i_dot_n / (r_right_i.x * virtual_normal.x + r_right_i.y * virtual_normal.y);
        u_arg_B = ((s_i.x + t * r_right_i.x) * virtual_colVec.x + (s_i.y + t * r_right_i.y) * virtual_colVec.y - u_0_virt) * T_u_virt_inv;
        const int iu_lo_i = int(floor(min(u_arg_A, u_arg_B))) - 2;
        const int iu_hi_i = int(ceil(max(u_arg_A, u_arg_B))) + 2;

        // Index range for view j
        t = -s_j_dot_n / (r_left_j.x * virtual_normal.x + r_left_j.y * virtual_normal.y);
        u_arg_A = ((s_j.x + t * r_left_j.x) * virtual_colVec.x + (s_j.y + t * r_left_j.y) * virtual_colVec.y - u_0_virt) * T_u_virt_inv;
        
        t = -s_j_dot_n / (r_right_j.x * virtual_normal.x + r_right_j.y * virtual_normal.y);
        u_arg_B = ((s_j.x + t * r_right_j.x) * virtual_colVec.x + (s_j.y + t * r_right_j.y) * virtual_colVec.y - u_0_virt) * T_u_virt_inv;
        const int iu_lo_j = int(floor(min(u_arg_A, u_arg_B))) - 2;
        const int iu_hi_j = int(ceil(max(u_arg_A, u_arg_B))) + 2;

        float accum_i = 0.0f;
        for (int iu = iu_lo_i-padding; iu <= iu_hi_i+padding; iu++)
        {
            const float u_virt = iu * T_u_virt + u_0_virt;
            const float3 vox = make_float3(virtual_colVec.x * u_virt, virtual_colVec.y * u_virt, v_virt); // backproject to this point

            const float integrandWeight_i = rsqrtf(D_virt_i * D_virt_i + v_virt * v_virt + u_virt * u_virt);

            const float v_denom_inv_i = 1.0f / (sod - cos_phi_i * vox.x - sin_phi_i * vox.y);
            float u_arg_i, v_arg_i;
            if (Delta_tilt == 0.0f)
            {
                u_arg_i = (-sin_phi_i * vox.x + cos_phi_i * vox.y + tau) * v_denom_inv_i;
                v_arg_i = vox.z * v_denom_inv_i;
            }
            else
            {
                u_arg_i = ((vox.x - tau * sin_phi_i) * Rphi_Rpsi_e2_i.x + (vox.y + tau * cos_phi_i) * Rphi_Rpsi_e2_i.y) * v_denom_inv_i;
                v_arg_i = ((vox.x - tau * sin_phi_i) * Rpsi_e3.x + (vox.y + tau * cos_phi_i) * Rpsi_e3.y) * v_denom_inv_i;
            }

            accum_i += integrandWeight_i * TEX3D(g, (u_arg_i - u_0) * T_u_inv + 0.5f, (v_arg_i - v_0) * T_v_inv + 0.5f, i + 0.5f);
        }

        float accum_j = 0.0f;
        for (int iu = iu_lo_j-padding; iu <= iu_hi_j+padding; iu++)
        {
            const float u_virt = iu * T_u_virt + u_0_virt;
            const float3 vox = make_float3(virtual_colVec.x * u_virt, virtual_colVec.y * u_virt, v_virt); // backproject to this point

            const float integrandWeight_j = rsqrtf(D_virt_j * D_virt_j + v_virt * v_virt + u_virt * u_virt);

            const float v_denom_inv_j = 1.0f / (sod - cos_phi_j * vox.x - sin_phi_j * vox.y);
            //const float u_arg_j = (-sin_phi_j * vox.x + cos_phi_j * vox.y + tau) * v_denom_inv_j;
            //const float v_arg_j = vox.z * v_denom_inv_j;
            float u_arg_j, v_arg_j;
            if (Delta_tilt == 0.0f)
            {
                u_arg_j = (-sin_phi_j * vox.x + cos_phi_j * vox.y + tau) * v_denom_inv_j;
                v_arg_j = vox.z * v_denom_inv_j;
            }
            else
            {
                u_arg_j = ((vox.x - tau * sin_phi_j) * Rphi_Rpsi_e2_j.x + (vox.y + tau * cos_phi_j) * Rphi_Rpsi_e2_j.y) * v_denom_inv_j;
                v_arg_j = ((vox.x - tau * sin_phi_j) * Rpsi_e3.x + (vox.y + tau * cos_phi_j) * Rpsi_e3.y) * v_denom_inv_j;
            }

            accum_j += integrandWeight_j * TEX3D(g, (u_arg_j - u_0) * T_u_inv + 0.5f, (v_arg_j - v_0) * T_v_inv + 0.5f, j + 0.5f);
        }
        cost_i += (accum_i - accum_j) * (accum_i - accum_j);
    }

    cost[ind] = cost_i;
}

float consistencyCost(float* g, parameters* params, bool data_on_cpu, float Delta_centerRow, float Delta_centerCol, float Delta_tau, float Delta_tilt)
{
    if (g == NULL || params == NULL)
        return -1.0;
    /*
    if (data_on_cpu == false) // FIXME
    {
        LOG(logERROR, "geometric_calibration", "consistencyCost") << "Currently only implemented for data on the CPU!" << std::endl;
        return -1.0;
    }
    //*/
    if (params->geometry != parameters::CONE || params->detectorType != parameters::FLAT || params->helicalPitch != 0.0)
    {
        LOG(logERROR, "geometric_calibration", "consistencyCost") << "Consistency metric only works for axial flat panel cone-beam geometries!" << std::endl;
        return -1.0;
    }

    // find projections spaced by 40 degrees
    float angularSeparation = 40.0 * PI / 180.0;
    //float angularSeparation = 20.0 * PI / 180.0;
    std::vector<int> proj_inds;
    int ind = 0;
    float phi_cur = params->phis[ind];
    proj_inds.push_back(ind);
    for (int i = 1; i < params->numAngles; i++)
    {
        float phi_diff = fabs(phi_cur - params->phis[i]);
        phi_diff -= floor(phi_diff / (2.0 * PI)) * phi_diff;
        if (phi_diff >= angularSeparation)
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
        if (data_on_cpu)
        {
            if ((cudaStatus = cudaMemcpy(dev_g_subset_ind, g_ind, projectionSize * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy(projection) failed!\n");
                printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(dev_g_subset);
                delete[] phis;
                return -1.0;
            }
        }
        else
        {
            if ((cudaStatus = cudaMemcpy(dev_g_subset_ind, g_ind, projectionSize * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy(projection) failed!\n");
                printf("cudaMemcpy Error: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(dev_g_subset);
                delete[] phis;
                return -1.0;
            }
        }
        phis[ind] = params->phis[proj_inds[ind]];
    }
    float* dev_phis = copy1DdataToGPU(phis, numAngles_subset, params->whichGPU);
    delete[] phis;

    bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
    params->normalizeConeAndFanCoordinateFunctions = false;
    int3 N = make_int3(numAngles_subset, params->numRows, params->numCols);
    float3 T = make_float3(params->T_phi(), params->pixelHeight, params->pixelWidth);
    float3 startVal = make_float3(params->phis[0], params->v(0) - Delta_centerRow * params->pixelHeight, params->u(0) - Delta_centerCol * params->pixelWidth);
    float tau = params->tau + Delta_tau;
    params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;

    // Copy to texture
    // FIXME: should copy directly from CPU to 3D array
    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = loadTexture(d_data_txt, dev_g_subset, N, false, true);
    
    // Reuse dev_g_subset for cost values
    float* dev_cost = dev_g_subset;
    setToConstant(dev_g_subset, 0.0, N, params->whichGPU);
    
    dim3 dimBlock(min(8, N.x), min(8, N.y));
    dim3 dimGrid(int(ceil(double(N.x) / double(dimBlock.x))), int(ceil(double(N.y) / double(dimBlock.y))));
    //dim3 dimBlock = setBlockSize(N);
    //dim3 dimGrid = setGridSize(N, dimBlock);

    consistencyCostKernel <<< dimGrid, dimBlock >>> (d_data_txt, dev_cost, N, T, startVal, dev_phis, params->sod, params->sdd, tau, Delta_tilt*PI/180.0);
    //float retVal = sum(dev_g_subset, N, params->whichGPU);
    float retVal = sum(dev_g_subset, make_int3(N.x, N.y, 1), params->whichGPU);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    freeTexture(d_data_array, d_data_txt);
    cudaFree(dev_g_subset);
    cudaFree(dev_phis);

    return retVal;
}

bool backprojection_sweep(float* g, parameters* params, float* shifts, int numShifts, float* tilts, int numTilts, int which_param, float* costValues, bool data_on_cpu)
{
    if (g == NULL || params == NULL || shifts == NULL || tilts == NULL || numShifts <= 0 || numTilts <= 0 || costValues == NULL || params->geometry != parameters::CONE || params->detectorType != parameters::FLAT)
        return false;
    //if (params->numZ != 1)
    //    return false;
    if (params->helicalPitch != 0.0)
        return false;
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    int numVol = 0;
    float memAvailable = getAvailableGPUmemory(params->whichGPU);
    if (params->volumeDataSize()*numShifts*numTilts*numVol + params->projectionDataSize() >= memAvailable)
    {
        printf("Not enough GPU memory for this operation!\n");
        return false;
    }

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    float* dev_f = 0;
    /*
    if ((cudaStatus = cudaMalloc((void**)&dev_f, uint64(numShifts) * uint64(params->numY) * uint64(params->numX) * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
    }
    //*/

    float* dev_metrics = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_metrics, uint64(numShifts*numTilts) * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(metrics) failed!\n");
    }
    cudaMemset(dev_metrics, 0, numShifts*numTilts*sizeof(float));

    int4 N_f_stack = make_int4(N_f.x, N_f.y, numShifts*numTilts, N_f.w);
    dim3 dimBlock = setBlockSize(N_f_stack);
    dim3 dimGrid = setGridSize(N_f_stack, dimBlock);

    // may have to split up g
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);


    TEX_DATA d_data_txt = {};
    TEX_ARRAY d_data_array = NULL;
    if (data_on_cpu)
        d_data_array = loadTexture_from_cpu(d_data_txt, g, N_g, params->doExtrapolation, true);
    else
        d_data_array = loadTexture(d_data_txt, g, N_g, params->doExtrapolation, true);

    float* dev_shifts = copy1DdataToGPU(shifts, numShifts, params->whichGPU);
    float* dev_tilts = copy1DdataToGPU(tilts, numTilts, params->whichGPU);

    float rFOVsq = params->rFOV()*params->rFOV();
    float* dev_phis = copyAngleArrayToGPU(params);

    for (int islice = 0; islice < params->numZ; islice++)
    {
        startVal_f.z = params->z_samples(islice);
        coneBeamBackprojectorKernel_sweep <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, params->tiltAngle*PI/180.0, params->pitchAngle*PI/180.0, rFOVsq, dev_phis, dev_shifts, numShifts, dev_tilts, numTilts, which_param, dev_metrics);
    }

    //dim3 dimBlock_2D(8, 8);
    //dim3 dimGrid_2D(int(ceil(double(params->numY) / double(dimBlock_2D.x))), int(ceil(double(params->numX) / double(dimBlock_2D.y))));
    if (dev_f != 0)
    {
        // tilts not yet handled here
        innerProduct_stack <<< int(ceil(double(numShifts) / 8.0)), 8 >>> (dev_f, N_f, dev_shifts, numShifts);
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "backprojection_sweep: kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    /*
    float* f = new float[params->numY*params->numX];
    pullVolumeDataFromGPU(f, params, &dev_f[5*params->numY*params->numX], params->whichGPU);
    char fileName[512];
    sprintf(fileName, "/home/kyle/Documents/data/test.tif");
    write_tif(fileName, f, params->numY, params->numX);
    delete [] f;
    //*/

    pull1DdataFromGPU(costValues, numShifts*numTilts, dev_metrics, params->whichGPU);

    cudaFree(dev_shifts);
    cudaFree(dev_tilts);
    if (dev_f != 0)
        cudaFree(dev_f);
    if (dev_metrics != 0)
        cudaFree(dev_metrics);
    freeTexture(d_data_array, d_data_txt);
    cudaFree(dev_phis);

    return true;
}
