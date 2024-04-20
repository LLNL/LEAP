////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module scatter simulation and correction
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "scatter_models.cuh"

__global__ void firstOrderScatterModel(float* dev_proj, const int4 N_g, const float4 T_g, const float4 startVal_g, cudaTextureObject_t f, const int4 N_f, const float4 T_f, const float4 startVal_f, const float* Df, const float3 x_0, const float3 moduleCenter, const float3 rowVector, const float3 colVector)
{
    const int m = threadIdx.x + blockIdx.x * blockDim.x; // rows
    const int n = threadIdx.y + blockIdx.y * blockDim.y; // columns
    if (m >= N_g.y || n >= N_g.z)
        return;

    const float t = m * T_g.y + startVal_g.y;
    const float s = n * T_g.z + startVal_g.z;

    const float3 x_f = make_float3(moduleCenter.x + t * rowVector.x + s * colVector.x, moduleCenter.y + t * rowVector.y + s * colVector.y, moduleCenter.z + t * rowVector.z + s * colVector.z);

    for (int k = 0; k < N_f.z; k++)
    {
        const float x_3 = k * T_f.z + startVal_f.z;
        for (int j = 0; j < N_f.y; j++)
        {
            const float x_2 = j * T_f.y + startVal_f.y;
            for (int i = 0; i < N_f.x; i++)
            {
                const float3 x = make_float3(i * T_f.x + startVal_f.x, x_2, x_3);

                const float f = tex3D<float>(f, i, j, k);

                if (f > 0.0f)
                {
                    uint64 ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);
                    const float Df_firstLeg = Df[ind];

                    const float3 r_0 = make_float3(x.x - x_0.x, x.y - x_0.y, x.z - x_0.z);
                    const float mag_r_0_inv = rsqrtf(r_0.x * r_0.x + r_0.y * r_0.y + r_0.z * r_0.z);
                    const float3 r_f = make_float3(x_f.x - x.x, x_f.y - x.y, x_f.z - x.z);
                    const float mag_r_f_inv = rsqrtf(r_f.x * r_f.x + r_f.y * r_f.y + r_f.z * r_f.z);

                    const float cos_theta = (r_0.x * r_f.x + r_0.y * r_f.y + r_0.z * r_f.z) * mag_r_0_inv * mag_r_f_inv;
                    //const float theta = acos(cos_theta);
                    for (int igamma = 0; igamma < 10; igamma++)
                    {
                        const float gamma_0 = 60.0f; // FIXME
                        const float gamma_f = 510.975f * gamma_0 / (510.975f + (1.0f - cos_theta) * gamma_0);


                    }
                }
            }
        }
    }
}
__device__ float divergentBeamTransform(cudaTextureObject_t mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
        (p.z - startVal.z) / T.z);
    const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction

        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return 0.0f;
            int ix_max = min(N.x - 1, int(ceil((dst.x - startVal.x) / T.x)));

            val = tex3D<float>(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(ix_start) - 0.5f) - max(-0.5f, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;

            for (int ix = ix_start; ix <= ix_max; ix++)
                val += tex3D<float>(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix));
        }
        else
        {
            if (ip.x <= -0.5f) return 0.0f;
            int ix_min = max(0, int(floor((dst.x - startVal.x) / T.x)));

            val = tex3D<float>(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.x) - 0.5f), ip.x) - (float(ix_start) + 0.5f));

            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix >= ix_min; ix--)
                val += tex3D<float>(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix));
        }
        return val * sqrt(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return 0.0f;
            int iy_max = min(N.y - 1, int(ceil((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, ix_start + 0.5f, float(iy_start) + 0.5f, iz_start + 0.5f) *
                ((float(iy_start) - 0.5f) - max(-0.5f, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy <= iy_max; iy++)
                val += tex3D<float>(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
        }
        else
        {
            if (ip.y <= -0.5f) return 0.0f;
            int iy_min = max(0, int(floor((dst.y - startVal.y) / T.y)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.y) - 0.5f), ip.y) - (float(iy_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy >= iy_min; iy--)
                val += tex3D<float>(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy));
        }
        return val * sqrt(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
    }
    else
    {
        // integral in z direction
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return 0.0f;
            int iz_max = min(N.z - 1, int(ceil((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(iz_start) - 0.5f) - max(-0.5f, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz <= iz_max; iz++)
                val += tex3D<float>(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f);
        }
        else
        {
            if (ip.z <= -0.5f) return 0.0f;
            int iz_min = max(0, int(floor((dst.z - startVal.z) / T.z)));

            val = tex3D<float>(mu, ix_start + 0.5f, iy_start + 0.5f, float(iz_start) + 0.5f) *
                (min((float(N.z) - 0.5f), ip.z) - (float(iz_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz >= iz_min; iz--)
                val += tex3D<float>(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f);
        }
        return val * sqrt(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
    }
}

__global__ void lineIntegralSourceToVoxels(cudaTextureObject_t f, float* Df, const int4 N_f, const float4 T_f, const float4 startVal_f, const float3 sourcePosition)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    const float x = i * T_f.x + startVal_f.x;
    const float y = j * T_f.y + startVal_f.y;
    const float z = k * T_f.z + startVal_f.z;
    uint64 ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

    Df[ind] = divergentBeamTransform(f, N_f, T_f, startVal_f, make_float3(x, y, z), sourcePosition);

    /*
    uint64 ind;
    if (volumeDimensionOrder == 0)
        ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
    else
        ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);
    //*/
}

bool simulateScatter_firstOrder_singleMaterial(float* g, float* f, parameters* params, float* source, float* energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false || source == NULL || energies == NULL || detector == NULL || sigma == NULL || scatterDist == NULL)
        return false;
    if (params->geometry != parameters::MODULAR)
    {
        printf("Error: scatter estimation algorithm only implemented for modular-beam geometries. Please convert to modular-beam before running this algorithm.\n");
        return false;
    }
    
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);
    float* dev_g = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(projections) failed!\n");
    }

    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);
    float* dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = loadTexture(f_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));

    // Allocate data for attenuation in the first leg
    float* dev_Df;
    if ((cudaStatus = cudaMalloc((void**)&dev_Df, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
    }

    for (int i = 0; i < params->numAngles; i++)
    {
        float3 sourcePosition = make_float3(params->sourcePositions[3 * i + 0], params->sourcePositions[3 * i + 1], params->sourcePositions[3 * i + 2]);
        float3 moduleCenter = make_float3(params->moduleCenters[3 * i + 0], params->moduleCenters[3 * i + 1], params->moduleCenters[3 * i + 2]);
        float3 rowVector = make_float3(params->rowVectors[3 * i + 0], params->rowVectors[3 * i + 1], params->rowVectors[3 * i + 2]);
        float3 colVector = make_float3(params->colVectors[3 * i + 0], params->colVectors[3 * i + 1], params->colVectors[3 * i + 2]);

        // First calculate the line integrals from the source to each voxel
        // These get reused many times so it is good to treat this line a look up table
        firstLeg(f_data_txt, params, dev_Df, sourcePosition);

        float* dev_proj = &dev_g[uint64(i) * uint64(params->numRows * params->numCols)];

        // Now calculate the full scatter model for a fixed source and detector module
        dim3 dimBlock(8, 8);
        dim3 dimGrid(int(ceil(double(params->numRows) / double(dimBlock.x))), int(ceil(double(params->numCols) / double(dimBlock.y))));
        firstOrderScatterModel <<< dimGrid, dimBlock >>> (dev_proj, N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, dev_Df, sourcePosition, moduleCenter, rowVector, colVector);
    }

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }
    //if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    //else
    //    g = dev_g;

    // Clean up
    cudaFreeArray(f_data_array);
    cudaDestroyTextureObject(f_data_txt);
    cudaFree(dev_Df);
    cudaFree(dev_f);
    cudaFree(dev_g);

    return true;
}

bool firstLeg(cudaTextureObject_t f_data_txt, parameters* params, float* dev_Df, float3 sourcePosition)
{
    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    // Allocate projection data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);

    lineIntegralSourceToVoxels <<< dimGrid, dimBlock >>>  (f_data_txt, dev_Df, N_f, T_f, startVal_f, sourcePosition);

    return true;
}
