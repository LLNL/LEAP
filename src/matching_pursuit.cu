////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for matching pursuit algorithms which are a part of
// dictionary denoising.  This file is an adaptation of code written by myself
// (Kyle) several years ago in a package called "3Ddensoing"
////////////////////////////////////////////////////////////////////////////////
#include "matching_pursuit.cuh"
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cuda_utils.h"

#include <iostream>
#include <vector>

__device__ float innerProduct(cudaTextureObject_t f, const float* patches, const int3 voxelIndices,
                              const int whichPatch, const int3 patchSize)
{
    float retVal = 0.0f;
    const int ind_shift = whichPatch * patchSize.x * patchSize.y * patchSize.z;
    for (int di = 0; di < patchSize.x; di++)
    {
        for (int dj = 0; dj < patchSize.y; dj++)
        {
            for (int dk = 0; dk < patchSize.z; dk++)
            {
                const int ind = di * patchSize.y * patchSize.z + dj * patchSize.z + dk + ind_shift;
                retVal += patches[ind] * //tex2D<float>(patches, whichPatch, ind) *
                    tex3D<float>(f, voxelIndices.z + dk, voxelIndices.y + dj, voxelIndices.x + di);
                         //tex3D<float>(f, voxelIndices.x + di, voxelIndices.y + dj, voxelIndices.z + dk);
            }
        }
    }
    return retVal;
}

__device__ void copyVolumePatch(cudaTextureObject_t f, const int3 voxelIndices, const int3 patchSize, float* patch_copy)
{
    for (int di = 0; di < patchSize.x; di++)
    {
        for (int dj = 0; dj < patchSize.y; dj++)
        {
            for (int dk = 0; dk < patchSize.z; dk++)
                patch_copy[di * patchSize.y * patchSize.z + dj * patchSize.z + dk] =
                tex3D<float>(f, voxelIndices.z + dk, voxelIndices.y + dj, voxelIndices.x + di);
                    //tex3D<float>(f, voxelIndices.x + di, voxelIndices.y + dj, voxelIndices.z + dk);
        }
    }
}

__device__ void OMPsinglePatch_basis(cudaTextureObject_t f, const float* patches, const int3 voxelIndices, const int numPatches,
                                       const int3 patchSize, const float epsilon, const int sparsityThreshold, int* indexMap, float* f_dot_d, float* patchWeights)
{
    // Initialize residual
    //copyVolumePatch(f, voxelIndices, patchSize, Rf);

    for (int n = 0; n < numPatches; n++) indexMap[n] = n;
    for (int n = 0; n < sparsityThreshold; n++) patchWeights[n] = 0.0f;

    float f_dot_f = 0.0f;
    for (int i = 0; i < numPatches; i++)
    {
        f_dot_d[i] = innerProduct(f, patches, voxelIndices, i, patchSize);
        f_dot_f += f_dot_d[i]* f_dot_d[i];
    }
    float residual = f_dot_f;

    int k = 0;
    for (; k < sparsityThreshold; k++)
    {
        int maxInd = 0;
        float maxVal = fabs(f_dot_d[0]);
        for (int i = 0; i < numPatches; i++)
        {
            if (fabs(f_dot_d[i]) > maxVal)
            {
                maxInd = i;
                maxVal = fabs(f_dot_d[i]);
            }
        }
        maxVal = f_dot_d[maxInd];
        patchWeights[k] = maxVal;
        indexMap[k] = maxInd;

        f_dot_d[maxInd] = 0.0f; // make sure this patch is not used again

        residual -= maxVal * maxVal;
        if (residual < epsilon * f_dot_f)
            break;
    }
}

__device__ float* OMPsinglePatch(cudaTextureObject_t f, const float* patches, const float* innerProductPairs, const int3 voxelIndices,
                               const int numPatches, const int3 patchSize, const float epsilon, const int sparsityThreshold,
                               int* indexMap, float* Rf, float* f_dot_d, float* A_k_inv)
{
    const int numPatchPixels = patchSize.x * patchSize.y * patchSize.z;

    // Pati, Rezaiifar, and Krishnaprasad, Orthogonal Matching Pursuit, 1993

    for (int n = 0; n < numPatches; n++) indexMap[n] = n;

    // Initialize residual
    copyVolumePatch(f, voxelIndices, patchSize, Rf);

    // Pre-calculate dot product of current volume patch with every element in the dictionary
    //for (int i = 0; i < numPatches; i++) f_dot_d[i] = innerProduct(f, patches, voxelIndices, i, patchSize);
    float f_dot_f = 0.0f;
    for (int i = 0; i < numPatches; i++)
    {
        f_dot_d[i] = innerProduct(f, patches, voxelIndices, i, patchSize);
        f_dot_f += f_dot_d[i] * f_dot_d[i];
    }

    float delta = 1.0e-12;
    //float delta = 1.0e-6;
    int A_k_inv_num_rows_max = sparsityThreshold;
    for (int n = 0; n < A_k_inv_num_rows_max * A_k_inv_num_rows_max + 3 * A_k_inv_num_rows_max + 1; n++)
        A_k_inv[n] = 0.0f;
    float* b_k = &A_k_inv[A_k_inv_num_rows_max * A_k_inv_num_rows_max];
    float* v_k = &A_k_inv[A_k_inv_num_rows_max * A_k_inv_num_rows_max + A_k_inv_num_rows_max];
    float* a_k = &A_k_inv[A_k_inv_num_rows_max * A_k_inv_num_rows_max + 2 * A_k_inv_num_rows_max];

    int k = 0;
    for (; k < sparsityThreshold; k++)
    {
        if (k >= sparsityThreshold) break;

        float curMaxVal = 0.0f;
        int n_k_plus_one = 0;
        float Rf_dot_Dk = 0.0f;
        for (int i = k; i < numPatches; i++)
        {
            // double Rf_dot_x = Rf->innerProduct(D->at(i));

            //*
            float Rf_dot_x = 0.0f;
            const float* curPatch = &patches[indexMap[i] * numPatchPixels];
            for (int l = 0; l < numPatchPixels; l++)
                Rf_dot_x += Rf[l] * curPatch[l];
            //*/
            //float Rf_dot_x = f_dot_d[indexMap[i]];
            if (fabs(Rf_dot_x) > curMaxVal)
            {
                n_k_plus_one = i;
                Rf_dot_Dk = Rf_dot_x;
                curMaxVal = fabs(Rf_dot_x);
            }
        }
        // printf("maximum inner product: %f\n", curMaxVal);
        if (curMaxVal < delta)
        {
            k -= 1;
            break;
        }

        // Swap n_k_plus_one and k+1
        int index_save = indexMap[k];
        indexMap[k] = indexMap[n_k_plus_one];
        indexMap[n_k_plus_one] = index_save;

        // Calculate A_k_inv and b_k
        if (k == 0)
        {
        }
        else if (k == 1)
        {
            A_k_inv[0] = 1.0f;
            v_k[0] = innerProductPairs[indexMap[1] * numPatches + indexMap[0]];
            b_k[0] = A_k_inv[0] * v_k[0];
        }
        else
        {
            float vk_dot_bk = 0.0f;
            for (int i = 0; i < k - 1; i++) vk_dot_bk += v_k[i] * b_k[i];
            float beta = 1.0f / (1.0f - vk_dot_bk);
            for (int i = 0; i < k - 1; i++)
            {
                for (int j = 0; j < k - 1; j++)
                    A_k_inv[i * A_k_inv_num_rows_max + j] += beta * b_k[i] * b_k[j];
                A_k_inv[i * A_k_inv_num_rows_max + k - 1] = -beta * b_k[i];
            }
            for (int j = 0; j < k - 1; j++)
                A_k_inv[(k - 1) * A_k_inv_num_rows_max + j] = -beta * b_k[j];
            A_k_inv[(k - 1) * A_k_inv_num_rows_max + k - 1] = beta;

            for (int i = 0; i < k; i++)
                v_k[i] = innerProductPairs[indexMap[i] * numPatches + indexMap[k]];

            for (int i = 0; i < k; i++)
            {
                b_k[i] = 0.0f;
                for (int j = 0; j < k; j++)
                    b_k[i] += A_k_inv[i * A_k_inv_num_rows_max + j] * v_k[j];
            }
        }

        // Update Model
        if (k == 0)
        {
            // a_k[0] = Rf->innerProduct(D->at(k));
            a_k[0] = Rf_dot_Dk;
        }
        else
        {
            float gamma_dot_gamma = innerProductPairs[indexMap[k] * numPatches + indexMap[k]];
            for (int i = 0; i < k; i++)
            {
                gamma_dot_gamma -= 2.0f * b_k[i] * innerProductPairs[indexMap[i] * numPatches + indexMap[k]];
                for (int j = 0; j < k; j++)
                    gamma_dot_gamma += b_k[i] * b_k[j] * innerProductPairs[indexMap[i] * numPatches + indexMap[j]];
            }
            // double alpha = Rf->innerProduct(D->at(k)) / gamma_dot_gamma;
            float alpha = Rf_dot_Dk / gamma_dot_gamma;

            for (int n = 0; n < k; n++) a_k[n] -= alpha * b_k[n];
            a_k[k] = alpha;
        }


        copyVolumePatch(f, voxelIndices, patchSize, Rf);
        for (int n = 0; n <= k; n++)
        {
            for (int i = 0; i < numPatchPixels; i++)
            {
                Rf[i] -= a_k[n] * patches[indexMap[n] * numPatchPixels + i];
            }
        }

        if (epsilon > 0.0)
        {
            float Rf_dot_Rf = 0.0f;
            for (int i = 0; i < numPatchPixels; i++) Rf_dot_Rf += Rf[i] * Rf[i];
            if (Rf_dot_Rf < epsilon * f_dot_f)
            {
                k += 1;
                break;
            }
        }
    }

    return a_k;
}

__global__ void OMP_basis(cudaTextureObject_t f, const float* patches,
    const int3 N, const int numPatches, const int3 patchSize, const float epsilon,
    const int sparsityThreshold, const int3 patchShift, float* Df, const int3 N_tiles,
    int* indexMap, float* f_dot_d, float* patchWeights)
{
    // if patches cause overlap, call a series of jobs where each job covers the whole volume
    // then we can rebuild the sparse representation and then just sum the overlaps at the end
    // patchShift \in {1, patchSize/2, patchSize}
    // first assume that patchShift = 1

    const int i_patch = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_patch = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_patch = threadIdx.z + blockIdx.z * blockDim.z;

    const int i = i_patch * patchSize.x + patchShift.x - (patchSize.x - 1) / 2;
    const int j = j_patch * patchSize.y + patchShift.y - (patchSize.y - 1) / 2;
    const int k = k_patch * patchSize.z + patchShift.z - (patchSize.z - 1) / 2;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    //if (i < 0 || j < 0 || k < 0) return;

    const int ind = i_patch * N_tiles.z * N_tiles.y + j_patch * N_tiles.z + k_patch;
    int* indexMap_0 = &indexMap[ind * numPatches];
    float* f_dot_d_0 = &f_dot_d[ind * numPatches];
    float* patchWeights_0 = &patchWeights[ind * sparsityThreshold];

    const int numPatchPixels = patchSize.x * patchSize.y * patchSize.z;
    const float numberOfOverlaps = 1.0f / float(numPatchPixels);
    const int3 voxelIndices = make_int3(i, j, k);
    OMPsinglePatch_basis(f, patches, voxelIndices, numPatches, patchSize, epsilon, sparsityThreshold, indexMap_0, f_dot_d_0, patchWeights_0);
    for (int di = 0; di < patchSize.x; di++)
    {
        const int ii = i + di;
        if (ii >= 0 && ii < N.x)
        {
            for (int dj = 0; dj < patchSize.y; dj++)
            {
                const int jj = j + dj;
                if (jj >= 0 && jj < N.y)
                {
                    for (int dk = 0; dk < patchSize.z; dk++)
                    {
                        const int kk = k + dk;
                        if (kk >= 0 && kk < N.z)
                        {
                            const int ind = di * patchSize.y * patchSize.z + dj * patchSize.z + dk;
                            float accum = 0.0f;
                            for (int n = 0; n < sparsityThreshold; n++)
                            {
                                if (patchWeights_0[n] != 0.0f)
                                    accum += patchWeights_0[n] * patches[indexMap_0[n] * numPatchPixels + ind]; //tex2D<float>(patches, n, ind);
                            }
                            Df[ii * N.z * N.y + jj * N.z + kk] += accum * numberOfOverlaps;
                        }
                    }
                }
            }
        }
    }
}

__global__ void OMP(cudaTextureObject_t f, const float* patches, const float* innerProductPairs,
                    const int3 N, const int numPatches, const int3 patchSize, const float epsilon,
                    const int sparsityThreshold, const int3 patchShift, float* Df, const int3 N_tiles,
                    int* indexMap, float* Rf, float* f_dot_d, float* A_k_inv)
{
    // if patches cause overlap, call a series of jobs where each job covers the whole volume
    // then we can rebuild the sparse representation and then just sum the overlaps at the end
    // patchShift \in {1, patchSize/2, patchSize}
    // first assume that patchShift = 1

    const int i_patch = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_patch = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_patch = threadIdx.z + blockIdx.z * blockDim.z;

    const int i = i_patch * patchSize.x + patchShift.x - (patchSize.x - 1) / 2;
    const int j = j_patch * patchSize.y + patchShift.y - (patchSize.y - 1) / 2;
    const int k = k_patch * patchSize.z + patchShift.z - (patchSize.z - 1) / 2;
    if (i >= N.x || j >= N.y || k >= N.z) return;
    //if (i < 0 || j < 0 || k < 0) return;

    const int ind = i_patch * N_tiles.z * N_tiles.y + j_patch * N_tiles.z + k_patch;
    int* indexMap_0 = &indexMap[ind * numPatches];
    float* Rf_0 = &Rf[ind * patchSize.x * patchSize.y * patchSize.z];
    float* f_dot_d_0 = &f_dot_d[ind * numPatches];
    float* A_k_inv_0 = &A_k_inv[ind * (sparsityThreshold * sparsityThreshold + 3 * sparsityThreshold + 1)];

    const int numPatchPixels = patchSize.x * patchSize.y * patchSize.z;
    const float numberOfOverlaps = 1.0f / float(numPatchPixels);
    const int3 voxelIndices = make_int3(i, j, k);
    float* patchWeights = OMPsinglePatch(f, patches, innerProductPairs, voxelIndices, numPatches, patchSize, epsilon, sparsityThreshold, indexMap_0, Rf_0, f_dot_d_0, A_k_inv_0);
    for (int di = 0; di < patchSize.x; di++)
    {
        const int ii = i+di;
        if (ii >= 0 && ii < N.x)
        {
            for (int dj = 0; dj < patchSize.y; dj++)
            {
                const int jj = j + dj;
                if (jj >= 0 && jj < N.y)
                {
                    for (int dk = 0; dk < patchSize.z; dk++)
                    {
                        const int kk = k + dk;
                        if (kk >= 0 && kk < N.z)
                        {
                            const int ind = di * patchSize.y * patchSize.z + dj * patchSize.z + dk;
                            float accum = 0.0f;
                            for (int n = 0; n < sparsityThreshold; n++)
                            {
                                if (patchWeights[n] != 0.0f)
                                    accum += patchWeights[n] * patches[indexMap_0[n] * numPatchPixels + ind]; //tex2D<float>(patches, n, ind);
                            }
                            Df[ii * N.z * N.y + jj * N.z + kk] += accum * numberOfOverlaps;
                        }
                    }
                }
            }
        }
    }
}

bool calcInnerProductPairs(float* dictionary, int numElements, int num1, int num2, int num3, float* innerProductPairs)
{
    bool isOrthonormal = true;
    int numPatchPixels = num1 * num2 * num3;
    for (int i = 0; i < numElements; i++)
    {
        float* basis_1 = &dictionary[i * numPatchPixels];
        for (int j = i; j < numElements; j++)
        {
            float* basis_2 = &dictionary[j * numPatchPixels];

            double accum = 0.0;
            for (int n = 0; n < numPatchPixels; n++)
            {
                accum += basis_1[n] * basis_2[n];
            }
            innerProductPairs[i * numElements + j] = float(accum);
            innerProductPairs[j * numElements + i] = float(accum);

            // Check is orthonormal
            if (i == j)
            {
                if (fabs(accum - 1.0) > 1.0e-6)
                {
                    //printf("elements %d is not normalized\n", i);
                    isOrthonormal = false;
                }
            }
            else
            {
                if (fabs(accum) > 0.0015)
                {
                    //printf("elements %d and %d are not ortogonal (%f)\n", i, j, accum);
                    isOrthonormal = false;
                }
            }
        }
    }
    if (numElements > numPatchPixels)
        isOrthonormal = false;
    return isOrthonormal;
}

double matchingPursuit_memory(int N_1, int N_2, int N_3, int numElements, int num1, int num2, int num3, int sparsityThreshold)
{
    // Calculate the number of non-overlapping patches across the whole volume
    int3 N_tiles = make_int3(int(ceil(float(N_1) / float(num1))), int(ceil(float(N_2) / float(num2))), int(ceil(float(N_3) / float(num3))));
    uint64 N_tiles_prod = uint64(N_tiles.x) * uint64(N_tiles.y) * uint64(N_tiles.z);

    int numPatchPixels = num1 * num2 * num3;

    //printf("N_tiles_prod = %d\n", int(N_tiles_prod));

    uint64 vol_size = uint64(N_1) * uint64(N_2) * uint64(N_3);
    uint64 N = 2 * vol_size + uint64(numElements + numPatchPixels + numElements + (sparsityThreshold * sparsityThreshold + 3 * sparsityThreshold + 1)) * N_tiles_prod;
    return 4.0 * double(N) / pow(2.0, 30.0);
}

bool matchingPursuit(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int num1, int num2, int num3, float epsilon, int sparsityThreshold, bool data_on_cpu, int whichGPU)
{
    if (f == NULL) return false;

    //printf("data size: %d x %d x %d\n", N_1, N_2, N_3);
    //printf("patch size(%d): %d x %d x %d\n", numElements, num1, num2, num3);

    // Calculate inner product pairs
    float* innerProductPairs = new float[numElements * numElements];
    bool isOrthonormal = calcInnerProductPairs(dictionary, numElements, num1, num2, num3, innerProductPairs);
    //*
    if (isOrthonormal)
    {
        delete[] innerProductPairs;
        return matchingPursuit_basis(f, N_1, N_2, N_3, dictionary, numElements, num1, num2, num3, epsilon, sparsityThreshold, data_on_cpu, whichGPU);
    }
    //*/

    cudaSetDevice(whichGPU);

    // Calculate the number of non-overlapping patches across the whole volume
    int3 N_tiles = make_int3(int(ceil(float(N_1) / float(num1))), int(ceil(float(N_2) / float(num2))), int(ceil(float(N_3) / float(num3))));
    uint64 N_tiles_prod = uint64(N_tiles.x) * uint64(N_tiles.y) * uint64(N_tiles.z);

    // Number of pixels in a patch
    int numPatchPixels = num1 * num2 * num3;

    // Allocate space for temporary arrays
    //int* indexMap = (int*)malloc(size_t(numPatches) * sizeof(int));
    int* dev_indexMap = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_indexMap, numElements * N_tiles_prod * sizeof(int)))
        fprintf(stderr, "cudaMalloc failed!\n");

    //float* Rf = (float*)malloc(size_t(numPatchPixels) * sizeof(float));
    float* dev_Rf = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_Rf, numPatchPixels * N_tiles_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    //float* f_dot_d = (float*)malloc(size_t(numPatches) * sizeof(float));
    float* dev_f_dot_d = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_f_dot_d, numElements * N_tiles_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    //float* A_k_inv = (float*)malloc(size_t(A_k_inv_num_rows_max * A_k_inv_num_rows_max + 3 * A_k_inv_num_rows_max + 1) * sizeof(float));
    int A_k_inv_num_rows_max = sparsityThreshold;
    float* dev_A_k_inv = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_A_k_inv, (A_k_inv_num_rows_max * A_k_inv_num_rows_max + 3 * A_k_inv_num_rows_max + 1) * N_tiles_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    //float* dev_patchWeights = 0;
    //if (cudaSuccess != cudaMalloc((void**)&dev_patchWeights, sparsityThreshold * N_tiles_prod * sizeof(float)))
    //    fprintf(stderr, "cudaMalloc failed!\n");

    // Copy inner product pairs to GPU
    float* dev_innerProductPairs = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_innerProductPairs, numElements * numElements * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_innerProductPairs, innerProductPairs, numElements * numElements * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(innerProductPairs) failed!\n");
    delete[] innerProductPairs;

    // Copy dictionary to GPU
    float* dev_dictionary = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_dictionary, numElements * numPatchPixels * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_dictionary, dictionary, numElements * numPatchPixels * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(dictionary) failed!\n");

    /*
    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;
    //*/

    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    int3 patchSize = make_int3(num1, num2, num3);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }
    setToConstant(dev_Df, 0.0, N, whichGPU);

    // Call kernel
    dim3 dimBlock = setBlockSize(N_tiles);
    dim3 dimGrid = setGridSize(N_tiles, dimBlock);
    
    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N, true, false, false);

    //BlurFilter1DKernel <<<dimGrid, dimBlock >>> (dev_f, dev_Df, N, FWHM);
    for (int shift_x = 0; shift_x < patchSize.x; shift_x++)
    {
        for (int shift_y = 0; shift_y < patchSize.y; shift_y++)
        {
            for (int shift_z = 0; shift_z < patchSize.z; shift_z++)
            {
                //if (shift_x == 0 && shift_y == 0 && shift_z == 0)
                {
                    int3 patchShift = make_int3(shift_x, shift_y, shift_z);
                    OMP <<< dimGrid, dimBlock >>> (d_data_txt, dev_dictionary, dev_innerProductPairs, N, numElements, patchSize,
                        epsilon, sparsityThreshold, patchShift, dev_Df, N_tiles, dev_indexMap, dev_Rf, dev_f_dot_d, dev_A_k_inv);
                }
            }
        }
    }

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        /*
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else //*/
            pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_innerProductPairs);
    cudaFree(dev_dictionary);
    cudaFree(dev_indexMap);
    cudaFree(dev_Rf);
    cudaFree(dev_f_dot_d);
    cudaFree(dev_A_k_inv);
    //cudaFree(dev_patchWeights);

    return true;
}

bool matchingPursuit_basis(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int num1, int num2, int num3, float epsilon, int sparsityThreshold, bool data_on_cpu, int whichGPU)
{
    //printf("using basis version\n");
    if (f == NULL) return false;

    //printf("basis is orthonormal, using faster algorithm\n");
    //printf("data size: %d x %d x %d\n", N_1, N_2, N_3);
    //printf("patch size(%d): %d x %d x %d\n", numElements, num1, num2, num3);

    // Calculate the number of non-overlapping patches across the whole volume
    int3 N_tiles = make_int3(int(ceil(float(N_1) / float(num1))), int(ceil(float(N_2) / float(num2))), int(ceil(float(N_3) / float(num3))));
    uint64 N_tiles_prod = uint64(N_tiles.x) * uint64(N_tiles.y) * uint64(N_tiles.z);

    // Number of pixels in a patch
    int numPatchPixels = num1 * num2 * num3;

    cudaSetDevice(whichGPU);

    // Allocate space for temporary arrays
    //int* indexMap = (int*)malloc(size_t(numPatches) * sizeof(int));
    int* dev_indexMap = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_indexMap, numElements * N_tiles_prod * sizeof(int)))
        fprintf(stderr, "cudaMalloc failed!\n");

    //float* f_dot_d = (float*)malloc(size_t(numPatches) * sizeof(float));
    float* dev_f_dot_d = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_f_dot_d, numElements * N_tiles_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    float* dev_patchWeights = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_patchWeights, sparsityThreshold * N_tiles_prod * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");

    // Copy dictionary to GPU
    float* dev_dictionary = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_dictionary, numElements * numPatchPixels * sizeof(float)))
        fprintf(stderr, "cudaMalloc failed!\n");
    if (cudaMemcpy(dev_dictionary, dictionary, numElements * numPatchPixels * sizeof(float), cudaMemcpyHostToDevice))
        fprintf(stderr, "cudaMemcpy(dictionary) failed!\n");

    /*
    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = N_1 - 1;
    sliceStart = max(0, min(N_1 - 1, sliceStart));
    sliceEnd = max(0, min(N_1 - 1, sliceEnd));
    if (sliceStart > sliceEnd)
        return false;
    //*/

    //cudaError_t cudaStatus;

    // Copy volume to GPU
    int3 N = make_int3(N_1, N_2, N_3);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copy3DdataToGPU(f, N, whichGPU);
    else
        dev_f = f;

    int3 patchSize = make_int3(num1, num2, num3);

    // Allocate space on GPU for the gradient
    float* dev_Df = 0;
    if (cudaMalloc((void**)&dev_Df, uint64(N.x) * uint64(N.y) * uint64(N.z) * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume %d x %d x %d) failed!\n", N_1, N_2, N_3);
        return false;
    }
    setToConstant(dev_Df, 0.0, N, whichGPU);

    // Call kernel
    dim3 dimBlock = setBlockSize(N_tiles);
    dim3 dimGrid = setGridSize(N_tiles, dimBlock);

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N, true, false, false);

    //BlurFilter1DKernel <<<dimGrid, dimBlock >>> (dev_f, dev_Df, N, FWHM);
    for (int shift_x = 0; shift_x < patchSize.x; shift_x++)
    {
        for (int shift_y = 0; shift_y < patchSize.y; shift_y++)
        {
            for (int shift_z = 0; shift_z < patchSize.z; shift_z++)
            {
                //if (shift_x == 0 && shift_y == 0 && shift_z == 0)
                {
                    int3 patchShift = make_int3(shift_x, shift_y, shift_z);
                    OMP_basis <<< dimGrid, dimBlock >>> (d_data_txt, dev_dictionary, N, numElements, patchSize,
                        epsilon, sparsityThreshold, patchShift, dev_Df, N_tiles, dev_indexMap, dev_f_dot_d, dev_patchWeights);
                }
            }
        }
    }

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    if (data_on_cpu)
    {
        // pull result off GPU
        /*
        if (f_out != NULL)
        {
            float* dev_Df_shift = &dev_Df[uint64(sliceStart) * uint64(N.y) * uint64(N.z)];
            int3 N_crop = make_int3(sliceEnd - sliceStart + 1, N_2, N_3);
            pull3DdataFromGPU(f_out, N_crop, dev_Df_shift, whichGPU);
        }
        else //*/
        pull3DdataFromGPU(f, N, dev_Df, whichGPU);

        if (dev_f != 0)
            cudaFree(dev_f);
    }
    else
    {
        // copy dev_Df to dev_f
        cudaMemcpy(dev_f, dev_Df, sizeof(float) * uint64(N.x) * uint64(N.y) * uint64(N.z), cudaMemcpyDeviceToDevice);
        //cudaDeviceSynchronize();
    }
    if (dev_Df != 0)
    {
        cudaFree(dev_Df);
    }
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_dictionary);
    cudaFree(dev_indexMap);
    cudaFree(dev_f_dot_d);
    cudaFree(dev_patchWeights);

    return true;
}
