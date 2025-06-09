////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
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

__constant__ int d_scatter_job_type;

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

__device__ float airScan(const float3 x_0, const float3 x_f, const float3 n_d, const int N_energies, cudaTextureObject_t source_txt, cudaTextureObject_t energies_txt, cudaTextureObject_t detector_txt)
{
    float val = 0.0f;
    const float3 x_f_minus_x_0 = make_float3(x_f.x - x_0.x, x_f.y - x_0.y, x_f.z - x_0.z);
    const float x_f_minus_x_0_dot_n_d = x_f_minus_x_0.x * n_d.x + x_f_minus_x_0.y * n_d.y + x_f_minus_x_0.z * n_d.z;
    const float x_f_minus_x_0_mag_inv = rsqrtf(x_f_minus_x_0.x * x_f_minus_x_0.x + x_f_minus_x_0.y * x_f_minus_x_0.y + x_f_minus_x_0.z * x_f_minus_x_0.z);
    const float direct_solid_angle = x_f_minus_x_0_dot_n_d * x_f_minus_x_0_mag_inv * x_f_minus_x_0_mag_inv * x_f_minus_x_0_mag_inv;
    for (int igamma = 0; igamma < N_energies; igamma++)
    {
        const float spec = tex1D<float>(source_txt, igamma); // spec = spectrum
        if (spec > 0.0f)
        {
            const float gamma_0 = tex1D<float>(energies_txt, igamma);
            val += tex1D<float>(detector_txt, gamma_0 - 0.5f) * spec;
        }
    }
    return val * direct_solid_angle;
}

__device__ float PrimaryScan(const float3 x_0, const float3 x_f, const float3 n_d, const int N_energies,
    cudaTextureObject_t source_txt, cudaTextureObject_t energies_txt, cudaTextureObject_t detector_txt,
    cudaTextureObject_t sigma_PE_txt, cudaTextureObject_t sigma_CS_txt, cudaTextureObject_t sigma_RS_txt,
    cudaTextureObject_t f, const hypercube* f_params)
{
    const float Prho = divergentBeamTransform(f, f_params->N, f_params->T, f_params->startVal, x_f, x_0);

    float val = 0.0f;
    const float3 x_f_minus_x_0 = make_float3(x_f.x - x_0.x, x_f.y - x_0.y, x_f.z - x_0.z);
    const float x_f_minus_x_0_dot_n_d = x_f_minus_x_0.x * n_d.x + x_f_minus_x_0.y * n_d.y + x_f_minus_x_0.z * n_d.z;
    const float x_f_minus_x_0_mag_inv = rsqrtf(x_f_minus_x_0.x * x_f_minus_x_0.x + x_f_minus_x_0.y * x_f_minus_x_0.y + x_f_minus_x_0.z * x_f_minus_x_0.z);
    const float direct_solid_angle = x_f_minus_x_0_dot_n_d * x_f_minus_x_0_mag_inv * x_f_minus_x_0_mag_inv * x_f_minus_x_0_mag_inv;
    for (int igamma = 0; igamma < N_energies; igamma++)
    {
        const float spec = tex1D<float>(source_txt, igamma); // spec = spectrum
        if (spec > 0.0f)
        {
            const float gamma_0 = tex1D<float>(energies_txt, igamma);

            const float sigma_PE_gamma_0 = tex1D<float>(sigma_PE_txt, gamma_0 - 0.5f);
            const float sigma_CS_gamma_0 = tex1D<float>(sigma_CS_txt, gamma_0 - 0.5f);
            const float sigma_RS_gamma_0 = tex1D<float>(sigma_RS_txt, gamma_0 - 0.5f);
            const float sigma_total_gamma_0 = sigma_PE_gamma_0 + sigma_CS_gamma_0 + sigma_RS_gamma_0;

            val += tex1D<float>(detector_txt, gamma_0 - 0.5f) * spec * expf(-sigma_total_gamma_0 * Prho);
        }
    }
    return val * direct_solid_angle;
}

__global__ void firstOrderScatterModel(float* dev_proj, const hypercube* g_params,
    cudaTextureObject_t f, const hypercube* f_params, const float* Df, const float* source_and_detector,
    cudaTextureObject_t source_txt, cudaTextureObject_t energies_txt, cudaTextureObject_t detector_txt, cudaTextureObject_t sigma_PE_txt,
    cudaTextureObject_t sigma_CS_txt, cudaTextureObject_t sigma_RS_txt, cudaTextureObject_t scatterDist_txt)
{
    const int m = threadIdx.x + blockIdx.x * blockDim.x; // rows
    const int n = threadIdx.y + blockIdx.y * blockDim.y; // columns
    if (m >= g_params->N.y || n >= g_params->N.z)
        return;

    const float t = m * g_params->T.y + g_params->startVal.y;
    const float s = n * g_params->T.z + g_params->startVal.z;

    // moduleCenter = source_and_detector[3,4,5]
    // v_vec = source_and_detector[6,7,8]
    // u_vec = source_and_detector[9,10,11]
    const float3 x_0 = make_float3(source_and_detector[0], source_and_detector[1], source_and_detector[2]);
    const float3 x_f = make_float3(source_and_detector[3] + t * source_and_detector[6] + s * source_and_detector[9], source_and_detector[4] + t * source_and_detector[7] + s * source_and_detector[10], source_and_detector[5] + t * source_and_detector[8] + s * source_and_detector[11]);

    const float3 n_d = make_float3(-1.0f*(source_and_detector[10] * source_and_detector[8] - source_and_detector[11] * source_and_detector[7]),
        -1.0f * (source_and_detector[11] * source_and_detector[6] - source_and_detector[9] * source_and_detector[8]),
            -1.0f * (source_and_detector[9] * source_and_detector[7] - source_and_detector[10] * source_and_detector[6]));

    // Calcuate the scatter response
    float val = 0.0f;
    for (int k = 0; k < f_params->N.z; k++)
    {
        const float x_3 = k * f_params->T.z + f_params->startVal.z;
        for (int j = 0; j < f_params->N.y; j++)
        {
            const float x_2 = j * f_params->T.y + f_params->startVal.y;
            for (int i = 0; i < f_params->N.x; i++)
            {
                const float3 x = make_float3(i * f_params->T.x + f_params->startVal.x, x_2, x_3);

                const float rho = tex3D<float>(f, i + 0.5f, j + 0.5f, k + 0.5f);

                if (rho > 0.0f)
                {
                    uint64 ind = uint64(k) * uint64(f_params->N.y * f_params->N.x) + uint64(j * f_params->N.x + i);
                    const float Df_firstLeg = Df[ind];
                    const float Df_secondLeg = divergentBeamTransform(f, f_params->N, f_params->T, f_params->startVal, x, x_f);
                    
                    const float3 r_0 = make_float3(x.x - x_0.x, x.y - x_0.y, x.z - x_0.z);
                    const float mag_r_0_inv = rsqrtf(r_0.x * r_0.x + r_0.y * r_0.y + r_0.z * r_0.z);
                    const float3 r_f = make_float3(x_f.x - x.x, x_f.y - x.y, x_f.z - x.z);
                    const float mag_r_f_inv = rsqrtf(r_f.x * r_f.x + r_f.y * r_f.y + r_f.z * r_f.z);

                    const float r_0_dot_detectorNormal = r_0.x * n_d.x + r_0.y * n_d.y + r_0.z * n_d.z;
                    const float r_f_dot_detectorNormal = r_f.x * n_d.x + r_f.y * n_d.y + r_f.z * n_d.z;
                    const float one_over_r_0_norm_mult_r_f_norm = mag_r_0_inv * mag_r_f_inv;
                    const float totalSolidAngle = r_0_dot_detectorNormal * r_f_dot_detectorNormal * one_over_r_0_norm_mult_r_f_norm * one_over_r_0_norm_mult_r_f_norm * one_over_r_0_norm_mult_r_f_norm;

                    const float cos_theta = (r_0.x * r_f.x + r_0.y * r_f.y + r_0.z * r_f.z) * one_over_r_0_norm_mult_r_f_norm;
                    const float theta = acos(cos_theta)*RAD_TO_DEG;

                    float val_inner = 0.0f;
                    for (int igamma = 0; igamma < g_params->N.w; igamma++)
                    {
                        const float spec = tex1D<float>(source_txt, igamma); // spec = spectrum
                        if (spec > 0.0f)
                        {
                            const float gamma_0 = tex1D<float>(energies_txt, igamma);
                            const float gamma_f = 510.975f * gamma_0 / (510.975f + (1.0f - cos_theta) * gamma_0);

                            const float sigma_PE_gamma_0 = tex1D<float>(sigma_PE_txt, gamma_0 - 0.5f);
                            const float sigma_CS_gamma_0 = tex1D<float>(sigma_CS_txt, gamma_0 - 0.5f);
                            const float sigma_RS_gamma_0 = tex1D<float>(sigma_RS_txt, gamma_0 - 0.5f);
                            const float sigma_total_gamma_0 = sigma_PE_gamma_0 + sigma_CS_gamma_0 + sigma_RS_gamma_0;

                            const float cur_CS = tex1D<float>(detector_txt, gamma_f - 0.5f) * sigma_CS_gamma_0 * tex3D<float>(scatterDist_txt, theta + 0.5f, gamma_0 - 0.5f, 0.5f);
                            const float cur_RS = tex1D<float>(detector_txt, gamma_0 - 0.5f) * sigma_RS_gamma_0 * tex3D<float>(scatterDist_txt, theta + 0.5f, gamma_0 - 0.5f, 1.5f);

                            //val_inner += spec * (cur_CS + cur_RS) * expf(-sigma_total_gamma_0 * (Df_firstLeg + Df_secondLeg));
                            //*
                            const float sigma_PE_gamma_f = tex1D<float>(sigma_PE_txt, gamma_f - 0.5f);
                            const float sigma_CS_gamma_f = tex1D<float>(sigma_CS_txt, gamma_f - 0.5f);
                            const float sigma_RS_gamma_f = tex1D<float>(sigma_RS_txt, gamma_f - 0.5f);
                            const float sigma_total_gamma_f = sigma_PE_gamma_f + sigma_CS_gamma_f + sigma_RS_gamma_f;
                            val_inner += spec * cur_CS * expf(-sigma_total_gamma_0 * Df_firstLeg - sigma_total_gamma_f * Df_secondLeg) +
                            spec * cur_RS * expf(-sigma_total_gamma_0 * (Df_firstLeg + Df_secondLeg));
                            //*/
                        }
                    }
                    val += val_inner * rho * totalSolidAngle;
                }
            }
        }
    }
    const float val_firstOrderScatter = val * f_params->T.x * f_params->T.y * f_params->T.z;

    if (d_scatter_job_type == 0)
    {
        // scatter transmission
        const float val_airScan = airScan(x_0, x_f, n_d, g_params->N.w, source_txt, energies_txt, detector_txt);
        dev_proj[m * g_params->N.z + n] = val_firstOrderScatter / val_airScan;
    }
    else
    {
        float val_primary = PrimaryScan(x_0, x_f, n_d, g_params->N.w, source_txt, energies_txt, detector_txt,
            sigma_PE_txt, sigma_CS_txt, sigma_RS_txt, f, f_params);

        if (d_scatter_job_type < 0)
        {
            // scatter correction
            dev_proj[m * g_params->N.z + n] = val_primary / (val_primary + val_firstOrderScatter);
        }
        else
        {
            // add scatter
            dev_proj[m * g_params->N.z + n] = (val_primary + val_firstOrderScatter) / val_primary;
        }
    }
}

__global__ void firstOrderScatterModel_fast(float* dev_proj, const hypercube* g_params,
    cudaTextureObject_t f, const hypercube* f_params, const float* Df, const float* source_and_detector,
    cudaTextureObject_t source_txt, cudaTextureObject_t energies_txt, cudaTextureObject_t detector_txt, cudaTextureObject_t sigma_PE_txt,
    cudaTextureObject_t sigma_CS_txt, cudaTextureObject_t sigma_RS_txt, cudaTextureObject_t scatterDist_txt)
{
    const int m = threadIdx.x + blockIdx.x * blockDim.x; // rows
    const int n = threadIdx.y + blockIdx.y * blockDim.y; // columns
    if (m >= g_params->N.y || n >= g_params->N.z)
        return;

    const float t = m * g_params->T.y + g_params->startVal.y;
    const float s = n * g_params->T.z + g_params->startVal.z;

    // moduleCenter = source_and_detector[3,4,5]
    // v_vec = source_and_detector[6,7,8]
    // u_vec = source_and_detector[9,10,11]
    const float3 x_0 = make_float3(source_and_detector[0], source_and_detector[1], source_and_detector[2]);
    const float3 x_f = make_float3(source_and_detector[3] + t * source_and_detector[6] + s * source_and_detector[9], source_and_detector[4] + t * source_and_detector[7] + s * source_and_detector[10], source_and_detector[5] + t * source_and_detector[8] + s * source_and_detector[11]);

    const float3 n_d = make_float3(-1.0f * (source_and_detector[10] * source_and_detector[8] - source_and_detector[11] * source_and_detector[7]),
        -1.0f * (source_and_detector[11] * source_and_detector[6] - source_and_detector[9] * source_and_detector[8]),
        -1.0f * (source_and_detector[9] * source_and_detector[7] - source_and_detector[10] * source_and_detector[6]));

    // Calcuate the scatter response
    float val = 0.0f;
    for (int k = 0; k < f_params->N.z; k+=3)
    {
        for (int j = 0; j < f_params->N.y; j+=3)
        {
            for (int i = 0; i < f_params->N.x; i+=3)
            {
                const int k_c = min(k + 1, f_params->N.z - 1);
                const int j_c = min(j + 1, f_params->N.y - 1);
                const int i_c = min(i + 1, f_params->N.x - 1);
                const float3 x_c = make_float3(i_c * f_params->T.x + f_params->startVal.x, j_c * f_params->T.y + f_params->startVal.y, k_c * f_params->T.z + f_params->startVal.z);

                const float Df_secondLeg = divergentBeamTransform(f, f_params->N, f_params->T, f_params->startVal, x_c, x_f);

                const int dk_min = max(-1, -k);
                const int dk_max = min(1, f_params->N.z - 1 - k);
                const int dj_min = max(-1, -j);
                const int dj_max = min(1, f_params->N.y - 1 - j);
                const int di_min = max(-1, -i);
                const int di_max = min(1, f_params->N.x - 1 - i);
                for (int dk = dk_min; dk <= dk_max; dk++)
                {
                    const float x_3 = (k+dk) * f_params->T.z + f_params->startVal.z;
                    for (int dj = dj_min; dj <= dj_max; dj++)
                    {
                        const float x_2 = (j+dj) * f_params->T.y + f_params->startVal.y;
                        for (int di = di_min; di <= di_max; di++)
                        {
                            const float3 x = make_float3((i+di) * f_params->T.x + f_params->startVal.x, x_2, x_3);

                            const float rho = tex3D<float>(f, i + 0.5f, j + 0.5f, k + 0.5f);

                            if (rho > 0.0f)
                            {
                                uint64 ind = uint64(k) * uint64(f_params->N.y * f_params->N.x) + uint64(j * f_params->N.x + i);
                                const float Df_firstLeg = Df[ind];

                                const float3 r_0 = make_float3(x.x - x_0.x, x.y - x_0.y, x.z - x_0.z);
                                const float mag_r_0_inv = rsqrtf(r_0.x * r_0.x + r_0.y * r_0.y + r_0.z * r_0.z);
                                const float3 r_f = make_float3(x_f.x - x.x, x_f.y - x.y, x_f.z - x.z);
                                const float mag_r_f_inv = rsqrtf(r_f.x * r_f.x + r_f.y * r_f.y + r_f.z * r_f.z);

                                const float r_0_dot_detectorNormal = r_0.x * n_d.x + r_0.y * n_d.y + r_0.z * n_d.z;
                                const float r_f_dot_detectorNormal = r_f.x * n_d.x + r_f.y * n_d.y + r_f.z * n_d.z;
                                const float one_over_r_0_norm_mult_r_f_norm = mag_r_0_inv * mag_r_f_inv;
                                const float totalSolidAngle = r_0_dot_detectorNormal * r_f_dot_detectorNormal * one_over_r_0_norm_mult_r_f_norm * one_over_r_0_norm_mult_r_f_norm * one_over_r_0_norm_mult_r_f_norm;

                                const float cos_theta = (r_0.x * r_f.x + r_0.y * r_f.y + r_0.z * r_f.z) * one_over_r_0_norm_mult_r_f_norm;
                                const float theta = acos(cos_theta) * RAD_TO_DEG;

                                float val_inner = 0.0f;
                                for (int igamma = 0; igamma < g_params->N.w; igamma++)
                                {
                                    const float spec = tex1D<float>(source_txt, igamma); // spec = spectrum
                                    if (spec > 0.0f)
                                    {
                                        const float gamma_0 = tex1D<float>(energies_txt, igamma);
                                        const float gamma_f = 510.975f * gamma_0 / (510.975f + (1.0f - cos_theta) * gamma_0);

                                        const float sigma_PE_gamma_0 = tex1D<float>(sigma_PE_txt, gamma_0 - 0.5f);
                                        const float sigma_CS_gamma_0 = tex1D<float>(sigma_CS_txt, gamma_0 - 0.5f);
                                        const float sigma_RS_gamma_0 = tex1D<float>(sigma_RS_txt, gamma_0 - 0.5f);
                                        const float sigma_total_gamma_0 = sigma_PE_gamma_0 + sigma_CS_gamma_0 + sigma_RS_gamma_0;

                                        const float cur_CS = tex1D<float>(detector_txt, gamma_f - 0.5f) * sigma_CS_gamma_0 * tex3D<float>(scatterDist_txt, theta + 0.5f, gamma_0 - 0.5f, 0.5f);
                                        const float cur_RS = tex1D<float>(detector_txt, gamma_0 - 0.5f) * sigma_RS_gamma_0 * tex3D<float>(scatterDist_txt, theta + 0.5f, gamma_0 - 0.5f, 1.5f);

                                        //val_inner += spec * (cur_CS + cur_RS) * expf(-sigma_total_gamma_0 * (Df_firstLeg + Df_secondLeg));
                                        //*
                                        const float sigma_PE_gamma_f = tex1D<float>(sigma_PE_txt, gamma_f - 0.5f);
                                        const float sigma_CS_gamma_f = tex1D<float>(sigma_CS_txt, gamma_f - 0.5f);
                                        const float sigma_RS_gamma_f = tex1D<float>(sigma_RS_txt, gamma_f - 0.5f);
                                        const float sigma_total_gamma_f = sigma_PE_gamma_f + sigma_CS_gamma_f + sigma_RS_gamma_f;
                                        val_inner += spec * cur_CS * expf(-sigma_total_gamma_0 * Df_firstLeg - sigma_total_gamma_f * Df_secondLeg) +
                                            spec * cur_RS * expf(-sigma_total_gamma_0 * (Df_firstLeg + Df_secondLeg));
                                        //*/
                                    }
                                }
                                val += val_inner * rho * totalSolidAngle;
                            }
                        }
                    }
                }
            }
        }
    }
    const float val_firstOrderScatter = val * f_params->T.x * f_params->T.y * f_params->T.z;

    if (d_scatter_job_type == 0)
    {
        // scatter transmission
        const float val_airScan = airScan(x_0, x_f, n_d, g_params->N.w, source_txt, energies_txt, detector_txt);
        dev_proj[m * g_params->N.z + n] = val_firstOrderScatter / val_airScan;
    }
    else
    {
        float val_primary = PrimaryScan(x_0, x_f, n_d, g_params->N.w, source_txt, energies_txt, detector_txt,
            sigma_PE_txt, sigma_CS_txt, sigma_RS_txt, f, f_params);

        if (d_scatter_job_type < 0)
        {
            // scatter correction
            dev_proj[m * g_params->N.z + n] = val_primary / (val_primary + val_firstOrderScatter);
        }
        else
        {
            // add scatter
            dev_proj[m * g_params->N.z + n] = (val_primary + val_firstOrderScatter) / val_primary;
        }
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

__global__ void interpolateViews(float* g, const int4 N_g, const float* phis, const bool* projectionCalculated)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    if (projectionCalculated[l])
        return;
    else
    {
        int l_lo, l_hi;
        for (int i = l-1; i >= 0; i--)
        {
            if (projectionCalculated[i])
            {
                l_lo = i;
                break;
            }
        }
        for (int i = l + 1; i < N_g.x; i++)
        {
            if (projectionCalculated[i])
            {
                l_hi = i;
                break;
            }
        }
        const uint64 ind = uint64(l) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n);
        const uint64 ind_lo = uint64(l_lo) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n);
        const uint64 ind_hi = uint64(l_hi) * uint64(N_g.z * N_g.y) + uint64(m * N_g.z + n);

        g[ind] = (phis[l] - phis[l_lo]) / (phis[l_hi] - phis[l_lo]) * g[ind_lo] + (phis[l_hi] - phis[l]) / (phis[l_hi] - phis[l_lo]) * g[ind_hi];
    }
}

bool simulateScatter_firstOrder_singleMaterial(float* g, float* f, parameters* params, float* source, float* energies, int N_energies, float* detector, float* sigma, float* scatterDist, bool data_on_cpu, int jobType)
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

    cudaMemcpyToSymbol(d_scatter_job_type, &(jobType), sizeof(int));

    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);
    N_g.w = N_energies;
    float* dev_g = 0;
    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;
    cudaTextureObject_t f_data_txt = NULL;
    cudaArray* f_data_array = loadTexture(f_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));

    // Allocate data for attenuation in the first leg
    float* dev_Df;
    if ((cudaStatus = cudaMalloc((void**)&dev_Df, params->volumeData_numberOfElements() * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(volume) failed!\n");
    }

    // source: the source spectra
    // energies: the energies of the source spectra
    // detector: the detector response sampled in 1 keV bins
    // sigma: the PE, CS, and RS cross sections sampled in 1 keV bins
    // scatterDist: the CS and RS distributions sampled in 1 keV bins and 0.1 degree angular bins
    float* dev_source = copy1DdataToGPU(source, N_energies, params->whichGPU);
    cudaTextureObject_t source_txt = NULL;
    cudaArray* source_array = loadTexture1D(source_txt, dev_source, N_energies, false, false);

    float* dev_energies = copy1DdataToGPU(energies, N_energies, params->whichGPU);
    cudaTextureObject_t energies_txt = NULL;
    cudaArray* energies_array = loadTexture1D(energies_txt, dev_energies, N_energies, false, false);

    int maxEnergy = int(ceil(energies[N_energies-1]));
    //printf("maxEnergy = %d\n", maxEnergy);

    float* dev_detector = copy1DdataToGPU(detector, maxEnergy, params->whichGPU);
    cudaTextureObject_t detector_txt = NULL;
    cudaArray* detector_array = loadTexture1D(detector_txt, dev_detector, maxEnergy, false, true);

    float* dev_sigma = copy1DdataToGPU(sigma, 3*maxEnergy, params->whichGPU);
    cudaTextureObject_t sigma_PE_txt = NULL;
    cudaArray* sigma_PE_array = loadTexture1D(sigma_PE_txt, &dev_sigma[0 * maxEnergy], maxEnergy, false, true);
    cudaTextureObject_t sigma_CS_txt = NULL;
    cudaArray* sigma_CS_array = loadTexture1D(sigma_CS_txt, &dev_sigma[1 * maxEnergy], maxEnergy, false, true);
    cudaTextureObject_t sigma_RS_txt = NULL;
    cudaArray* sigma_RS_array = loadTexture1D(sigma_RS_txt, &dev_sigma[2 * maxEnergy], maxEnergy, false, true);

    float* dev_scatterDist = copy1DdataToGPU(scatterDist, 2 * maxEnergy * 181, params->whichGPU);
    cudaTextureObject_t scatterDist_txt = NULL;
    cudaArray* scatterDist_array = loadTexture(scatterDist_txt, dev_scatterDist, make_int3(2, maxEnergy, 181), false, true);

    hypercube g_params;
    g_params.N = N_g;
    g_params.T = T_g;
    g_params.startVal = startVal_g;
    hypercube* dev_g_params = NULL;
    cudaMalloc((void**)&dev_g_params, sizeof(g_params));
    cudaMemcpy((void*)dev_g_params, (const void*)&(g_params), sizeof(hypercube), cudaMemcpyHostToDevice);

    hypercube f_params;
    f_params.N = N_f;
    f_params.T = T_f;
    f_params.startVal = startVal_f;
    hypercube* dev_f_params = NULL;
    cudaMalloc((void**)&dev_f_params, sizeof(f_params));
    cudaMemcpy((void*)dev_f_params, (const void*)&(f_params), sizeof(hypercube), cudaMemcpyHostToDevice);

    /*
    PhysicsTables xsec;
    xsec.source_txt = source_txt;
    xsec.energies_txt = energies_txt;
    xsec.sigma_PE_txt = sigma_PE_txt;
    xsec.sigma_CS_txt = sigma_CS_txt;
    xsec.sigma_RS_txt = sigma_RS_txt;
    xsec.scatterDist_txt = scatterDist_txt;
    xsec.N_energies = N_energies;
    xsec.maxEnergy = maxEnergy;
    //*/

    // upper bound on number of angles: numAngles = PI * numCols * angularRange/360
    // let's do: numAngles = numCols * angularRange / 360
    // numAngles / angularRange = numCols / 360
    // T_phi = 360 / numCols
    float T_phi_max = 360.0 / params->numCols * PI / 180.0;
    float lastAngle = -1.0e16;
    bool* projectionCalculated = new bool[params->numAngles];

    float source_and_detector[12];
    for (int i = 0; i < params->numAngles; i++)
    {
        float3 sourcePosition = make_float3(params->sourcePositions[3 * i + 0], params->sourcePositions[3 * i + 1], params->sourcePositions[3 * i + 2]);
        float3 moduleCenter = make_float3(params->moduleCenters[3 * i + 0], params->moduleCenters[3 * i + 1], params->moduleCenters[3 * i + 2]);
        float3 rowVector = make_float3(params->rowVectors[3 * i + 0], params->rowVectors[3 * i + 1], params->rowVectors[3 * i + 2]);
        float3 colVector = make_float3(params->colVectors[3 * i + 0], params->colVectors[3 * i + 1], params->colVectors[3 * i + 2]);

        for (int n = 0; n < 3; n++)
        {
            source_and_detector[0 + n] = params->sourcePositions[3 * i + n];
            source_and_detector[3 + n] = params->moduleCenters[3 * i + n];
            source_and_detector[6 + n] = params->rowVectors[3 * i + n];
            source_and_detector[9 + n] = params->colVectors[3 * i + n];
        }
        float* dev_source_and_detector = copy1DdataToGPU(source_and_detector, 12, params->whichGPU);
        if (dev_source_and_detector == 0)
            printf("failed to copy!\n");

        // First calculate the line integrals from the source to each voxel
        // These get reused many times so it is good to treat this line a look up table
        firstLeg(f_data_txt, params, dev_Df, sourcePosition);

        float* dev_proj = &dev_g[uint64(i) * uint64(params->numRows * params->numCols)];

        // Now calculate the full scatter model for a fixed source and detector module
        dim3 dimBlock(8, 8);
        dim3 dimGrid(int(ceil(double(params->numRows) / double(dimBlock.x))), int(ceil(double(params->numCols) / double(dimBlock.y))));

        bool doCalculation = true;
        if (params->phis != NULL)
        {
            if (fabs(params->phis[i] - lastAngle) < T_phi_max)
                doCalculation = false;
        }
        if (i == 0 || i == params->numAngles-1)
            doCalculation = true;
        if (doCalculation)
        {
            //printf("%d: calculating scatter model phi[%d] = %f\n", params->whichGPU, i, params->phis[i]*180.0/PI);
            /*
            firstOrderScatterModel <<< dimGrid, dimBlock >>> (dev_proj, dev_g_params, f_data_txt, dev_f_params, dev_Df, dev_source_and_detector,
                source_txt, energies_txt, detector_txt, sigma_PE_txt, sigma_CS_txt, sigma_RS_txt, scatterDist_txt);
            //*/
            //*
            firstOrderScatterModel_fast <<< dimGrid, dimBlock >>> (dev_proj, dev_g_params, f_data_txt, dev_f_params, dev_Df, dev_source_and_detector,
                source_txt, energies_txt, detector_txt, sigma_PE_txt, sigma_CS_txt, sigma_RS_txt, scatterDist_txt);
            //*/
            projectionCalculated[i] = true;
            if (params->phis != NULL)
                lastAngle = params->phis[i];
        }
        else
            projectionCalculated[i] = false;

        cudaFree(dev_source_and_detector);
    }

    // Interpolate views
    float* dev_phis = copyAngleArrayToGPU(params);
    bool* dev_projectionCalculated = copy1DbooleanToGPU(projectionCalculated, params->numAngles, params->whichGPU);
    dim3 dimBlock_interp = setBlockSize(N_g);
    dim3 dimGrid_interp = setGridSize(N_g, dimBlock_interp);
    interpolateViews <<< dimGrid_interp, dimBlock_interp >>> (dev_g , N_g, dev_phis, dev_projectionCalculated);
    cudaFree(dev_phis);
    cudaFree(dev_projectionCalculated);

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
    if (data_on_cpu)
        cudaFree(dev_f);
    
    cudaFreeArray(source_array);
    cudaDestroyTextureObject(source_txt);
    cudaFree(dev_source);

    cudaFreeArray(energies_array);
    cudaDestroyTextureObject(energies_txt);
    cudaFree(dev_energies);

    cudaFreeArray(detector_array);
    cudaDestroyTextureObject(detector_txt);
    cudaFree(dev_detector);

    cudaFreeArray(sigma_PE_array);
    cudaDestroyTextureObject(sigma_PE_txt);
    cudaFreeArray(sigma_CS_array);
    cudaDestroyTextureObject(sigma_CS_txt);
    cudaFreeArray(sigma_RS_array);
    cudaDestroyTextureObject(sigma_RS_txt);
    cudaFree(dev_sigma);

    cudaFreeArray(scatterDist_array);
    cudaDestroyTextureObject(scatterDist_txt);
    cudaFree(dev_scatterDist);

    cudaFree(dev_Df);
    if (data_on_cpu)
        cudaFree(dev_g);
    cudaFree(dev_g_params);

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
