
#include "projectors_extendedSF.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "projectors.h"
#include "cuda_utils.h"

__global__ void parallelBeamBackprojectorKernel_eSF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k;
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const int iv = int(floor(0.5f + (z - startVals_g.y) / T_g.y));
    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x;

        const float u_c = x_dot_theta_perp;
        const int iu_c = (u_c - startVals_g.z) / T_g.z;
        const float l_phi = T_f.x / max(fabs(cos_phi), fabs(sin_phi));

        if (fabs(sin_phi) >= fabs(cos_phi))
        {
            // x determines the width
            const float u_A = (x_dot_theta_perp - 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) / T_g.z;
            const float u_B = (x_dot_theta_perp + 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) / T_g.z;
            const int diu = max(1, int(ceil(T_f.x * fabs(sin_phi) / (0.5f * T_g.z))));
            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
                val += tex3D<float>(g, iu, iv, l) * l_phi * max(0.0, min(float(iu) + 0.5f, u_B) - max(float(iu) - 0.5f, u_A));
        }
        else
        {
            // y determines the width
            const float u_A = (x_dot_theta_perp - 0.5f * T_f.x * fabs(cos_phi) - startVals_g.z) / T_g.z;
            const float u_B = (x_dot_theta_perp + 0.5f * T_f.x * fabs(cos_phi) - startVals_g.z) / T_g.z;
            const int diu = max(1, int(ceil(T_f.x * fabs(cos_phi) / (0.5f * T_g.z))));
            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
                val += tex3D<float>(g, iu, iv, l) * l_phi * max(0.0, min(float(iu) + 0.5f, u_B) - max(float(iu) - 0.5f, u_A));
        }

    }
    f[ind] = val;
}

__global__ void fanBeamBackprojectorKernel_eSF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k;
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const int iv = int(floor(0.5f + (z - startVals_g.y)/T_g.y));

    const float x_lo = i - 0.5f;
    const float x_hi = i + 0.5f;

    const float y_lo = j - 0.5f;
    const float y_hi = j + 0.5f;

    const float T_x_inv = 1.0f / T_f.x;

    const float vox_half = 0.5f * T_f.x;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float u_c = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabs(u_c * cos_phi - sin_phi);
        const float y_denom = fabs(u_c * sin_phi + cos_phi);
        const float l_phi = T_f.x * sqrt(1.0f + u_c * u_c) / max(x_denom, y_denom);

        const int iu_c = (u_c - startVals_g.z) / T_g.z;

        if (x_denom > y_denom)
        {
            const float u_A = ((x_dot_theta_perp + sin_phi * vox_half) / (R_minus_x_dot_theta + vox_half * cos_phi) - startVals_g.z) / T_g.z;
            const float u_B = ((x_dot_theta_perp - sin_phi * vox_half) / (R_minus_x_dot_theta - vox_half * cos_phi) - startVals_g.z) / T_g.z;

            const int diu = max(1, int(ceil(T_f.x * R_minus_x_dot_theta_inv * fabs(sin_phi) / (0.5f * T_g.z))));

            // use x_lo, x_hi
            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
            {
                const float u = iu * T_g.z + startVals_g.z;
                const float uWeight = l_phi * max(0.0, min(float(iu) + 0.5f, max(u_A, u_B)) - max(float(iu) - 0.5f, min(u_A, u_B)));
                val += tex3D<float>(g, iu, iv, l) * uWeight;
            }
        }
        else
        {
            // use y_lo, y_hi
            const float u_A = ((x_dot_theta_perp - cos_phi * vox_half) / (R_minus_x_dot_theta + vox_half * sin_phi) - startVals_g.z) / T_g.z;
            const float u_B = ((x_dot_theta_perp + cos_phi * vox_half) / (R_minus_x_dot_theta - vox_half * sin_phi) - startVals_g.z) / T_g.z;

            const int diu = max(1, int(ceil(T_f.x * R_minus_x_dot_theta_inv * fabs(cos_phi) / (0.5f * T_g.z))));

            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
            {
                const float u = iu * T_g.z + startVals_g.z;
                const float uWeight = l_phi * max(0.0, min(float(iu) + 0.5f, max(u_A, u_B)) - max(float(iu) - 0.5f, min(u_A, u_B)));
                val += tex3D<float>(g, iu, iv, l) * uWeight;

            }
        }
    }
    f[ind] = val;
}

__global__ void coneBeamBackprojectorKernel_eSF(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;
    if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
        return;

    int ind;
    if (volumeDimensionOrder == 0)
        ind = i * N_f.y * N_f.z + j * N_f.z + k;
    else
        ind = k * N_f.y * N_f.x + j * N_f.x + i;

    const float x = i * T_f.x + startVals_f.x;
    const float y = j * T_f.y + startVals_f.y;
    const float z = k * T_f.z + startVals_f.z;
    if (x * x + y * y > rFOVsq)
    {
        f[ind] = 0.0f;
        return;
    }

    const float x_lo = i - 0.5f;
    const float x_hi = i + 0.5f;

    const float y_lo = j - 0.5f;
    const float y_hi = j + 0.5f;

    const float T_x_inv = 1.0f / T_f.x;
    const float T_z_inv = 1.0f / T_f.z;

    const float vox_half = 0.5f * T_f.x;
    const float voxz_half = 0.5f * T_f.z;

    float val = 0.0;
    for (int l = 0; l < N_g.x; l++)
    {
        const float sin_phi = sin(phis[l]);
        const float cos_phi = cos(phis[l]);

        const float x_dot_theta_perp = cos_phi * y - sin_phi * x + tau;
        const float R_minus_x_dot_theta = R - x * cos_phi - y * sin_phi;
        const float R_minus_x_dot_theta_inv = 1.0f / R_minus_x_dot_theta;

        const float u_c = x_dot_theta_perp * R_minus_x_dot_theta_inv;
        const float x_denom = fabs(u_c * cos_phi - sin_phi);
        const float y_denom = fabs(u_c * sin_phi + cos_phi);
        const float v_c = z * R_minus_x_dot_theta_inv;
        const float l_phi = T_f.x * sqrt(1.0f + u_c * u_c) / max(x_denom, y_denom);// *sqrt(1.0f + v_c * v_c);

        const int iv_c = (v_c - startVals_g.y) / T_g.y;
        const int iu_c = (u_c - startVals_g.z) / T_g.z;

        const float v_A = ((z - voxz_half) * R_minus_x_dot_theta_inv - startVals_g.y) / T_g.y;
        const float v_B = ((z + voxz_half) * R_minus_x_dot_theta_inv - startVals_g.y) / T_g.y;

        //const int div = max(1, int(ceil(0.5f*T_g.y * R_minus_x_dot_theta * T_z_inv))); // FIXME
        const int div = max(1, int(ceil(R_minus_x_dot_theta_inv * T_f.z / (0.5f * T_g.y)))); // FIXME

        if (x_denom > y_denom)
        {
            //const float z_A = ((v - 0.5f * T_g.y) * rayParam_inv - startVals_f.z) * T_z_inv;
            //const float z_B = ((v + 0.5f * T_g.y) * rayParam_inv - startVals_f.z) * T_z_inv;
            const float u_A = ((x_dot_theta_perp + sin_phi * vox_half) / (R_minus_x_dot_theta + vox_half * cos_phi) - startVals_g.z) / T_g.z;
            const float u_B = ((x_dot_theta_perp - sin_phi * vox_half) / (R_minus_x_dot_theta - vox_half * cos_phi) - startVals_g.z) / T_g.z;

            //const int diu = max(1, int(ceil(0.5f * T_g.z / (T_f.x * R_minus_x_dot_theta_inv * fabs(sin_phi)))));
            const int diu = max(1, int(ceil(T_f.x * R_minus_x_dot_theta_inv * fabs(sin_phi) / (0.5f * T_g.z))));

            // use x_lo, x_hi
            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
            {
                const float u = iu * T_g.z + startVals_g.z;

                // calculate x index for u-0.5*T_g.z and u+0.5*T_g.z
                // u-0.5f*T_g.z = x_dot_theta_perp * R_minus_x_dot_theta_inv

                const float uWeight = l_phi * max(0.0, min(float(iu)+0.5f, max(u_A, u_B)) - max(float(iu)-0.5f, min(u_A, u_B)));
                if (uWeight == 0.0f)
                    continue;
                for (int iv = iv_c - div; iv <= iv_c + div; iv++)
                {
                    const float v = iv * T_g.y + startVals_g.y;

                    // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                    const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));
                
                    val += tex3D<float>(g, iu, iv, l) * vWeight * uWeight * sqrt(1.0f + v * v);
                }
            }
        }
        else
        {
            // use y_lo, y_hi
            const float u_A = ((x_dot_theta_perp - cos_phi * vox_half) / (R_minus_x_dot_theta + vox_half * sin_phi) - startVals_g.z) / T_g.z;
            const float u_B = ((x_dot_theta_perp + cos_phi * vox_half) / (R_minus_x_dot_theta - vox_half * sin_phi) - startVals_g.z) / T_g.z;

            //const int diu = max(1, int(ceil(0.5f * T_g.z / (T_f.x * R_minus_x_dot_theta_inv * fabs(cos_phi)))));
            const int diu = max(1, int(ceil(T_f.x * R_minus_x_dot_theta_inv * fabs(cos_phi) / (0.5f * T_g.z))));

            for (int iu = iu_c - diu; iu <= iu_c + diu; iu++)
            {
                const float u = iu * T_g.z + startVals_g.z;

                // calculate x index for u-0.5*T_g.z and u+0.5*T_g.z
                // u-0.5f*T_g.z = x_dot_theta_perp * R_minus_x_dot_theta_inv

                const float uWeight = l_phi * max(0.0, min(float(iu) + 0.5f, max(u_A, u_B)) - max(float(iu) - 0.5f, min(u_A, u_B)));
                if (uWeight == 0.0f)
                    continue;
                for (int iv = iv_c - div; iv <= iv_c + div; iv++)
                {
                    const float v = iv * T_g.y + startVals_g.y;

                    // calculate z index for v-0.5*T_g.y and v+0.5*T_g.y
                    const float vWeight = max(0.0, min(float(iv) + 0.5f, max(v_A, v_B)) - max(float(iv) - 0.5f, min(v_A, v_B)));

                    val += tex3D<float>(g, iu, iv, l) * vWeight * uWeight * sqrt(1.0f + v * v);
                }
            }
        }
    }
    f[ind] = val;
}

__global__ void parallelBeamProjectorKernel_eSF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const int iz = int(floor(0.5f + (v - startVals_f.z) / T_f.z));

    const float u_lo = n - 0.5f;
    const float u_hi = n + 0.5f;

    const float v_lo = m - 0.5f;
    const float v_hi = m + 0.5f;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;

    const float vox_half = 0.5f * T_f.x;

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    const float l_phi = 1.0f / max(fabs(cos_phi), fabs(sin_phi));

    float g_output = 0.0f;

    // x = s*theta_perp - l*theta
    if (fabs(sin_phi) > fabs(cos_phi))
    {
        // primary direction is y
        for (int iy = 0; iy < N_f.y; iy++)
        {
            const float y = (float)iy * T_f.y + startVals_f.y;

            // u = cos_phi * y - sin_phi * x
            const float x_c = (cos_phi * y - u) / sin_phi;
            const int dix = max(1, int(ceil(0.5f * T_g.z / (T_f.x * fabs(sin_phi)))));
            const int ix_c = int(0.5f + (x_c - startVals_f.x) / T_f.x);

            for (int ix = ix_c - dix; ix <= ix_c + dix; ix++)
            {
                // calculate u index for x-0.5*T_f.x and x+0.5*T_f.x
                const float x = ix * T_f.x + startVals_f.x;

                const float x_A = (cos_phi * y - sin_phi * x - 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) * T_u_inv;
                const float x_B = (cos_phi * y - sin_phi * x + 0.5f * T_f.x * fabs(sin_phi) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(x_B, u_hi) - max(x_A, u_lo));

                if (volumeDimensionOrder == 0)
                    g_output += tex3D<float>(f, iz, iy, ix) * uFootprint;
                else
                    g_output += tex3D<float>(f, ix, iy, iz) * uFootprint;
            }
        }
    }
    else
    {
        // primary direction is x
        for (int ix = 0; ix < N_f.x; ix++)
        {
            const float x = (float)ix * T_f.x + startVals_f.x;

            // u = cos_phi * y - sin_phi * x
            const float y_c = (u + sin_phi * x) / cos_phi;
            const int diy = max(1, int(ceil(0.5f * T_g.z / (T_f.y * fabs(cos_phi)))));
            const int iy_c = int(0.5f + (y_c - startVals_f.y) / T_f.y);

            for (int iy = iy_c - diy; iy <= iy_c + diy; iy++)
            {
                // calculate u index for y-0.5*T_f.y and y+0.5*T_f.y
                const float y = iy * T_f.y + startVals_f.y;

                const float y_A = (cos_phi * y - sin_phi * x - 0.5f * T_f.y * fabs(cos_phi) - startVals_g.z) * T_u_inv;
                const float y_B = (cos_phi * y - sin_phi * x + 0.5f * T_f.y * fabs(cos_phi) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(y_B, u_hi) - max(y_A, u_lo));

                if (volumeDimensionOrder == 0)
                    g_output += tex3D<float>(f, iz, iy, ix) * uFootprint;
                else
                    g_output += tex3D<float>(f, ix, iy, iz) * uFootprint;
            }
        }
    }
    g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * l_phi * g_output;
}

__global__ void fanBeamProjectorKernel_eSF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const int iz = int(floor(0.5f + (v-startVals_f.z)/T_f.z));

    const float u_lo = n - 0.5f;
    const float u_hi = n + 0.5f;

    const float v_lo = m - 0.5f;
    const float v_hi = m + 0.5f;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;

    const float vox_half = 0.5f * T_f.x;

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    float g_output = 0.0f;

    // x = R*theta - tau*theta_perp + l*[-theta + u*theta_perp + v*z]
    // -sin_phi*u - cos_phi; u*cos_phi - sin_phi
    if (fabs(u * cos_phi - sin_phi) > fabs(u * sin_phi + cos_phi))
    {
        const float rayParam_slope = 1.0 / (-sin_phi + u * cos_phi);
        const float rayParam_offset = (-R * sin_phi + tau * cos_phi) * rayParam_slope;

        const float x_shift = R * cos_phi + tau * sin_phi;
        const float x_slope = -cos_phi - u * sin_phi;

        const float cos_over_sin = cos_phi / sin_phi;

        // primary direction is y
        for (int iy = 0; iy < N_f.y; iy++)
        {
            const float y = (float)iy * T_f.y + startVals_f.y;

            const float rayParam = y * rayParam_slope + rayParam_offset;
            const float x_c = x_shift + x_slope * rayParam;

            const float rayParam_inv = 1.0f / rayParam;
            const float rayParam_sin_inv = 1.0f / (rayParam * sin_phi);

            const int dix = max(1, int(ceil(0.5f * T_g.z / (T_f.x * fabs(rayParam_sin_inv)))));

            const int ix_c = int(0.5f + (x_c - startVals_f.x) / T_f.x);

            for (int ix = ix_c - dix; ix <= ix_c + dix; ix++)
            {
                // calculate u index for x-0.5*T_f.x and x+0.5*T_f.x
                const float x = ix * T_f.x + startVals_f.x;

                const float x_A = (((x_shift - x + vox_half) * rayParam_sin_inv - cos_over_sin) - startVals_g.z) * T_u_inv;
                const float x_B = (((x_shift - x - vox_half) * rayParam_sin_inv - cos_over_sin) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(max(x_A, x_B), u_hi) - max(min(x_A, x_B), u_lo));

                if (volumeDimensionOrder == 0)
                    g_output += tex3D<float>(f, iz, iy, ix) * uFootprint;
                else
                    g_output += tex3D<float>(f, ix, iy, iz) * uFootprint;
            }
        }
        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * g_output;
    }
    else
    {
        const float rayParam_slope = 1.0 / (-cos_phi - u * sin_phi);
        const float rayParam_offset = (-R * cos_phi - tau * sin_phi) * rayParam_slope;

        const float y_shift = R * sin_phi - tau * cos_phi;
        const float y_slope = -sin_phi + u * cos_phi;

        const float sin_over_cos = sin_phi / cos_phi;

        // primary direction is x
        for (int ix = 0; ix < N_f.x; ix++)
        {
            const float x = (float)ix * T_f.x + startVals_f.x;

            const float rayParam = x * rayParam_slope + rayParam_offset;
            const float y_c = y_shift + y_slope * rayParam;

            const float rayParam_inv = 1.0f / rayParam;
            const float rayParam_cos_inv = 1.0f / (rayParam * cos_phi);

            const int diy = max(1, int(ceil(0.5f * T_g.z / (T_f.y * fabs(rayParam_cos_inv)))));

            const int iy_c = int(0.5f + (y_c - startVals_f.y) / T_f.y);

            for (int iy = iy_c - diy; iy <= iy_c + diy; iy++)
            {
                // calculate u index for y-0.5*T_f.y and y+0.5*T_f.y
                const float y = iy * T_f.y + startVals_f.y;

                const float y_A = (((y - vox_half - y_shift) * rayParam_cos_inv + sin_over_cos) - startVals_g.z) * T_u_inv;
                const float y_B = (((y + vox_half - y_shift) * rayParam_cos_inv + sin_over_cos) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(max(y_A, y_B), u_hi) - max(min(y_A, y_B), u_lo));
                
                if (volumeDimensionOrder == 0)
                    g_output += tex3D<float>(f, iz, iy, ix) * uFootprint;
                else
                    g_output += tex3D<float>(f, ix, iy, iz) * uFootprint;
            }
        }

        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * g_output;
    }
}

__global__ void coneBeamProjectorKernel_eSF(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float tau, float rFOVsq, float* phis, int volumeDimensionOrder)
{
    const int l = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = threadIdx.y + blockIdx.y * blockDim.y;
    const int n = threadIdx.z + blockIdx.z * blockDim.z;
    if (l >= N_g.x || m >= N_g.y || n >= N_g.z)
        return;

    const float v = m * T_g.y + startVals_g.y;
    const float u = n * T_g.z + startVals_g.z;

    const float u_lo = n - 0.5f;
    const float u_hi = n + 0.5f;

    const float v_lo = m - 0.5f;
    const float v_hi = m + 0.5f;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;

    const float vox_half = 0.5f * T_f.x;

    const float sin_phi = sin(phis[l]);
    const float cos_phi = cos(phis[l]);

    float g_output = 0.0f;

    // x = R*theta - tau*theta_perp + l*[-theta + u*theta_perp + v*z]
    // -sin_phi*u - cos_phi; u*cos_phi - sin_phi
    if (fabs(u * cos_phi - sin_phi) > fabs(u * sin_phi + cos_phi))
    {
        const float rayParam_slope = 1.0 / (-sin_phi + u * cos_phi);
        const float rayParam_offset = (-R * sin_phi + tau * cos_phi) * rayParam_slope;

        const float x_shift = R * cos_phi + tau * sin_phi;
        const float x_slope = -cos_phi - u * sin_phi;

        const float cos_over_sin = cos_phi / sin_phi;

        // primary direction is y
        for (int iy = 0; iy < N_f.y; iy++)
        {
            const float y = (float)iy * T_f.y + startVals_f.y;

            const float rayParam = y * rayParam_slope + rayParam_offset;
            const float x_c = x_shift + x_slope * rayParam;
            const float z_c = v * rayParam;

            const float rayParam_inv = 1.0f / rayParam;
            const float rayParam_sin_inv = 1.0f / (rayParam * sin_phi);

            //const int dix = max(1,int(ceil(T_f.x * rayParam_sin_inv * T_u_inv)));
            //const int diz = max(1, int(ceil(T_f.z * rayParam_inv * T_v_inv)));
            const int dix = max(1, int(ceil(0.5f * T_g.z / (T_f.x * fabs(rayParam_sin_inv)))));
            const int diz = max(1, int(ceil(0.5f * T_g.y / (T_f.z * fabs(rayParam_inv)))));

            const int iz_c = int(0.5f + (z_c - startVals_f.z) / T_f.z);
            const int ix_c = int(0.5f + (x_c - startVals_f.x) / T_f.x);

            for (int ix = ix_c - dix; ix <= ix_c + dix; ix++)
            {
                // calculate u index for x-0.5*T_f.x and x+0.5*T_f.x
                const float x = ix * T_f.x + startVals_f.x;

                const float x_A = (((x_shift - x + vox_half) * rayParam_sin_inv - cos_over_sin) - startVals_g.z) * T_u_inv;
                const float x_B = (((x_shift - x - vox_half) * rayParam_sin_inv - cos_over_sin) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(max(x_A, x_B), u_hi) - max(min(x_A, x_B), u_lo));
                if (uFootprint == 0.0f)
                    continue;
                for (int iz = iz_c - diz; iz <= iz_c + diz; iz++)
                {
                    // calculate v index for z-0.5*T_f.z and z+0.5*T_f.z
                    const float z = iz * T_f.z + startVals_f.z;
                    const float z_A = ((z - 0.5f * T_f.z) * rayParam_inv - startVals_g.y) * T_v_inv;
                    const float z_B = ((z + 0.5f * T_f.z) * rayParam_inv - startVals_g.y) * T_v_inv;

                    const float vFootprint = max(0.0f, min(max(z_A, z_B), v_hi) - max(min(z_A, z_B), v_lo));

                    if (volumeDimensionOrder == 0)
                        g_output += tex3D<float>(f, iz, iy, ix) * vFootprint * uFootprint;
                    else
                        g_output += tex3D<float>(f, ix, iy, iz) * vFootprint * uFootprint;
                }
            }
        }
        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * cos_phi - sin_phi) * sqrt(1.0f + v * v) * g_output;
    }
    else
    {
        const float rayParam_slope = 1.0 / (-cos_phi - u * sin_phi);
        const float rayParam_offset = (-R * cos_phi - tau * sin_phi) * rayParam_slope;

        const float y_shift = R * sin_phi - tau * cos_phi;
        const float y_slope = -sin_phi + u * cos_phi;

        const float sin_over_cos = sin_phi / cos_phi;

        // primary direction is x
        for (int ix = 0; ix < N_f.x; ix++)
        {
            const float x = (float)ix * T_f.x + startVals_f.x;

            const float rayParam = x * rayParam_slope + rayParam_offset;
            const float y_c = y_shift + y_slope * rayParam;
            const float z_c = v * rayParam;

            const float rayParam_inv = 1.0f / rayParam;
            const float rayParam_cos_inv = 1.0f / (rayParam * cos_phi);

            const int diy = max(1, int(ceil(0.5f * T_g.z / (T_f.y * fabs(rayParam_cos_inv)))));
            const int diz = max(1, int(ceil(0.5f * T_g.y / (T_f.z * fabs(rayParam_inv)))));

            const int iz_c = int(0.5f + (z_c - startVals_f.z) / T_f.z);
            const int iy_c = int(0.5f + (y_c - startVals_f.y) / T_f.y);

            for (int iy = iy_c - diy; iy <= iy_c + diy; iy++)
            {
                // calculate u index for y-0.5*T_f.y and y+0.5*T_f.y
                const float y = iy * T_f.y + startVals_f.y;

                const float y_A = (((y - vox_half - y_shift) * rayParam_cos_inv + sin_over_cos) - startVals_g.z) * T_u_inv;
                const float y_B = (((y + vox_half - y_shift) * rayParam_cos_inv + sin_over_cos) - startVals_g.z) * T_u_inv;
                const float uFootprint = max(0.0f, min(max(y_A, y_B), u_hi) - max(min(y_A, y_B), u_lo));
                if (uFootprint == 0.0f)
                    continue;
                for (int iz = iz_c - diz; iz <= iz_c + diz; iz++)
                {
                    // calculate v index for z-0.5*T_f.z and z+0.5*T_f.z
                    const float z = iz * T_f.z + startVals_f.z;
                    const float z_A = ((z - 0.5f * T_f.z) * rayParam_inv - startVals_g.y) * T_v_inv;
                    const float z_B = ((z + 0.5f * T_f.z) * rayParam_inv - startVals_g.y) * T_v_inv;

                    const float vFootprint = max(0.0f, min(max(z_A, z_B), v_hi) - max(min(z_A, z_B), v_lo));

                    if (volumeDimensionOrder == 0)
                        g_output += tex3D<float>(f, iz, iy, ix) * vFootprint * uFootprint;
                    else
                        g_output += tex3D<float>(f, ix, iy, iz) * vFootprint * uFootprint;
                }
            }
        }

        g[l * N_g.z * N_g.y + m * N_g.z + n] = T_f.x * sqrt(1.0f + u * u) / fabs(u * sin_phi + cos_phi) * sqrt(1.0f + v * v) * g_output;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Routines
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool project_eSF_parallel(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_eSF(g, f, params, cpu_to_gpu);
}

bool backproject_eSF_parallel(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_eSF(g, f, params, cpu_to_gpu);
}

bool project_eSF_fan(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_eSF(g, f, params, cpu_to_gpu);
}

bool backproject_eSF_fan(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_eSF(g, f, params, cpu_to_gpu);
}

bool project_eSF_cone(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    return project_eSF(g, f, params, cpu_to_gpu);
}

bool backproject_eSF_cone(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    return backproject_eSF(g, f, params, cpu_to_gpu);
}

bool project_eSF(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    // Allocate planogram data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float rFOVsq = params->rFOV() * params->rFOV();

    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (cpu_to_gpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;

    cudaTextureObject_t d_data_txt = NULL;
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, false, bool(params->volumeDimensionOrder == 1));

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_g);
    dim3 dimGrid = setGridSize(N_g, dimBlock);
    if (params->geometry == parameters::CONE)
        coneBeamProjectorKernel_eSF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    else if (params->geometry == parameters::FAN)
        fanBeamProjectorKernel_eSF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    else if (params->geometry == parameters::PARALLEL)
        parallelBeamProjectorKernel_eSF <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    if (cpu_to_gpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_phis);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}

bool backproject_eSF(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
    if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
        return false;

    cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    float* dev_g = 0;
    float* dev_f = 0;

    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    if (cpu_to_gpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc(volume) failed!\n");
        }
    }
    else
        dev_f = f;

    float* dev_phis = copyAngleArrayToGPU(params);

    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, true);

    float rFOVsq = params->rFOV() * params->rFOV();

    if (cpu_to_gpu)
        dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
    else
        dev_g = g;

    cudaTextureObject_t d_data_txt = NULL;
    //cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, false);
    cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, params->doExtrapolation, false);

    // Call Kernel
    dim3 dimBlock = setBlockSize(N_f);
    dim3 dimGrid = setGridSize(N_f, dimBlock);
    if (params->geometry == parameters::CONE)
        coneBeamBackprojectorKernel_eSF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    else if (params->geometry == parameters::FAN)
        fanBeamBackprojectorKernel_eSF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->tau, rFOVsq, dev_phis, params->volumeDimensionOrder);
    else if (params->geometry == parameters::PARALLEL)
        parallelBeamBackprojectorKernel_eSF <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (cpu_to_gpu)
        pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
    else
        f = dev_f;

    // Clean up
    cudaFreeArray(d_data_array);
    cudaDestroyTextureObject(d_data_txt);
    cudaFree(dev_phis);

    if (cpu_to_gpu)
    {
        if (dev_g != 0)
            cudaFree(dev_g);
        if (dev_f != 0)
            cudaFree(dev_f);
    }

    return true;
}
