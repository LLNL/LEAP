////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// CPU-based resampling of 3D arrays
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include "leap_defines.h"
#include "resample_cpu.h"
#include "cpu_utils.h"
#include "parameters.h"
#include "maximally_flat_filter.h"
#include "finite_difference_filters.h"
#include "geometry/rebin.h"

using namespace std;

/*
// will need this in the cmake file: add_compile_options(-mavx -mavx2 -mfma -msse3)
#include <immintrin.h>
float dot_product8(const float* a, const float* b) {
    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);
    __m256 vmul = _mm256_mul_ps(va, vb);

    // Horizontal add (sum all elements)
    __m128 low = _mm256_castps256_ps128(vmul);
    __m128 high = _mm256_extractf128_ps(vmul, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    return _mm_cvtss_f32(sum128);
}
//*/

inline float smooth_abs6_scaled(float x, float delta)
{
    constexpr float epsilon = 1.0 / 65535.0;
    if (x <= 0.0 || delta <= 0.0) return x;
    /*
    float u  = x / delta;
    float u2 = u * u;
    float u6 = u2 * u2 * u2;
    return delta * (std::sqrt(std::cbrt(1.0 + u6)) - 1.0) + epsilon;
    //*/
    float x2 = x*x;
    float x6 = x2 * x2 * x2;
    float delta2 = delta*delta;
    float delta6 = delta2 * delta2 * delta2;
    //return std::cbrt(std::cbrt(x6 + delta6)) - delta + epsilon;
    return std::sqrt(std::cbrt(x6 + delta6)) - delta + epsilon;
    //return pow(x6 + delta6, 1.0/6.0) - delta + epsilon;
}

inline double smooth_abs6_scaled_inv(double y, float delta)
{
    constexpr double epsilon = 1.0 / 65535.0;

    if (y <= epsilon || delta <= 0.0) return y;

    float u = y - epsilon + delta;
    float u2 = u * u;
    float u6 = u2 * u2 * u2;
    float delta2 = delta*delta;
    float delta6 = delta2 * delta2 * delta2;

    return std::sqrt(std::cbrt(u6 - delta6));

    /*
    // Recover u
    double u = 1.0 + (y - epsilon) / delta;

    // Compute u^6 via multiplies
    double u2 = u * u;
    double u6 = u2 * u2 * u2;

    double t = u6 - 1.0;

    // Sixth root via double cbrt
    double r = std::sqrt(std::cbrt(t));

    return std::copysign(delta * r, y);
    //*/
}

float* bumpFcn(float W, float delay, int& L, int order)
{
    if (order <= -1)
    {
        L = 1;
        float* h = new float[2 * L + 1];
        h[0] = std::max(0.0, 1.0 - abs(-1.0-delay));
        h[1] = std::max(0.0, 1.0 - abs(0.0-delay));
        h[2] = std::max(0.0, 1.0 - abs(1.0-delay));
        return h;
    }
    else if (order < 2)
    {
        L = max(1, int(floor(W)));
        float* h = new float[2 * L + 1];
        float accum = 0.0;
        for (int l = -L; l <= L; l++)
        {
            if (fabs(float(l) - delay) > W)
                h[l + L] = 0.0;
            else
            {
                h[l + L] = cos(0.5 * PI * (float(l) - delay) / W);
                h[l + L] *= h[l + L];
                accum += h[l + L];
            }
        }
        for (int l = 0; l <= 2 * L; l++)
            h[l] = h[l] / accum;
        //N_taps = 2 * L + 1;
        return h;
    }
    else
    {
        int N = max(3, int(order*W+1));
        //int N = std::max(int(order*W+1), int(ceil(9.0*W+1)));
        N = N + (N-1)%2; // make it odd
        //2*L+1 = N
        L = (N-1)/2;
        float* h = new float[2 * L + 1];
        float accum = 0.0;
        for (int i = 0; i < N; i++)
        {
            float x = float(i - L - delay)/W;
            float y = mff_filter(x, order);
            if (fabs(y) < 1.0e-8)
                y = 0.0;
            h[i] = y;
            accum += y;
        }
        for (int i = 0; i < N; i++)
            h[i] = h[i] / accum;
        return h;
    }
}

/*
void axpy_auto(float* __restrict y,
               const float* __restrict x,
               float h, std::size_t N)
{
    #pragma GCC ivdep   // or: #pragma clang loop vectorize(enable)
    for (std::size_t i = 0; i < N; ++i)
        y[i] += h * x[i];
}
//*/

bool resampleProjectionAngles_cpu(float* g, parameters* params, float* g_new, float* phis_new, int N_phis_new, float filter_width, int row_min, int row_max)
{
    if (g == nullptr || params == nullptr || g_new == nullptr || phis_new == nullptr || N_phis_new <= 0 || params->numAngles <= 1)
    {
        printf("invalid inputs\n");
        return false;
    }

    if (row_min < 0)
        row_min = 0;
    if (row_max < 0 || row_max > params->numRows-1)
        row_max = params->numRows-1;
    if (row_min > row_max)
        return false;
    int numRows_out = row_max - row_min + 1;
    int ind_offset = row_min * params->numCols;

    float T_phi = fabs(params->T_phi()) * 180.0 / PI;
    if (filter_width <= 0.0)
    {
        if (N_phis_new > 1)
            filter_width = max(float(1.0), fabs(phis_new[1]-phis_new[0]) / T_phi);
        else
            filter_width = 1.0;
    }
    int L = int(ceil(filter_width));
    //float* phis_rad = params->phis;
    float* phis = new float[params->numAngles];
    params->get_angles(phis, true);
    float phi_new_lo = min(phis_new[0], phis_new[N_phis_new-1]);
    float phi_new_hi = max(phis_new[0], phis_new[N_phis_new-1]);
    float phi_lo = min(phis[0], phis[params->numAngles-1]);
    float phi_hi = max(phis[0], phis[params->numAngles-1]);
    if (phi_new_lo <= phi_lo - T_phi || phi_new_hi >= phi_hi + T_phi)
    {
        printf("New projection angles must be within %f and %f\n", phi_lo, phi_hi);
        delete [] phis;
        return false;
    }

    //printf("original: %f to %f and T_phi = %f\n", phi_lo, phi_hi, T_phi);
    //printf("new: %f to %f and T_phi = %f\n", phi_new_lo, phi_new_hi, phis_new[1]-phis_new[0]);
    //printf("filter_width = %f, L = %d\n", filter_width, L);

    uint64 proj_sz = uint64(params->numRows) * uint64(params->numCols);
    uint64 proj_out_sz = uint64(numRows_out) * uint64(params->numCols);

    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_phis_new; i++)
    {
        float* newProj = &g_new[uint64(i)*proj_out_sz];
        memset(newProj, 0, sizeof(float)*proj_out_sz);
        //(phis[i] + 0.5 * PI) * 180.0 / PI;
        int phi_ind_mid = int(floor(0.5 + params->phi_inv(phis_new[i]*PI/180.0 - 0.5*PI)));
        float* h = new float[2*L+1];
        float sum_h = 0.0;
        for (int n = -L; n <= L; n++)
        {
            float arg = (phis_new[i] - phis[max(0, min(params->numAngles-1, phi_ind_mid+n))]) / (filter_width * T_phi);
            if (fabs(arg) < 1.0)
            {
                float h_cur = cos(0.5 * PI * arg);
                h_cur *= h_cur;
                h[n+L] = h_cur;
                sum_h += h_cur;
            }
            else
                h[n+L] = 0.0;
        }
        if (sum_h > 0.0)
        {
            for (int n = -L; n <= L; n++)
                h[n+L] = h[n+L] / sum_h;
        }
        else
            h[L] = 1.0;

        for (int n = -L; n <= L; n++)
        {
            float* aProj = &g[uint64(max(0, min(params->numAngles-1, phi_ind_mid+n))) * proj_sz];
            for (int j = 0; j < numRows_out*params->numCols; j++)
                newProj[j] += h[n+L]*aProj[j + ind_offset];
        }

        delete [] h;
    }
    //printf("done\n");
    params->set_angles(phis_new, N_phis_new, true);
    delete [] phis;
    params->numRows = numRows_out;
    params->centerRow = params->centerRow - row_min;
    return true;
}

bool downSample_cpu(float* I, int* N, float* I_dn, int* N_dn, float* factors, int order, float maxWidth, float* offset)
{
    if (I == NULL || N == NULL || I_dn == NULL || N_dn == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;

    if (order == 2)
        order = 0;

    //printf("down-sample: (%d, %d, %d) ==> (%d, %d, %d)\n", N[0], N[1], N[2], N_dn[0], N_dn[1], N_dn[2]);

    float u_c = 0.5 * float(N[0] - 1);
    float v_c = 0.5 * float(N[1] - 1);
    float w_c = 0.5 * float(N[2] - 1);

    float zero_offset[3] = {0.0, 0.0, 0.0};
    if (offset == NULL)
        offset = &zero_offset[0];

    float filterWidths[3] = {factors[0], factors[1], factors[2]};
    if (maxWidth > 1.0)
    {
        filterWidths[0] = min(filterWidths[0], maxWidth);
        filterWidths[1] = min(filterWidths[1], maxWidth);
        filterWidths[2] = min(filterWidths[2], maxWidth);
    }


    // Pre-compute the filters for the last dimension
    int* h_2_lengths = (int*) malloc(sizeof(int)*N_dn[2]);
    float** h_2_filters = (float**) malloc(sizeof(float*)*N_dn[2]);
    for (int k = 0; k < N_dn[2]; k++)
    {
        float z = (k - 0.5 * float(N_dn[2] - 1)) * factors[2] + offset[2];
        int kk = int(floor(0.5 + z + w_c));
        float delay_2 = z + w_c - float(kk);

        int L_2;
        float* h_2 = bumpFcn(filterWidths[2], delay_2, L_2, order);
        
        size_t size_bytes = size_t(2*L_2+1) * sizeof(float);
        float* h_2_aligned = malloc_aligned(size_bytes);

        memcpy(h_2_aligned, h_2, size_bytes);
        delete [] h_2;

        h_2_filters[k] = h_2_aligned;
        h_2_lengths[k] = L_2;
        //printf("L_2[%d] = %d\n", k, L_2);
        //printf("h = %f, %f, %f\n", h_2_filters[k][0], h_2_filters[k][1], h_2_filters[k][2]);
    }

    // Compute some maximum extents
    int maxFilterLength = max(int(factors[1]), max(int(factors[2]), h_2_lengths[0])) + 2;
    
    float y_min = (-0.5 * float(N_dn[1] - 1)) * factors[1] + offset[1];
    int y_ind_min = max(0, int(floor(0.5 + y_min + v_c)) - maxFilterLength);
    float y_max = (0.5 * float(N_dn[1] - 1)) * factors[1] + offset[1];
    int y_ind_max = min(N[1]-1, int(floor(0.5 + y_max + v_c)) + maxFilterLength);

    float z_min = (-0.5 * float(N_dn[2] - 1)) * factors[2] + offset[2];
    int z_ind_min = max(0, int(floor(0.5 + z_min + w_c)) - maxFilterLength);
    float z_max = (0.5 * float(N_dn[2] - 1)) * factors[2] + offset[2];
    int z_ind_max = min(N[2]-1, int(floor(0.5 + z_max + w_c)) + maxFilterLength);

    //y_ind_min = 0;
    //z_ind_min = 0;
    //y_ind_max = N[1]-1;
    //z_ind_max = N[2]-1;

    //printf("dim 2 bounds: %d to %d (%d)\n", y_ind_min, y_ind_max, N[1]);
    //printf("dim 3 bounds: %d to %d (%d)\n", z_ind_min, z_ind_max, N[2]);

    uint64 img_sz = uint64(N[2] * N[1]);

    int num_threads = num_cpu_threads();

    float** source_2Ds = NULL;
    size_t size_bytes = size_t(N[1]) * size_t(N[2]) * sizeof(float);
    if (factors[0] != 1.0)
    {
        source_2Ds = (float**) malloc(sizeof(float*)*num_threads);
        omp_set_num_threads(num_threads);
        #pragma omp parallel for
        for (int i = 0; i < num_threads; i++)
        {
            int ind = omp_get_thread_num();
            source_2Ds[ind] = calloc_aligned(size_bytes);
        }
    }

    float** source_1Ds = NULL;
    if (factors[1] != 1.0)
    {
        source_1Ds = (float**) malloc(sizeof(float*)*num_threads);
        omp_set_num_threads(num_threads);
        #pragma omp parallel for
        for (int i = 0; i < num_threads; i++)
        {
            int ind = omp_get_thread_num();
            source_1Ds[ind] = malloc_aligned(size_t(N[2])*sizeof(float));
        }
    }

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_dn[0]; i++)
    {
        float* source_2D = NULL;
        if (factors[0] == 1.0)
            source_2D = &I[uint64(i) * img_sz];
        else
        {
            source_2D = source_2Ds[omp_get_thread_num()];
            //memset(source_2D, 0, size_bytes);

            float x = (i - 0.5 * float(N_dn[0] - 1)) * factors[0] + offset[0];
            int ii = int(floor(0.5 + x + u_c));
            float delay_0 = x + u_c - float(ii);

            int L_0;
            float* h_0 = bumpFcn(filterWidths[0], delay_0, L_0, order);
            //printf("size(%d) = %d\n", L_0, 2*L_0+1);

            /*
            memset(source_2D, 0, size_bytes);
            for (int l_0 = -L_0; l_0 <= L_0; l_0++)
            {
                float h_0_cur = h_0[l_0 + L_0];
                if (h_0_cur != 0.0)
                {
                    int ind_0 = max(0, min(N[0]-1, ii + l_0));
                    float* I_offs = &I[uint64(ind_0) * img_sz];
                    for (int j = y_ind_min; j <= y_ind_max; j++)
                    {
                        for (int k = z_ind_min; k <= z_ind_max; k++)
                            source_2D[j*N[2]+k] += h_0_cur * I_offs[j*N[2]+k];
                    }
                    //for (int n = 0; n < N[1]*N[2]; n++)
                    //    source_2D[n] += h_0_cur * I_offs[n];
                }
            }
            //*/

            /*
            int ind_0 = max(0, min(N[0]-1, ii + 0));
            float* I_offs = &I[uint64(ind_0) * img_sz];
            for (int j = y_ind_min; j <= y_ind_max; j++)
            {
                for (int k = z_ind_min; k <= z_ind_max; k++)
                    source_2D[j*N[2]+k] = h_0[0] * I_offs[j*N[2]+k];
            }

            int l_0 = -L_0+1;
            while (l_0+1 <= L_0)
            {
                int ind_0 = max(0, min(N[0]-1, ii + l_0));
                int ind_1 = max(0, min(N[0]-1, ii + l_0+1));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                        source_2D[j*N[2]+k] += h_0[l_0+L_0] * I_offs_0[j*N[2]+k] + h_0[l_0+1+L_0] * I_offs_1[j*N[2]+k];
                }
                l_0 += 2;
            }
            //*/

            //*
            if (L_0 == 1)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 1));
                int ind_1 = max(0, min(N[0]-1, ii + 0));
                int ind_2 = max(0, min(N[0]-1, ii + 1));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 2)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 2));
                int ind_1 = max(0, min(N[0]-1, ii - 1));
                int ind_2 = max(0, min(N[0]-1, ii + 0));
                int ind_3 = max(0, min(N[0]-1, ii + 1));
                int ind_4 = max(0, min(N[0]-1, ii + 2));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 3)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 3));
                int ind_1 = max(0, min(N[0]-1, ii - 2));
                int ind_2 = max(0, min(N[0]-1, ii - 1));
                int ind_3 = max(0, min(N[0]-1, ii + 0));
                int ind_4 = max(0, min(N[0]-1, ii + 1));
                int ind_5 = max(0, min(N[0]-1, ii + 2));
                int ind_6 = max(0, min(N[0]-1, ii + 3));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 4)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 4));
                int ind_1 = max(0, min(N[0]-1, ii - 3));
                int ind_2 = max(0, min(N[0]-1, ii - 2));
                int ind_3 = max(0, min(N[0]-1, ii - 1));
                int ind_4 = max(0, min(N[0]-1, ii + 0));
                int ind_5 = max(0, min(N[0]-1, ii + 1));
                int ind_6 = max(0, min(N[0]-1, ii + 2));
                int ind_7 = max(0, min(N[0]-1, ii + 3));
                int ind_8 = max(0, min(N[0]-1, ii + 4));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];
                float* I_offs_7 = &I[uint64(ind_7) * img_sz];
                float* I_offs_8 = &I[uint64(ind_8) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k] +
                                              h_0[7] * I_offs_7[j*N[2]+k] +
                                              h_0[8] * I_offs_8[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 5)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 5));
                int ind_1 = max(0, min(N[0]-1, ii - 4));
                int ind_2 = max(0, min(N[0]-1, ii - 3));
                int ind_3 = max(0, min(N[0]-1, ii - 2));
                int ind_4 = max(0, min(N[0]-1, ii - 1));
                int ind_5 = max(0, min(N[0]-1, ii + 0));
                int ind_6 = max(0, min(N[0]-1, ii + 1));
                int ind_7 = max(0, min(N[0]-1, ii + 2));
                int ind_8 = max(0, min(N[0]-1, ii + 3));
                int ind_9 = max(0, min(N[0]-1, ii + 4));
                int ind_10 = max(0, min(N[0]-1, ii + 5));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];
                float* I_offs_7 = &I[uint64(ind_7) * img_sz];
                float* I_offs_8 = &I[uint64(ind_8) * img_sz];
                float* I_offs_9 = &I[uint64(ind_9) * img_sz];
                float* I_offs_10 = &I[uint64(ind_10) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k] +
                                              h_0[7] * I_offs_7[j*N[2]+k] +
                                              h_0[8] * I_offs_8[j*N[2]+k] +
                                              h_0[9] * I_offs_9[j*N[2]+k] +
                                              h_0[10] * I_offs_10[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 6)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 6));
                int ind_1 = max(0, min(N[0]-1, ii - 5));
                int ind_2 = max(0, min(N[0]-1, ii - 4));
                int ind_3 = max(0, min(N[0]-1, ii - 3));
                int ind_4 = max(0, min(N[0]-1, ii - 2));
                int ind_5 = max(0, min(N[0]-1, ii - 1));
                int ind_6 = max(0, min(N[0]-1, ii + 0));
                int ind_7 = max(0, min(N[0]-1, ii + 1));
                int ind_8 = max(0, min(N[0]-1, ii + 2));
                int ind_9 = max(0, min(N[0]-1, ii + 3));
                int ind_10 = max(0, min(N[0]-1, ii + 4));
                int ind_11 = max(0, min(N[0]-1, ii + 5));
                int ind_12 = max(0, min(N[0]-1, ii + 6));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];
                float* I_offs_7 = &I[uint64(ind_7) * img_sz];
                float* I_offs_8 = &I[uint64(ind_8) * img_sz];
                float* I_offs_9 = &I[uint64(ind_9) * img_sz];
                float* I_offs_10 = &I[uint64(ind_10) * img_sz];
                float* I_offs_11 = &I[uint64(ind_11) * img_sz];
                float* I_offs_12 = &I[uint64(ind_12) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k] +
                                              h_0[7] * I_offs_7[j*N[2]+k] +
                                              h_0[8] * I_offs_8[j*N[2]+k] +
                                              h_0[9] * I_offs_9[j*N[2]+k] +
                                              h_0[10] * I_offs_10[j*N[2]+k] +
                                              h_0[11] * I_offs_11[j*N[2]+k] +
                                              h_0[12] * I_offs_12[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 7)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 7));
                int ind_1 = max(0, min(N[0]-1, ii - 6));
                int ind_2 = max(0, min(N[0]-1, ii - 5));
                int ind_3 = max(0, min(N[0]-1, ii - 4));
                int ind_4 = max(0, min(N[0]-1, ii - 3));
                int ind_5 = max(0, min(N[0]-1, ii - 2));
                int ind_6 = max(0, min(N[0]-1, ii - 1));
                int ind_7 = max(0, min(N[0]-1, ii + 0));
                int ind_8 = max(0, min(N[0]-1, ii + 1));
                int ind_9 = max(0, min(N[0]-1, ii + 2));
                int ind_10 = max(0, min(N[0]-1, ii + 3));
                int ind_11 = max(0, min(N[0]-1, ii + 4));
                int ind_12 = max(0, min(N[0]-1, ii + 5));
                int ind_13 = max(0, min(N[0]-1, ii + 6));
                int ind_14 = max(0, min(N[0]-1, ii + 7));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];
                float* I_offs_7 = &I[uint64(ind_7) * img_sz];
                float* I_offs_8 = &I[uint64(ind_8) * img_sz];
                float* I_offs_9 = &I[uint64(ind_9) * img_sz];
                float* I_offs_10 = &I[uint64(ind_10) * img_sz];
                float* I_offs_11 = &I[uint64(ind_11) * img_sz];
                float* I_offs_12 = &I[uint64(ind_12) * img_sz];
                float* I_offs_13 = &I[uint64(ind_13) * img_sz];
                float* I_offs_14 = &I[uint64(ind_14) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k] +
                                              h_0[7] * I_offs_7[j*N[2]+k] +
                                              h_0[8] * I_offs_8[j*N[2]+k] +
                                              h_0[9] * I_offs_9[j*N[2]+k] +
                                              h_0[10] * I_offs_10[j*N[2]+k] +
                                              h_0[11] * I_offs_11[j*N[2]+k] +
                                              h_0[12] * I_offs_12[j*N[2]+k] +
                                              h_0[13] * I_offs_13[j*N[2]+k] +
                                              h_0[14] * I_offs_14[j*N[2]+k];
                    }
                }
            }
            else if (L_0 == 8)
            {
                int ind_0 = max(0, min(N[0]-1, ii - 8));
                int ind_1 = max(0, min(N[0]-1, ii - 7));
                int ind_2 = max(0, min(N[0]-1, ii - 6));
                int ind_3 = max(0, min(N[0]-1, ii - 5));
                int ind_4 = max(0, min(N[0]-1, ii - 4));
                int ind_5 = max(0, min(N[0]-1, ii - 3));
                int ind_6 = max(0, min(N[0]-1, ii - 2));
                int ind_7 = max(0, min(N[0]-1, ii - 1));
                int ind_8 = max(0, min(N[0]-1, ii + 0));
                int ind_9 = max(0, min(N[0]-1, ii + 1));
                int ind_10 = max(0, min(N[0]-1, ii + 2));
                int ind_11 = max(0, min(N[0]-1, ii + 3));
                int ind_12 = max(0, min(N[0]-1, ii + 4));
                int ind_13 = max(0, min(N[0]-1, ii + 5));
                int ind_14 = max(0, min(N[0]-1, ii + 6));
                int ind_15 = max(0, min(N[0]-1, ii + 7));
                int ind_16 = max(0, min(N[0]-1, ii + 8));
                float* I_offs_0 = &I[uint64(ind_0) * img_sz];
                float* I_offs_1 = &I[uint64(ind_1) * img_sz];
                float* I_offs_2 = &I[uint64(ind_2) * img_sz];
                float* I_offs_3 = &I[uint64(ind_3) * img_sz];
                float* I_offs_4 = &I[uint64(ind_4) * img_sz];
                float* I_offs_5 = &I[uint64(ind_5) * img_sz];
                float* I_offs_6 = &I[uint64(ind_6) * img_sz];
                float* I_offs_7 = &I[uint64(ind_7) * img_sz];
                float* I_offs_8 = &I[uint64(ind_8) * img_sz];
                float* I_offs_9 = &I[uint64(ind_9) * img_sz];
                float* I_offs_10 = &I[uint64(ind_10) * img_sz];
                float* I_offs_11 = &I[uint64(ind_11) * img_sz];
                float* I_offs_12 = &I[uint64(ind_12) * img_sz];
                float* I_offs_13 = &I[uint64(ind_13) * img_sz];
                float* I_offs_14 = &I[uint64(ind_14) * img_sz];
                float* I_offs_15 = &I[uint64(ind_15) * img_sz];
                float* I_offs_16 = &I[uint64(ind_16) * img_sz];

                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        source_2D[j*N[2]+k] = h_0[0] * I_offs_0[j*N[2]+k] +
                                              h_0[1] * I_offs_1[j*N[2]+k] +
                                              h_0[2] * I_offs_2[j*N[2]+k] +
                                              h_0[3] * I_offs_3[j*N[2]+k] +
                                              h_0[4] * I_offs_4[j*N[2]+k] +
                                              h_0[5] * I_offs_5[j*N[2]+k] +
                                              h_0[6] * I_offs_6[j*N[2]+k] +
                                              h_0[7] * I_offs_7[j*N[2]+k] +
                                              h_0[8] * I_offs_8[j*N[2]+k] +
                                              h_0[9] * I_offs_9[j*N[2]+k] +
                                              h_0[10] * I_offs_10[j*N[2]+k] +
                                              h_0[11] * I_offs_11[j*N[2]+k] +
                                              h_0[12] * I_offs_12[j*N[2]+k] +
                                              h_0[13] * I_offs_13[j*N[2]+k] +
                                              h_0[14] * I_offs_14[j*N[2]+k] +
                                              h_0[15] * I_offs_15[j*N[2]+k] +
                                              h_0[16] * I_offs_16[j*N[2]+k];
                    }
                }
            }
            else
            {
                for (int j = y_ind_min; j <= y_ind_max; j++)
                {
                    for (int k = z_ind_min; k <= z_ind_max; k++)
                    {
                        uint64 ind_shift = uint64(j*N[2]+k);
                        float val = 0.0;
                        for (int l_0 = -L_0; l_0 <= L_0; l_0++)
                            val += h_0[l_0 + L_0] * I[uint64(max(0, min(N[0]-1, ii + l_0)))*img_sz + ind_shift];
                        source_2D[ind_shift] = val;
                    }
                }
            }
            //*/

            delete[] h_0;
        }

        float* target_2D = &I_dn[uint64(i) * uint64(N_dn[2] * N_dn[1])];
        for (int j = 0; j < N_dn[1]; j++)
        {
            float* source_1D = NULL;

            if (factors[1] == 1.0)
                source_1D = &source_2D[j*N[2]];
            else
            {
                source_1D = source_1Ds[omp_get_thread_num()];
                memset(source_1D, 0, size_t(N[2])*sizeof(float));

                float y = (j - 0.5 * float(N_dn[1] - 1)) * factors[1] + offset[1];
                int jj = int(floor(0.5 + y + v_c));
                float delay_1 = y + v_c - float(jj);

                int L_1;
                float* h_1 = bumpFcn(filterWidths[1], delay_1, L_1, order);

                for (int l_1 = -L_1; l_1 <= L_1; l_1++)
                {
                    float h_1_cur = h_1[l_1 + L_1];
                    if (h_1_cur != 0.0)
                    {
                        int ind_1 = max(0, min(N[1] - 1, jj + l_1));
                        float* source_2D_offs = &source_2D[ind_1 * N[2]];
                        for (int n = z_ind_min; n <= z_ind_max; n++)
                            source_1D[n] += h_1_cur * source_2D_offs[n];
                    }
                }

                delete[] h_1;
            }

            // about 7.5% speed-up
            float* target_1D = &target_2D[j * N_dn[2]];
            for (int k = 0; k < N_dn[2]; k++)
            {
                float z = (k - 0.5 * float(N_dn[2] - 1)) * factors[2] + offset[2];
                int kk = int(floor(0.5 + z + w_c));

                int L_2 = h_2_lengths[k];
                float* h_2 = h_2_filters[k];
                float val = 0.0;
                int l_2 = -L_2;
                while (l_2+1 <= L_2)
                {
                    val += h_2[l_2   + L_2] * source_1D[max(0, min(N[2] - 1, kk + l_2))] +
                           h_2[l_2+1 + L_2] * source_1D[max(0, min(N[2] - 1, kk + l_2+1))];
                    l_2 += 2;
                }
                target_1D[k] = val + h_2[2*L_2] * source_1D[max(0, min(N[2] - 1, kk + L_2))];
            }
        }
    }

    // clean up
    if (source_2Ds != NULL)
    {
        for (int i = 0; i < num_threads; i++)
            free_aligned(source_2Ds[i]);
        free(source_2Ds);
    }

    if (source_1Ds != NULL)
    {
        for (int i = 0; i < num_threads; i++)
            free_aligned(source_1Ds[i]);
        free(source_1Ds);
    }

    if (h_2_lengths != NULL)
        free(h_2_lengths);
    if (h_2_filters != NULL)
    {
        for (int k = 0; k < N_dn[2]; k++)
        {
            float* h_2 = h_2_filters[k];
            //delete [] h_2;
            free_aligned(h_2);
            h_2_filters[k] = NULL;
        }
        free(h_2_filters);
    }

    return true;
}

/* MMF upsample support radius in input-index units (matches mff_upsample_1d in maximally_flat_filter.py). */
static int upSample_mff_support_radius(int order)
{
    if (order < 2)
        return 1;
    order = order - (order % 2);
    return order / 2 + 2;
}

/* Precomputed 1D MMF upsample filters for one axis (O(Ndst * max_taps) memory, not volume-sized). */
struct UpSample1dMffPlan
{
    int Ndst;
    int max_taps;
    int* k_lo;
    int* n_taps;
    float* h;
};

static void upSample_1d_mff_plan_free(UpSample1dMffPlan* plan);

static bool upSample_1d_mff_plan_build(UpSample1dMffPlan* plan, int Nsrc, int Ndst, float L, int order)
{
    if (plan == NULL || Ndst <= 0 || Nsrc <= 0)
        return false;

    memset(plan, 0, sizeof(UpSample1dMffPlan));
    plan->Ndst = Ndst;

    const int radius = upSample_mff_support_radius(order);
    plan->max_taps = 2 * radius + 1;

    const size_t h_bytes = size_t(Ndst) * size_t(plan->max_taps) * sizeof(float);

    plan->k_lo = (int*)malloc(size_t(Ndst) * sizeof(int));
    plan->n_taps = (int*)malloc(size_t(Ndst) * sizeof(int));
    plan->h = (float*)malloc_aligned(h_bytes);
    if (plan->k_lo == NULL || plan->n_taps == NULL || plan->h == NULL)
    {
        upSample_1d_mff_plan_free(plan);
        return false;
    }
    memset(plan->h, 0, h_bytes);

    const float c_dst = 0.5f * float(Ndst - 1);
    const float c_src = 0.5f * float(Nsrc - 1);

    #pragma omp parallel for
    for (int m = 0; m < Ndst; m++)
    {
        const float t = (float(m) - c_dst) / L + c_src;
        const int k_lo = max(0, int(ceilf(t - float(radius))));
        const int k_hi = min(Nsrc - 1, int(floorf(t + float(radius))));
        const int ntaps = max(0, k_hi - k_lo + 1);

        plan->k_lo[m] = k_lo;
        plan->n_taps[m] = ntaps;

        float* hf = plan->h + size_t(m) * size_t(plan->max_taps);
        for (int j = 0; j < ntaps; j++)
            hf[j] = mff_filter(t - float(k_lo + j), order);
    }

    return true;
}

static void upSample_1d_mff_plan_free(UpSample1dMffPlan* plan)
{
    if (plan == NULL)
        return;
    if (plan->k_lo != NULL)
        free(plan->k_lo);
    if (plan->n_taps != NULL)
        free(plan->n_taps);
    if (plan->h != NULL)
        free_aligned(plan->h);
    memset(plan, 0, sizeof(UpSample1dMffPlan));
}

static void upSample_1d_mff_apply_planned(const float* src, float* dst, const UpSample1dMffPlan* plan)
{
    if (src == NULL || dst == NULL || plan == NULL || plan->Ndst <= 0)
        return;

    for (int m = 0; m < plan->Ndst; m++)
    {
        const int ntaps = plan->n_taps[m];
        if (ntaps <= 0)
        {
            dst[m] = 0.0f;
            continue;
        }

        const float* hf = plan->h + size_t(m) * size_t(plan->max_taps);
        const int k0 = plan->k_lo[m];
        const float* src0 = src + k0;

        float acc = 0.0f;
        int j = 0;
        while (j + 3 < ntaps)
        {
            acc += hf[j] * src0[j] + hf[j + 1] * src0[j + 1] + hf[j + 2] * src0[j + 2] + hf[j + 3] * src0[j + 3];
            j += 4;
        }
        for (; j < ntaps; j++)
            acc += hf[j] * src0[j];

        dst[m] = acc;
    }
}

bool upSample_high_order_cpu(float* I, int* N, float* I_up, int* N_up, float* factors, int set_type, int order)
{
    /*
    Separable 3-pass MMF upsampling (z, y, x) with precomputed 1D filters.  Centers match upSample_cpu.
    Uses stage12 (N[0] x N_up[1] x N_up[2]) for high parallelism; pass 1 packs into the start of I_up.
    //*/
    if (I == NULL || N == NULL || I_up == NULL || N_up == NULL || factors == NULL)
        return false;
    if (set_type != 0)
    {
        printf("Error: upSample_high_order_cpu currently only supports set_type == 0\n");
        return false;
    }
    if (factors[0] < 1.0f || factors[1] < 1.0f || factors[2] < 1.0f)
        return false;

    const uint64 vol_up = uint64(N_up[0]) * uint64(N_up[1]) * uint64(N_up[2]);
    const uint64 pack1_elems = uint64(N[0]) * uint64(N[1]) * uint64(N_up[2]);
    if (pack1_elems > vol_up)
    {
        printf("Error: upSample_high_order_cpu output buffer too small for intermediate layout "
               "(need at least N[0]*N[1]*N_up[2] = %llu elements, have %llu)\n",
               (unsigned long long)pack1_elems, (unsigned long long)vol_up);
        return false;
    }

    if (order == 2)
        order = 0;

    UpSample1dMffPlan plan_z, plan_y, plan_x;
    memset(&plan_z, 0, sizeof(plan_z));
    memset(&plan_y, 0, sizeof(plan_y));
    memset(&plan_x, 0, sizeof(plan_x));

    if (!upSample_1d_mff_plan_build(&plan_z, N[2], N_up[2], factors[2], order) ||
        !upSample_1d_mff_plan_build(&plan_y, N[1], N_up[1], factors[1], order) ||
        !upSample_1d_mff_plan_build(&plan_x, N[0], N_up[0], factors[0], order))
    {
        upSample_1d_mff_plan_free(&plan_z);
        upSample_1d_mff_plan_free(&plan_y);
        upSample_1d_mff_plan_free(&plan_x);
        return false;
    }

    const uint64 img_sz = uint64(N[1]) * uint64(N[2]);
    const uint64 plane_up = uint64(N_up[1]) * uint64(N_up[2]);
    const size_t tls_stride = max((size_t)N[1] + (size_t)N_up[1], (size_t)N[0] + (size_t)N_up[0]);
    const int num_th = num_cpu_threads();

    const size_t stage12_sz = size_t(N[0]) * size_t(N_up[1]) * size_t(N_up[2]);
    float* stage12 = (float*)malloc_aligned(stage12_sz * sizeof(float));
    float* tls = (float*)malloc(sizeof(float) * tls_stride * (size_t)num_th);
    if (stage12 == NULL || tls == NULL)
    {
        free_aligned(stage12);
        free(tls);
        upSample_1d_mff_plan_free(&plan_z);
        upSample_1d_mff_plan_free(&plan_y);
        upSample_1d_mff_plan_free(&plan_x);
        return false;
    }

    omp_set_num_threads(num_th);

    const size_t pack1_stride = size_t(N[1]) * size_t(N_up[2]);

    // --- Pass 1 (z): parallel over (i,j) ---
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N[0]; ii++)
        for (int jj = 0; jj < N[1]; jj++)
        {
            float* pack1_base = &I_up[size_t(ii) * pack1_stride];
            const float* src_line = &I[uint64(ii) * img_sz + uint64(jj) * uint64(N[2])];
            float* dst_line = &pack1_base[size_t(jj) * size_t(N_up[2])];
            upSample_1d_mff_apply_planned(src_line, dst_line, &plan_z);
        }

    // --- Pass 2 (y): parallel over (i,k) ---
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N[0]; ii++)
        for (int k = 0; k < N_up[2]; k++)
        {
            const float* slab_in = &I_up[size_t(ii) * pack1_stride];
            float* slab_out = &stage12[size_t(ii) * plane_up];
            float* col_src = tls + (size_t)omp_get_thread_num() * tls_stride;
            float* col_dst = col_src + N[1];
            for (int jj = 0; jj < N[1]; jj++)
                col_src[jj] = slab_in[size_t(jj) * size_t(N_up[2]) + size_t(k)];
            upSample_1d_mff_apply_planned(col_src, col_dst, &plan_y);
            for (int jout = 0; jout < N_up[1]; jout++)
                slab_out[size_t(jout) * size_t(N_up[2]) + size_t(k)] = col_dst[jout];
        }

    // --- Pass 3 (x): parallel over (j,k) ---
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N_up[1]; j++)
        for (int k = 0; k < N_up[2]; k++)
        {
            float* row_src = tls + (size_t)omp_get_thread_num() * tls_stride;
            float* row_dst = row_src + N[0];
            for (int ii = 0; ii < N[0]; ii++)
                row_src[ii] = stage12[size_t(ii) * plane_up + size_t(j) * size_t(N_up[2]) + size_t(k)];
            upSample_1d_mff_apply_planned(row_src, row_dst, &plan_x);
            for (int iout = 0; iout < N_up[0]; iout++)
                I_up[uint64(iout) * plane_up + uint64(j) * uint64(N_up[2]) + uint64(k)] = row_dst[iout];
        }

    free_aligned(stage12);
    free(tls);
    upSample_1d_mff_plan_free(&plan_z);
    upSample_1d_mff_plan_free(&plan_y);
    upSample_1d_mff_plan_free(&plan_x);

    return true;
}

bool upSample_cpu(float* I, int* N, float* I_up, int* N_up, float* factors, int set_type, int order)
{
    if (order > 2)
        return upSample_high_order_cpu(I, N, I_up, N_up, factors, set_type, order);
    if (I == NULL || N == NULL || I_up == NULL || N_up == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;
    //printf("up-sample: (%d, %d, %d) ==> (%d, %d, %d)\n", N[0], N[1], N[2], N_up[0], N_up[1], N_up[2]);

    float u_c = 0.5 * float(N[0] - 1);
    float v_c = 0.5 * float(N[1] - 1);
    float w_c = 0.5 * float(N[2] - 1);

    float min_trans = 1.0 / 65535.0;

    // u = (ii - 0.5*(N[0]-1))
    // x = (i - 0.5*(N_up[0]-1))/factors[0]

    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_up[0]; i++)
    {
        float* target_2D = &I_up[uint64(i) * uint64(N_up[2] * N_up[1])];
        if (factors[0] == 1.0)
        {
            float* source_2D = &I[uint64(i) * uint64(N[2] * N[1])];
            float* half_up_1D = new float[N[2]];
            for (int j = 0; j < N_up[1]; j++)
            {
                float* target_1D = &target_2D[j * N_up[2]];
                
                float y = (j - 0.5 * float(N_up[1] - 1)) / factors[1];
                if (y+v_c < 0.0)
                {
                    float d_2 = y+v_c;
                    for (int k = 0; k < N[2]; k++)
                        half_up_1D[k] = d_2 * source_2D[1 * N[2] + k] + (1.0 - d_2) * source_2D[k];

                }
                else if (N[1]-1 < y+v_c)
                {
                    float d_2 = y+v_c-(N[1]-1);
                    for (int k = 0; k < N[2]; k++)
                        half_up_1D[k] = (1.0 + d_2) * source_2D[(N[1]-1) * N[2] + k] - d_2 * source_2D[(N[1]-2) * N[2] + k];
                }
                else //if (0.0 <= y+v_c && y+v_c <= N[1]-1)
                {
                    int ind_2_lo = int(floor(y + v_c));
                    int ind_2_hi = ind_2_lo + 1;
                    float d_2 = y + v_c - float(ind_2_lo);
                    ind_2_lo = max(0, min(ind_2_lo, N[1] - 1));
                    ind_2_hi = max(0, min(ind_2_hi, N[1] - 1));
                    
                    for (int k = 0; k < N[2]; k++)
                        half_up_1D[k] = (1.0 - d_2) * source_2D[ind_2_lo * N[2] + k] + d_2 * source_2D[ind_2_hi * N[2] + k];
                }
                
                for (int k = 0; k < N_up[2]; k++)
                {
                    float z = (k - 0.5 * float(N_up[2] - 1)) / factors[2];
                    float partA;
                    if (z+w_c < 0.0)
                    {
                        float d_3 = z+w_c;
                        partA = d_3 * half_up_1D[1] + (1.0 - d_3) * half_up_1D[0];
                    }
                    else if (N[2]-1 < z+w_c)
                    {
                        float d_3 = z+w_c-(N[2]-1);
                        partA = (1.0 + d_3) * half_up_1D[N[2]-1] - d_3 * half_up_1D[N[2]-2];
                    }
                    else
                    {
                        int ind_3_lo = int(floor(z + w_c));
                        int ind_3_hi = ind_3_lo + 1;
                        float d_3 = z + w_c - float(ind_3_lo);
                        ind_3_lo = max(0, min(ind_3_lo, N[2] - 1));
                        ind_3_hi = max(0, min(ind_3_hi, N[2] - 1));
                        partA = (1.0 - d_3) * half_up_1D[ind_3_lo] + d_3 * half_up_1D[ind_3_hi];
                    }

                    if (set_type == 0)
                        target_1D[k] = partA;
                    else if (set_type == 1)
                        target_1D[k] += partA;
                    else if (set_type == 2)
                        target_1D[k] *= partA;
                    else  if (set_type == 3)
                    {
                        float temp = partA;
                        float curVal = target_1D[k];
                        if (curVal != 0.0 && temp != 0.0)
                            target_1D[k] = smooth_abs6_scaled(curVal, temp);
                            //target_1D[k] = sqrt(sqrt(curVal*curVal*curVal*curVal + temp*temp*temp*temp)) - temp + min_trans;
                            //target_1D[k] = curVal * curVal / (curVal + temp);
                    }
                    else  //if (set_type == 4)
                    {
                        float temp = partA;
                        float curVal = target_1D[k];
                        //float p_plus_s = curVal + temp - min_trans;
                        if (curVal >= min_trans && temp != 0.0)
                            target_1D[k] = smooth_abs6_scaled_inv(curVal, temp);
                            //target_1D[k] = sqrt(sqrt(p_plus_s*p_plus_s*p_plus_s*p_plus_s - temp*temp*temp*temp));
                            //target_1D[k] = curVal * (curVal + temp) / curVal;
                    }
                }
            }
            delete [] half_up_1D;
        }
        else
        {
            float x = (i - 0.5 * float(N_up[0] - 1)) / factors[0];
            int ind_1_lo = int(floor(x + u_c));
            int ind_1_hi = ind_1_lo + 1;
            float d_1 = x + u_c - float(ind_1_lo);
            ind_1_lo = max(0, min(ind_1_lo, N[0] - 1));
            ind_1_hi = max(0, min(ind_1_hi, N[0] - 1));
            float* source_2D_lo = &I[uint64(ind_1_lo) * uint64(N[2] * N[1])];
            float* source_2D_hi = &I[uint64(ind_1_hi) * uint64(N[2] * N[1])];
            for (int j = 0; j < N_up[1]; j++)
            {
                float* target_1D = &target_2D[j * N_up[2]];
                float y = (j - 0.5 * float(N_up[1] - 1)) / factors[1];
                int ind_2_lo = int(floor(y + v_c));
                int ind_2_hi = ind_2_lo + 1;
                float d_2 = y + v_c - float(ind_2_lo);
                ind_2_lo = max(0, min(ind_2_lo, N[1] - 1));
                ind_2_hi = max(0, min(ind_2_hi, N[1] - 1));
                for (int k = 0; k < N_up[2]; k++)
                {
                    float z = (k - 0.5 * float(N_up[2] - 1)) / factors[2];
                    int ind_3_lo = int(floor(z + w_c));
                    int ind_3_hi = ind_3_lo + 1;
                    float d_3 = z + w_c - float(ind_3_lo);
                    ind_3_lo = max(0, min(ind_3_lo, N[2] - 1));
                    ind_3_hi = max(0, min(ind_3_hi, N[2] - 1));
                    float partA = (1.0-d_2)*((1.0 - d_3) * source_2D_lo[ind_2_lo * N[2] + ind_3_lo] + d_3 * source_2D_lo[ind_2_lo * N[2] + ind_3_hi])
                    + d_2 * ((1.0 - d_3) * source_2D_lo[ind_2_hi * N[2] + ind_3_lo] + d_3 * source_2D_lo[ind_2_hi * N[2] + ind_3_hi]);
                    float partB = (1.0 - d_2) * ((1.0 - d_3) * source_2D_hi[ind_2_lo * N[2] + ind_3_lo] + d_3 * source_2D_hi[ind_2_lo * N[2] + ind_3_hi])
                    +d_2 * ((1.0 - d_3) * source_2D_hi[ind_2_hi * N[2] + ind_3_lo] + d_3 * source_2D_hi[ind_2_hi * N[2] + ind_3_hi]);
                    if (set_type == 0)
                        target_1D[k] = (1.0 - d_1) * partA + d_1 * partB;
                    else if (set_type == 1)
                        target_1D[k] += (1.0 - d_1) * partA + d_1 * partB;
                    else if (set_type == 2)
                        target_1D[k] *= (1.0 - d_1) * partA + d_1 * partB;
                    else  if (set_type == 3)
                    {
                        float temp = (1.0 - d_1) * partA + d_1 * partB;
                        float curVal = target_1D[k];
                        if (curVal != 0.0 && temp != 0.0)
                            target_1D[k] = smooth_abs6_scaled(curVal, temp);
                            //target_1D[k] = sqrt(sqrt(curVal*curVal*curVal*curVal + temp*temp*temp*temp)) - temp + min_trans;
                            //target_1D[k] = curVal * curVal / (curVal + temp);
                    }
                    else  //if (set_type == 4)
                    {
                        float temp = (1.0 - d_1) * partA + d_1 * partB;
                        float curVal = target_1D[k];
                        //float p_plus_s = curVal + temp - min_trans;
                        if (curVal >= min_trans && temp != 0.0)
                            target_1D[k] = smooth_abs6_scaled_inv(curVal, temp);
                            //target_1D[k] = sqrt(sqrt(p_plus_s*p_plus_s*p_plus_s*p_plus_s - temp*temp*temp*temp));
                            //target_1D[k] = curVal * (curVal + temp) / curVal;
                    }
                }
            }
        }
    }
    return true;
}

bool finite_difference(float* volume, int* N, int order, int shift, bool* axis, float scalar)
{
    int N_h;
    float* h = first_order_finite_difference_filter(N_h, order, shift);
    float* h_pad = NULL;
    if (N_h % 2 == 0)
    {
        h_pad = (float*) calloc(size_t(N_h+1), sizeof(float));
        for (int i = 0; i < N_h; i++)
        {
            if (shift < 0)
                h_pad[i] = h[i];
            else
                h_pad[i+1] = h[i];
        }
        free(h);
        h = h_pad;
        N_h += 1;
    }

    if (scalar != 1.0)
    {
        for (int i = 0; i < N_h; i++)
            h[i] *= scalar;
    }

    /*
    for (int i = 0; i < N_h; i++)
        printf("%f ", h[i]);
    printf("\n");
    //*/

    bool delete_axis = false;
    if (axis == NULL)
    {
        delete_axis = true;
        axis = (bool*) malloc(sizeof(bool)*3);
        axis[0] = false;
        axis[1] = false;
        axis[2] = true;
    }

    bool retVal = antialias_filter(volume, N, 2, 2, h, N_h, axis);
    free(h);
    if (delete_axis && axis != NULL)
        free(axis);
    return retVal;
}

bool antialias_filter(float* volume, int* N, float L, int order, float* h_in, int N_h, bool* axis)
{
    if (volume == NULL || N == NULL || N[0] <= 0 || N[1] <= 0 || N[2] <= 0)
        return false;
    if (L <= 1.0)
        return true;
    order = max(-2, min(16, order));
    if (order == 2)
        order = 0;

    uint64 img_sz = uint64(N[1]) * uint64(N[2]);

    int L_x = 1;
    float* h = NULL;
    float* h_unaligned = NULL;

    if (h_in == NULL || N_h <= 0)
    {
        h_unaligned = bumpFcn(L, 0.0, L_x, order);

        size_t size_bytes = size_t(2*L_x+1) * sizeof(float);
        h = malloc_aligned(size_bytes);
        memcpy(h, h_unaligned, size_bytes);
        delete [] h_unaligned;
    }
    else
    {
        if (N_h < 1)
        {
            printf("antialias_filter: Invalid filter size\n");
            return false;
        }

        if (N_h % 2 == 0)
        {
            L_x = N_h/2;
            size_t size_bytes = size_t(2*L_x+1) * sizeof(float);
            h = malloc_aligned(size_bytes);
            memcpy(h, h_in, size_bytes-sizeof(float));
            h[2*L_x] = 0.0;
        }
        else
        {
            L_x = (N_h-1)/2;
            size_t size_bytes = size_t(2*L_x+1) * sizeof(float);
            h = malloc_aligned(size_bytes);
            memcpy(h, h_in, size_bytes);
        }
    }

    int num_threads = num_cpu_threads();

    float** lines_0a = NULL;
    float** lines_0b = NULL;
    float** lines_0c = NULL;
    float** lines_0d = NULL;

    if (axis == NULL || axis[0] == true)
    {
        lines_0a = (float**) malloc(sizeof(float*)*num_threads);
        lines_0b = (float**) malloc(sizeof(float*)*num_threads);
        lines_0c = (float**) malloc(sizeof(float*)*num_threads);
        lines_0d = (float**) malloc(sizeof(float*)*num_threads);
    }
    float** lines_1 = (float**) malloc(sizeof(float*)*num_threads);
    float** lines_2 = (float**) malloc(sizeof(float*)*num_threads);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++)
    {
        int thread_num = omp_get_thread_num();
        if (lines_0a != NULL)
            lines_0a[thread_num] = malloc_aligned(size_t(N[0])*sizeof(float));
        if (lines_0b != NULL)
            lines_0b[thread_num] = malloc_aligned(size_t(N[0])*sizeof(float));
        if (lines_0c != NULL)
            lines_0c[thread_num] = malloc_aligned(size_t(N[0])*sizeof(float));
        if (lines_0d != NULL)
            lines_0d[thread_num] = malloc_aligned(size_t(N[0])*sizeof(float));
        lines_1[thread_num] = malloc_aligned(size_t(N[1])*sizeof(float));
        lines_2[thread_num] = malloc_aligned(size_t(N[2])*sizeof(float));
    }

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N[0]; i++)
    {
        float* input_2D = &volume[uint64(i)*img_sz];

        int thread_num = omp_get_thread_num();
        float* aLine_1 = lines_1[thread_num];
        float* aLine_2 = lines_2[thread_num];

        // filter over columns
        if (axis == NULL || axis[2] == true)
        {
            for (int j = 0; j < N[1]; j++)
            {
                float* input_1D = &input_2D[j*N[2]];
                float* aLine = aLine_2;
                memcpy(aLine, input_1D, sizeof(float) * N[2]);
                for (int k = 0; k < N[2]; k++)
                {
                    float val = 0.0;
                    for (int n = -L_x; n <= L_x; n++)
                    {
                        float h_cur = h[n+L_x];
                        int ind = max(0, min(N[2]-1, k+n));
                        if (h_cur != 0.0)
                            val += h_cur * aLine[ind];
                    }
                    input_1D[k] = val;
                }
            }
        }

        // filter over rows
        if (axis == NULL || axis[1] == true)
        {
            for (int k = 0; k < N[2]; k++)
            {
                float* aLine = aLine_1;
                for (int j = 0; j < N[1]; j++)
                    aLine[j] = input_2D[j*N[2] + k];

                for (int j = 0; j < N[1]; j++)
                {
                    float val = 0.0;
                    for (int n = -L_x; n <= L_x; n++)
                    {
                        float h_cur = h[n+L_x];
                        int ind = max(0, min(N[1]-1, j+n));
                        if (h_cur != 0.0)
                            val += h_cur * aLine[ind];
                    }
                    input_2D[j*N[2]+k] = val;
                }
            }
        }
    }

    if (axis == NULL || axis[0] == true)
    {
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < N[1]; j++)
        {
            int thread_num = omp_get_thread_num();
            float* aLine_0 = lines_0a[thread_num];
            float* aLine_1 = lines_0b[thread_num];
            float* aLine_2 = lines_0c[thread_num];
            float* aLine_3 = lines_0d[thread_num];

            // filter over first dimension
            // filter 4 at a time to take advantage of cache lines
            for (int k = 0; k < N[2]; k+=4)
            {
                uint64 ind_offs_0 = uint64(j*N[2] + min(k+0, N[2]-1));
                uint64 ind_offs_1 = uint64(j*N[2] + min(k+1, N[2]-1));
                uint64 ind_offs_2 = uint64(j*N[2] + min(k+2, N[2]-1));
                uint64 ind_offs_3 = uint64(j*N[2] + min(k+3, N[2]-1));

                for (int i = 0; i < N[0]; i++)
                {
                    aLine_0[i] = volume[uint64(i)*img_sz + ind_offs_0];
                    aLine_1[i] = volume[uint64(i)*img_sz + ind_offs_1];
                    aLine_2[i] = volume[uint64(i)*img_sz + ind_offs_2];
                    aLine_3[i] = volume[uint64(i)*img_sz + ind_offs_3];
                }

                for (int i = 0; i < N[0]; i++)
                {
                    float val_0 = 0.0;
                    float val_1 = 0.0;
                    float val_2 = 0.0;
                    float val_3 = 0.0;
                    for (int n = -L_x; n <= L_x; n++)
                    {
                        float h_cur = h[n+L_x];
                        int ind = max(0, min(N[0]-1, i+n));
                        if (h_cur != 0.0)
                        {
                            val_0 += h_cur * aLine_0[ind];
                            val_1 += h_cur * aLine_1[ind];
                            val_2 += h_cur * aLine_2[ind];
                            val_3 += h_cur * aLine_3[ind];
                        }
                    }
                    volume[uint64(i)*img_sz + ind_offs_0] = val_0;
                    volume[uint64(i)*img_sz + ind_offs_1] = val_1;
                    volume[uint64(i)*img_sz + ind_offs_2] = val_2;
                    volume[uint64(i)*img_sz + ind_offs_3] = val_3;
                }

            }
        }
    }

    for (int i = 0; i < num_threads; i++)
    {
        if (lines_0a != NULL)
            free_aligned(lines_0a[i]);
        if (lines_0b != NULL)
            free_aligned(lines_0b[i]);
        if (lines_0c != NULL)
            free_aligned(lines_0c[i]);
        if (lines_0d != NULL)
            free_aligned(lines_0d[i]);
        free_aligned(lines_1[i]);
        free_aligned(lines_2[i]);
    }
    if (lines_0a != NULL)
        free(lines_0a);
    if (lines_0b != NULL)
        free(lines_0b);
    if (lines_0c != NULL)
        free(lines_0c);
    if (lines_0d != NULL)
        free(lines_0d);
    free(lines_1);
    free(lines_2);

    free_aligned(h);
    return true;
}
