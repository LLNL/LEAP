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
#include "leap_defines.h"
#include "resample_cpu.h"

using namespace std;

float* bumpFcn(float W, float delay, int& L)
{
    //delay *= -1.0;
    L = int(floor(W));
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

bool downSample_cpu(float* I, int* N, float* I_dn, int* N_dn, float* factors)
{
    if (I == NULL || N == NULL || I_dn == NULL || N_dn == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;

    //printf("down-sample: (%d, %d, %d) ==> (%d, %d, %d)\n", N[0], N[1], N[2], N_dn[0], N_dn[1], N_dn[2]);

    float u_c = 0.5 * float(N[0] - 1);
    float v_c = 0.5 * float(N[1] - 1);
    float w_c = 0.5 * float(N[2] - 1);

    // u = (ii - 0.5*(N[0]-1))
    // x = (i - 0.5*(N_dn[0]-1))*factors[0]
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < N_dn[0]; i++)
    {
        float* target_2D = &I_dn[uint64(i) * uint64(N_dn[2]*N_dn[1])];

        float x = (i - 0.5 * float(N_dn[0] - 1)) * factors[0];
        int ii = int(floor(0.5 + x + u_c));
        float delay_0 = x + u_c - float(ii);

        int L_0;
        float* h_0 = bumpFcn(factors[0], delay_0, L_0);

        for (int j = 0; j < N_dn[1]; j++)
        {
            float* target_1D = &target_2D[j * N_dn[2]];

            float y = (j - 0.5 * float(N_dn[1] - 1)) * factors[1];
            int jj = int(floor(0.5 + y + v_c));
            float delay_1 = y + v_c - float(jj);

            int L_1;
            float* h_1 = bumpFcn(factors[1], delay_1, L_1);

            for (int k = 0; k < N_dn[2]; k++)
            {
                float z = (k - 0.5 * float(N_dn[2] - 1)) * factors[2];
                int kk = int(floor(0.5 + z + w_c));
                float delay_2 = z + w_c - float(kk);

                int L_2;
                float* h_2 = bumpFcn(factors[2], delay_2, L_2);
                float val = 0.0;
                for (int l_0 = -L_0; l_0 <= L_0; l_0++)
                {
                    int ind_0 = max(0, min(N[0]-1, ii + l_0));
                    float* source_2D = &I[uint64(ind_0) * uint64(N[2] * N[1])];

                    float h_0_cur = h_0[l_0 + L_0];
                    if (h_0_cur == 0.0)
                        continue;

                    for (int l_1 = -L_1; l_1 <= L_1; l_1++)
                    {
                        int ind_1 = max(0, min(N[1] - 1, jj + l_1));
                        float* source_1D = &source_2D[ind_1 * N[2]];

                        float h_1_cur = h_1[l_1 + L_1];
                        if (h_1_cur == 0.0)
                            continue;

                        for (int l_2 = -L_2; l_2 <= L_2; l_2++)
                        {
                            int ind_2 = max(0, min(N[2] - 1, kk + l_2));
                            //float h_2_cur = h_2[l_2 + L_2];

                            val += h_0_cur * h_1_cur * h_2[l_2 + L_2] * source_1D[ind_2];
                        }
                    }
                }
                target_1D[k] = val;
                delete[] h_2;
            }
            delete[] h_1;
        }
        delete[] h_0;
    }
    return true;
}

bool upSample_cpu(float* I, int* N, float* I_up, int* N_up, float* factors)
{
    if (I == NULL || N == NULL || I_up == NULL || N_up == NULL || factors == NULL)
        return false;
    if (factors[0] < 1.0 || factors[1] < 1.0 || factors[2] < 1.0)
        return false;

    //printf("up-sample: (%d, %d, %d) ==> (%d, %d, %d)\n", N[0], N[1], N[2], N_up[0], N_up[1], N_up[2]);

    float u_c = 0.5 * float(N[0] - 1);
    float v_c = 0.5 * float(N[1] - 1);
    float w_c = 0.5 * float(N[2] - 1);

    // u = (ii - 0.5*(N[0]-1))
    // x = (i - 0.5*(N_up[0]-1))/factors[0]

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < N_up[0]; i++)
    {
        float* target_2D = &I_up[uint64(i) * uint64(N_up[2] * N_up[1])];

        float x = (i - 0.5 * float(N_up[0] - 1)) / factors[0];
        int ind_1_lo = int(floor(x + u_c));
        int ind_1_hi = ind_1_lo + 1;
        float d_1 = x + u_c - float(ind_1_lo);
        ind_1_lo = max(0, min(ind_1_lo, N[0] - 1));
        ind_1_hi = max(0, min(ind_1_hi, N[0] - 1));
        float* source_2D_lo = &I[uint64(ind_1_lo) * uint64(N[2] * N[1])];
        float* source_2D_hi = &I[uint64(ind_1_hi) * uint64(N[2] * N[1])];

        //printf("%f(%d) => %f(%d)\n", x, i, float(ind_1_lo)+d_1, ind_1_lo);

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

                target_1D[k] = (1.0 - d_1) * partA + d_1 * partB;
                //target_1D[k] = (1.0-d_3)*source_2D_lo[ind_2_lo * N[2] + ind_3_lo] + d_3*source_2D_lo[ind_2_lo * N[2] + ind_3_hi];
            }
        }
    }
    return true;
}
