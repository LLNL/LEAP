////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for CPU-based ramp and Hilbert filters
////////////////////////////////////////////////////////////////////////////////
#include "ramp_filter_cpu.h"
#include "ray_weighting_cpu.h"
#include "cpu_utils.h"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include <iostream>
#include <valarray>
#include <omp.h>

//#ifndef __USE_CPU
//#include "ray_weighting.cuh"
//#endif

#ifndef PI
#define PI 3.141592653589793
#endif

//typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// source: https://tfetimes.com/c-fast-fourier-transform/
// Cooley-Tukey FFT (in-place)
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray even = x[std::slice(0, N / 2, 2)];
    CArray  odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k)
    {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    //x /= x.size();
}

bool rampFilter1D_cpu(float*& g, parameters* params, float scalar)
{
    return conv1D_cpu(g, params, scalar, 0);
}

bool Hilbert1D_cpu(float*& g, parameters* params, float scalar)
{
    return conv1D_cpu(g, params, scalar, 1);
}

bool conv1D_cpu(float*& g, parameters* params, float scalar, int whichFilter)
{
    int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    //int N_H = optimalFFTsize(2 * params->numCols);
    //int N_H_over2 = N_H / 2 + 1;
    float* H_real = NULL;
    Complex* H_comp = NULL;
    if (whichFilter == 0)
        H_real = rampFilterFrequencyResponseMagnitude_cpu(N_H, params);
    else
        H_comp = HilbertTransformFrequencyResponse_cpu(N_H, params);
    if (scalar != 1.0)
    {
        for (int i = 0; i < N_H; i++)
        {
            if (H_real != NULL)
                H_real[i] *= scalar;
            if (H_comp != NULL)
                H_comp[i] *= scalar;
        }
    }
    //for (int i = 0; i < N_H; i++)
    //    printf("%f, %f\n", real(H_comp[i])*float(N_H), imag(H_comp[i]) * float(N_H));

    int numExtrapolate = 0;
    if (params->truncatedScan)
        numExtrapolate = std::min(N_H-params->numCols-1, 100);

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        Complex* paddedArray = new Complex[N_H];

        for (int iRow = 0; iRow < params->numRows; iRow++)
        {
            float* proj_line = &g[uint64(iphi) * uint64(params->numRows * params->numCols) + uint64(iRow * params->numCols)];

            float leftVal = proj_line[0];
            float rightVal = proj_line[params->numCols - 1];

            // Put in padded array
            for (int i = 0; i < N_H; i++)
            {
                if (i < params->numCols)
                    paddedArray[i] = proj_line[i];
                else
                    paddedArray[i] = 0.0;
            }

            if (numExtrapolate > 0)
            {
                float leftVal = proj_line[0];
                float rightVal = proj_line[params->numCols - 1];
                for (int i = params->numCols; i < params->numCols + numExtrapolate; i++)
                    paddedArray[i] = rightVal;
                for (int i = N_H - numExtrapolate; i < N_H; i++)
                    paddedArray[i] = leftVal;
            }

            CArray data(paddedArray, N_H);

            // Do FFT
            fft(data);

            // Multiply by filter
            for (int i = 0; i < N_H; i++)
            {
                if (H_real != NULL)
                    data[i] *= H_real[i];
                if (H_comp != NULL)
                    data[i] *= H_comp[i];
            }

            // Do IFFT
            ifft(data);

            // Copy back to array
            for (int i = 0; i < params->numCols; i++)
                proj_line[i] = real(data[i]);
        }
        delete[] paddedArray;
    }
    if (H_real != NULL)
        delete[] H_real;
    if (H_comp != NULL)
        delete[] H_comp;

    return true;
}

float* rampFilterFrequencyResponseMagnitude_cpu(int N, parameters* params)
{
    float T = params->pixelWidth;
    bool isCurved = false;
    if (params->geometry == parameters::FAN || params->geometry == parameters::CONE || params->geometry == parameters::MODULAR)
    {
        T *= params->sod / params->sdd;
        if (params->detectorType == parameters::CURVED)
            isCurved = true;
    }

    double* h_d = rampImpulseResponse(N, T, params);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = h_d[i];

        if (i != 0 && isCurved == true)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= temp * temp;
        }
    }
    delete[] h_d;

    Complex* test = new Complex[N];
    for (int i = 0; i < N; i++)
        test[i] = h[i];
    CArray data(test, N);

    // forward fft
    fft(data);

    float theExponent = 1.0;
    if (params->FBPlowpass >= 2.0)
    {
        theExponent = 1.0 / (1.0 - log2(1.0 + cos(PI / params->FBPlowpass)));
        //printf("theExponent = %f\n", theExponent);
    }

    float* H_real = new float[N];
    for (int i = 0; i < N; i++)
    {
        H_real[i] = real(data[i]) / float(N);
        if (params->FBPlowpass >= 2.0)
        {
            //float omega = float(i)*PI / N_over2;
            float omega = float(i) * PI / N;
            if (i > N / 2)
                omega = float(i - N) * PI / N;

            float theWeight = pow(std::max(float(0.0), float(cos(omega))), 2.0 * theExponent);

            H_real[i] *= theWeight;
            //printf("H(%f) = %f (%d)\n", omega, theWeight, i);
        }
    }

    // Clean up
    delete[] h;
    delete[] test;

    return H_real;
}

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

void fftshift(float* h, int N)
{
    for (int i = 0; i < N / 2; i++)
    {
        int i_conj = i + N/2;
        float temp = h[i];
        h[i] = h[i_conj];
        h[i_conj] = temp;
    }
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

double* rampImpulseResponse(int N, double T, parameters* params)
{
    double* h = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++)
    {
        if (params->muCoeff == 0.0)
            h[i] = rampImpulseResponse(N, T, i, params->rampID);
        else
            h[i] = rampImpulseResponse(N, T, i, 2) - rampImpulseResponse_bandLimited(N, T, i, fabs(params->muCoeff) * params->pixelWidth / (2.0*PI));
    }
    return h;
}

double rampImpulseResponse_bandLimited(int N, double T, int i, float mu)
{
    double retVal;
    double s = timeSamples(i, N);
    if (s != 0.0)
        retVal = (2.0 * PI * s * mu * sin(2.0 * PI * mu * s) + cos(2.0 * PI * mu * s) - 1.0) / (PI * s * s);
    else
        retVal = 2.0 * PI * mu * mu;

    retVal = retVal / T;
    return retVal;
}

double* HilbertTransformImpulseResponse(int N, int whichDirection/* = 1*/)
{
    double shifter;
    if (whichDirection < 0)
        shifter = -0.5;
    else if (whichDirection > 0)
        shifter = 0.5;
    else
        shifter = 0.0;
    double* h = (double*)malloc(sizeof(double) * N);
    double s;
    for (int i = 0; i < N; i++)
    {
        s = timeSamples(i, N) + shifter;
        if (shifter != 0.0)
            h[i] = 1.0 / (PI * s);
        else
        {
            if (i % 2 == 0)
                h[i] = 0.0;
            else
                h[i] = 2.0 / (PI * s);
        }
    }

    return h;
}

Complex* HilbertTransformFrequencyResponse_cpu(int N, parameters* params)
{
    float T = params->pixelWidth * params->sod / params->sdd;

    //double* h_d = HilbertTransformImpulseResponse(N);
    double* h_d = HilbertTransformImpulseResponse(N, 0);
    float* h = new float[N];
    for (int i = 0; i < N; i++)
    {
        h[i] = h_d[i];
        if (i != 0 && params->geometry == parameters::CONE && params->detectorType == parameters::CURVED)
        {
            double s = timeSamples(i, N) * T / params->sod;
            double temp = s / sin(s);
            h[i] *= temp;// *temp;
        }
    }
    delete[] h_d;

    Complex* test = new Complex[N];
    for (int i = 0; i < N; i++)
        test[i] = h[i];
    CArray data(test, N);

    // forward fft
    fft(data);

    Complex* H_comp = new Complex[N];
    for (int i = 0; i < N; i++)
    {
        H_comp[i] = data[i] / Complex(N);
    }

    // Clean up
    delete[] h;
    delete[] test;

    return H_comp;
}

bool splitLeftAndRight(float* g, float* g_left, float* g_right, parameters* params)
{
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < params->numRows; i++)
    {
        float* g_line = &g[i * params->numCols];
        float* g_left_line = &g_left[i * params->numCols];
        float* g_right_line = &g_right[i * params->numCols];
        for (int j = 0; j < params->numCols; j++)
        {
            float val = g_line[j];

            float s = j * params->pixelWidth + params->u_0();
            float s_conj = - s;
            float s_conj_ind = (s_conj - params->u_0()) / params->pixelWidth;
            float val_conj = 0.0;
            if (0.0 <= s_conj_ind && s_conj_ind <= float(params->numCols - 1))
            {
                int s_lo = int(s_conj_ind);
                int s_hi = std::min(s_lo + 1, params->numCols - 1);
                float ds = s_conj_ind - float(s_lo);
                val_conj = (1.0 - ds) * g_line[s_lo] + ds * g_line[s_hi];
            }

            if (s > 0.0)
            {
                g_right_line[j] = val;
                g_left_line[j] = val_conj;
            }
            else if (s < 0.0)
            {
                g_right_line[j] = val_conj;
                g_left_line[j] = val;
            }
            else
            {
                g_left_line[j] = val;
                g_right_line[j] = val;
            }
        }
    }
    return true;
}

bool mergeLeftAndRight(float* g, float* g_left, float* g_right, parameters* params)
{
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < params->numRows; i++)
    {
        float* g_line = &g[i * params->numCols];
        float* g_left_line = &g_left[i * params->numCols];
        float* g_right_line = &g_right[i * params->numCols];
        for (int j = 0; j < params->numCols; j++)
        {
            float s = j * params->pixelWidth + params->u_0();
            if (s >= 0.0)
                g_line[j] = g_right_line[j];
            else
                g_line[j] = g_left_line[j];
        }
    }
    return true;
}

bool Laplacian_cpu(float*& g, int numDims, bool smooth, parameters* params, float scalar)
{
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < params->numAngles; i++)
    {
        float* tempProj = (float*)malloc(sizeof(float) * params->numRows * params->numCols);
        float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];
        for (int j = 0; j < params->numRows; j++)
        {
            // Copy data to new array
            float* aLine = &aProj[j * params->numCols];
            for (int k = 0; k < params->numCols; k++)
                tempProj[j * params->numCols + k] = aLine[k];
        }

        for (int j = 0; j < params->numRows; j++)
        {
            int j_minus, j_plus;
            if (smooth)
            {
                j_minus = std::max(j - 2, 0);
                j_plus = std::min(j + 2, params->numRows - 1);
            }
            else
            {
                j_minus = std::max(j - 1, 0);
                j_plus = std::min(j + 1, params->numRows - 1);
            }
            float* aLine = &aProj[j * params->numCols];
            for (int k = 0; k < params->numCols; k++)
            {
                if (smooth)
                {
                    int k_minus = std::max(k - 2, 0);
                    int k_plus = std::min(k + 2, params->numCols - 1);

                    float diff = 0.25 * (tempProj[j * params->numCols + k_plus] + tempProj[j * params->numCols + k_minus]) - 0.5 * tempProj[j * params->numCols + k];
                    if (numDims >= 2)
                        diff += 0.25 * (tempProj[j_plus * params->numCols + k] + tempProj[j_minus * params->numCols + k]) - 0.5 * tempProj[j * params->numCols + k];
                    aLine[k] = diff * scalar;
                }
                else
                {
                    int k_minus = std::max(k - 1, 0);
                    int k_plus = std::min(k + 1, params->numCols - 1);

                    float diff = tempProj[j * params->numCols + k_plus] + tempProj[j * params->numCols + k_minus] - 2.0 * tempProj[j * params->numCols + k];
                    if (numDims >= 2)
                        diff += tempProj[j_plus * params->numCols + k] + tempProj[j_minus * params->numCols + k] - 2.0 * tempProj[j * params->numCols + k];
                    aLine[k] = diff * scalar;
                }
            }
        }
        free(tempProj);
    }
    return true;
}

bool ray_derivative_cpu(float* g, parameters* params, float sampleShift, float scalar)
{
    float c = scalar / params->pixelWidth;
    if (sampleShift == 0.0)
        c *= 0.5;
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int i = 0; i < params->numAngles; i++)
    {
        float* temp = (float*)malloc(sizeof(float) * params->numCols);
        float* aProj = &g[uint64(i) * uint64(params->numRows * params->numCols)];
        for (int j = 0; j < params->numRows; j++)
        {
            float* aLine = &aProj[j * params->numCols];
            for (int k = 0; k < params->numCols; k++)
                temp[k] = aLine[k];

            if (sampleShift == 0.0)
            {
                aLine[0] = (temp[1] - temp[0]) * c;
                aLine[params->numCols - 1] = (temp[params->numCols - 1] - temp[params->numCols - 2]) * c;
                for (int k = 1; k < params->numCols - 1; k++)
                    aLine[k] = (temp[k + 1] - temp[k - 1]) * c;
            }
            else if (sampleShift > 0.0)
            {
                aLine[params->numCols - 1] = 0.0;
                for (int k = 0; k < params->numCols - 1; k++)
                    aLine[k] = (temp[k + 1] - temp[k]) * c;
            }
            else
            {
                aLine[0] = 0.0;
                for (int k = 1; k < params->numCols; k++)
                    aLine[k] = (temp[k] - temp[k - 1]) * c;
            }
        }
        free(temp);
    }
    return true;
}

int zeroPadForOffsetScan_numberOfColsToAdd(parameters* params)
{
    bool padOnLeft;
    return zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);
}

int zeroPadForOffsetScan_numberOfColsToAdd(parameters* params, bool& padOnLeft)
{
    if (params == NULL)
        return 0;
    else if (params->helicalPitch != 0.0 || params->offsetScan == false || params->angularRange < 360.0 - fabs(params->T_phi()) * 180.0 / PI)
        return 0;

    if (params->geometry == parameters::MODULAR && params->modularbeamIsAxiallyAligned() == false)
        return 0;

    int N_add = 0;
    float abs_minVal = 0.0;
    float abs_maxVal = 0.0;
    if (params->geometry == parameters::CONE || params->geometry == parameters::FAN || params->geometry == parameters::MODULAR)
    {
        bool normalizeConeAndFanCoordinateFunctions_save = params->normalizeConeAndFanCoordinateFunctions;
        params->normalizeConeAndFanCoordinateFunctions = true;

        float alpha_min = params->u(0);
        float alpha_max = params->u(params->numCols - 1);

        if (params->detectorType == parameters::FLAT)
        {
            alpha_min = atan(alpha_min);
            alpha_max = atan(alpha_max);
        }
        abs_minVal = fabs(params->sod * sin(alpha_min) - params->tau * cos(alpha_min));
        abs_maxVal = fabs(params->sod * sin(alpha_max) - params->tau * cos(alpha_max));

        float delta = std::min(abs_minVal, abs_maxVal);

        if (abs_minVal < abs_maxVal)
        {
            // zero pad on left
            float alpha_0 = asin(-params->rFOV() / params->sod) + asin(params->tau / params->sod);
            if (params->detectorType == parameters::FLAT)
                N_add = int(ceil(fabs(params->u_inv(tan(alpha_0)))));
            else
                N_add = int(ceil(fabs(params->u_inv(alpha_0))));
        }
        else
        {
            // zero pad on right
            float alpha_end = asin(params->rFOV() / params->sod) + asin(params->tau / params->sod);
            //float lateral_end;
            if (params->detectorType == parameters::FLAT)
                N_add = int(ceil(params->u_inv(tan(alpha_end)))) - params->numCols;
            else
                N_add = int(ceil(params->u_inv(alpha_end))) - params->numCols;
        }

        params->normalizeConeAndFanCoordinateFunctions = normalizeConeAndFanCoordinateFunctions_save;
    }
    else //if (params->geometry == parameters::PARALLEL)
    {
        abs_minVal = fabs(params->u(0));
        abs_maxVal = fabs(params->u(params->numCols - 1));

        double delta = std::min(abs_minVal, abs_maxVal);

        if (abs_minVal < abs_maxVal)
            N_add = int(ceil(fabs(params->u_inv(-params->rFOV()))));
        else
            N_add = int(ceil(params->u_inv(params->rFOV()))) - params->numCols;
    }
    N_add = std::max(0, N_add);
    if (abs_minVal < abs_maxVal)
        padOnLeft = true;
    else
        padOnLeft = false;
    return N_add;
}

float* zeroPadForOffsetScan(float* g, parameters* params, float* g_out)
{
    if (g == NULL || params == NULL)
        return NULL;
    else if (params->helicalPitch != 0.0 || params->offsetScan == false)
        return NULL;
    if (params->geometry == parameters::MODULAR && params->modularbeamIsAxiallyAligned() == false)
        return NULL;

    bool padOnLeft;
    int N_add = zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);

    float* offsetScanWeights = setOffsetScanWeights(params);
    if (N_add > 0 && offsetScanWeights != NULL)
    {
        float* g_pad = NULL;
        if (g_out == NULL)
            g_pad = (float*)calloc(size_t(uint64(params->numAngles) * uint64(params->numRows) * uint64(params->numCols + N_add)), sizeof(float));
        else
            g_pad = g_out;
        if (padOnLeft)
        {
            // zero pad on the left
            omp_set_num_threads(omp_get_num_procs());
            #pragma omp parallel for
            for (int i = 0; i < params->numAngles; i++)
            {
                float* aProj = &g[uint64(i) * uint64(params->numRows) * uint64(params->numCols)];
                float* aProj_pad = &g_pad[uint64(i) * uint64(params->numRows) * uint64(params->numCols+ N_add)];
                for (int j = 0; j < params->numRows; j++)
                {
                    float* aLine = &aProj[j*params->numCols];
                    float* aLine_pad = &aProj_pad[j * (params->numCols+N_add)];
                    for (int k = 0; k < params->numCols; k++)
                        aLine_pad[k + N_add] = aLine[k] * 2.0 * offsetScanWeights[j * params->numCols + k];
                    if (g_out != NULL)
                    {
                        for (int k = 0; k < N_add; k++)
                            aLine_pad[k] = 0.0;
                    }
                }
            }
            params->centerCol += N_add;
        }
        else
        {
            // zero pad on the right
            omp_set_num_threads(omp_get_num_procs());
            #pragma omp parallel for
            for (int i = 0; i < params->numAngles; i++)
            {
                float* aProj = &g[uint64(i) * uint64(params->numRows) * uint64(params->numCols)];
                float* aProj_pad = &g_pad[uint64(i) * uint64(params->numRows) * uint64(params->numCols + N_add)];
                for (int j = 0; j < params->numRows; j++)
                {
                    float* aLine = &aProj[j * params->numCols];
                    float* aLine_pad = &aProj_pad[j * (params->numCols + N_add)];
                    for (int k = 0; k < params->numCols; k++)
                        aLine_pad[k] = aLine[k] *2.0 * offsetScanWeights[j * params->numCols + k];
                    if (g_out != NULL)
                    {
                        for (int k = 0; k < N_add; k++)
                            aLine_pad[params->numCols+k] = 0.0;
                    }
                }
            }
        }
        free(offsetScanWeights);
        params->numCols += N_add;
        return g_pad;
    }
    else
    {
        if (offsetScanWeights != NULL)
            free(offsetScanWeights);
        return NULL;
    }
}
