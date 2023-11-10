#include "ramp_filter_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include <complex>
#include <iostream>
#include <valarray>
#include <omp.h>

#ifndef PI
#define PI 3.141592653589793
#endif

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// source: https://tfetimes.com/c-fast-fourier-transform/
// Cooley–Tukey FFT (in-place)
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
    int N_H = int(pow(2.0, ceil(log2(2 * params->numCols))));
    //int N_H_over2 = N_H / 2 + 1;
    float* H_real = rampFilterFrequencyResponseMagnitude_cpu(N_H, params);
    if (scalar != 1.0)
    {
        for (int i = 0; i < N_H; i++)
            H_real[i] *= scalar;
    }

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int iphi = 0; iphi < params->numAngles; iphi++)
    {
        Complex* paddedArray = new Complex[N_H];

        for (int iRow = 0; iRow < params->numRows; iRow++)
        {
            float* proj_line = &g[iphi * params->numRows * params->numCols + iRow * params->numCols];

            // Put in padded array
            for (int i = 0; i < N_H; i++)
            {
                if (i < params->numCols)
                    paddedArray[i] = proj_line[i];
                else
                    paddedArray[i] = 0.0;
            }
            CArray data(paddedArray, N_H);

            // Do FFT
            fft(data);

            // Multiply by filter
            for (int i = 0; i < N_H; i++)
                data[i] *= H_real[i];

            // Do IFFT
            ifft(data);

            // Copy back to array
            for (int i = 0; i < params->numCols; i++)
                proj_line[i] = real(data[i]);
        }
        delete[] paddedArray;
    }
    delete[] H_real;

    return true;
}

float* rampFilterFrequencyResponseMagnitude_cpu(int N, parameters* params)
{
    float T = params->pixelWidth;
    bool isCurved = false;
    if (params->geometry == parameters::FAN || params->geometry == parameters::CONE)
    {
        T *= params->sod / params->sdd;
        if (params->detectorType == parameters::CURVED)
            isCurved = true;
    }

    int rampID = 2;

    double* h_d = rampImpulseResponse(N, T, rampID);
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

    float* H_real = new float[N];
    for (int i = 0; i < N; i++)
    {
        H_real[i] = real(data[i]) / float(N);
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

double* rampImpulseResponse(int N, double T, int rampID)
{
    double* h = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++)
        h[i] = rampImpulseResponse(N, T, i, rampID);
    return h;
}
