////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for statistics operations
////////////////////////////////////////////////////////////////////////////////

#include "statistics.h"
#include "cpu_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstdint>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <numeric>

using namespace std;

float sum(float* I, int N_1, int N_2, int N_3)
{
    return sum(I, uint64(N_1)*uint64(N_2)*uint64(N_3));
}

float maximum(float* I, int N_1, int N_2, int N_3)
{
    return maximum(I, uint64(N_1)*uint64(N_2)*uint64(N_3));
}

float minimum(float* I, int N_1, int N_2, int N_3)
{
    return minimum(I, uint64(N_1)*uint64(N_2)*uint64(N_3));
}

bool range(float* I, int N_1, int N_2, int N_3, float& minVal, float& maxVal)
{
    return range(I, uint64(N_1)*uint64(N_2)*uint64(N_3), minVal, maxVal);
}

bool basicStats(float* I, int N_1, int N_2, int N_3, float* stats)
{
    return basicStats(I, uint64(N_1)*uint64(N_2)*uint64(N_3), stats);
}

float sum(float* I, uint64 N)
{
    if (I == NULL || N <= 0)
        return 0.0;
    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);
    double* sums = (double*)calloc(size_t(max_threads), sizeof(double));
    uint64 max_chunk_size = uint64(ceil(double(N) / double(max_threads)));

    omp_set_num_threads(max_threads);
	#pragma omp parallel for
    for (int i = 0; i < max_threads; i++)
    {
        uint64 ind_offset = uint64(i) * max_chunk_size;
        double sum = 0.0;

        for (uint64 j = 0; j < max_chunk_size; j++)
        {
            uint64 ind = ind_offset + j;
            if (ind < N)
            {
                sum += I[ind];
            }
        }
        sums[i] = sum;
    }

    double retVal = 0.0;
    for (int i = 0; i < max_threads; i++)
        retVal += sums[i];

    free(sums);
    return float(retVal);
}

float maximum(float* I, uint64 N)
{
    if (I == NULL || N <= 0)
        return 0.0;
    /*
    float retVal = I[0];
    for (uint64 i = 1; i < N; i++)
        retVal = max(retVal, I[i]);
    return retVal;
    //*/

    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);
    float* maxes = (float*)calloc(size_t(max_threads), sizeof(float));
    uint64 max_chunk_size = uint64(ceil(double(N) / double(max_threads)));

    omp_set_num_threads(max_threads);
	#pragma omp parallel for
    for (int i = 0; i < max_threads; i++)
    {
        uint64 ind_offset = uint64(i) * max_chunk_size;
        float cur_max = I[ind_offset];

        for (uint64 j = 1; j < max_chunk_size; j++)
        {
            uint64 ind = ind_offset + j;
            if (ind < N)
            {
                cur_max = max(cur_max, I[ind]);
            }
        }
        maxes[i] = cur_max;
    }

    float retVal = maxes[0];
    for (int i = 1; i < max_threads; i++)
        retVal = max(retVal, maxes[i]);

    free(maxes);
    return retVal;
}

float minimum(float* I, uint64 N)
{
    if (I == NULL || N <= 0)
        return 0.0;
    /*
    float retVal = I[0];
    for (uint64 i = 1; i < N; i++)
        retVal = min(retVal, I[i]);
    return retVal;
    //*/

    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);
    float* mins = (float*)calloc(size_t(max_threads), sizeof(float));
    uint64 max_chunk_size = uint64(ceil(double(N) / double(max_threads)));

    omp_set_num_threads(max_threads);
	#pragma omp parallel for
    for (int i = 0; i < max_threads; i++)
    {
        uint64 ind_offset = uint64(i) * max_chunk_size;
        float cur_min = I[ind_offset];

        for (uint64 j = 1; j < max_chunk_size; j++)
        {
            uint64 ind = ind_offset + j;
            if (ind < N)
            {
                cur_min = min(cur_min, I[ind]);
            }
        }
        mins[i] = cur_min;
    }

    float retVal = mins[0];
    for (int i = 1; i < max_threads; i++)
        retVal = min(retVal, mins[i]);

    free(mins);
    return retVal;
}

bool range(float* I, uint64 N, float& minVal, float& maxVal)
{
    if (I == NULL || N <= 0)
        return false;
    
    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);
    float* maxs = (float*)calloc(size_t(max_threads), sizeof(float));
    float* mins = (float*)calloc(size_t(max_threads), sizeof(float));
    uint64 max_chunk_size = uint64(ceil(double(N) / double(max_threads)));

    omp_set_num_threads(max_threads);
    #pragma omp parallel for
    for (int i = 0; i < max_threads; i++)
    {
        uint64 ind_offset = uint64(i) * max_chunk_size;
        float cur_max = I[ind_offset];
        float cur_min = I[ind_offset];

        for (uint64 j = 1; j < max_chunk_size; j++)
        {
            uint64 ind = ind_offset + j;
            if (ind < N)
            {
                cur_max = max(cur_max, I[ind]);
                cur_min = min(cur_min, I[ind]);
            }
        }
        maxs[i] = cur_max;
        mins[i] = cur_min;
    }

    maxVal = maxs[0];
    minVal = mins[0];
    for (int i = 1; i < max_threads; i++)
    {
        maxVal = max(maxVal, maxs[i]);
        minVal = min(minVal, mins[i]);
    }

    free(maxs);
    free(mins);
    return true;
}

bool basicStats(float* I, uint64 N, float* stats)
{
    if (I == NULL || N <= 0 || stats == NULL)
        return false;
    
    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);
    double* stats_threads = (double*)calloc(size_t(4*max_threads), sizeof(double));

    for (int i = 0; i < max_threads; i++)
    {
        stats_threads[4*i+0] = I[0];
        stats_threads[4*i+1] = I[0];
    }

    omp_set_num_threads(max_threads);
    #pragma omp parallel for
    for (long long i = 0; i < (long long)N; i++)
    {
        int ithread = omp_get_thread_num();
        double curVal = I[i];
        stats_threads[4*ithread + 0] = min(curVal, stats_threads[4*ithread + 0]);
        stats_threads[4*ithread + 1] = max(curVal, stats_threads[4*ithread + 1]);
        stats_threads[4*ithread + 2] += curVal;
        stats_threads[4*ithread + 3] += curVal * curVal;
    }

    stats[0] = stats_threads[0];
    stats[1] = stats_threads[1];
    stats[2] = 0.0;
    stats[3] = 0.0;
    for (int i = 0; i < max_threads; i++)
    {
        stats[0] = min(stats[0], float(stats_threads[4*i+0]));
        stats[1] = max(stats[1], float(stats_threads[4*i+1]));
        stats[2] += stats_threads[4*i+2];
        stats[3] += stats_threads[4*i+3];
    }

    float mean = stats[2] / float(N);
    float stdev = sqrt((stats[3] - stats[2]*stats[2] / float(N)) / float(N - 1));
    stats[2] = mean;
    stats[3] = stdev;

    free(stats_threads);
    return true;
}

float percentile_1D(float* data, uint64 N, float q)
{
    if (data == NULL || N < 1 || q < 0.0 || q > 100.0)
        return 0.0;
 
    if (N == 1)
        return data[0];

    // NumPy's "linear" interpolation index in [0, N-1]
    float idx = (q / 100.0) * (static_cast<float>(N) - 1.0);
    std::size_t k_low  = static_cast<std::size_t>(std::floor(idx));
    std::size_t k_high = static_cast<std::size_t>(std::ceil(idx));

    // First order statistic: k_low
    std::nth_element(data, data + k_low, data + N);
    float v_low = data[k_low];

    if (k_low == k_high)
        return v_low;

    // Second order statistic: k_high
    // It's fine to call nth_element again on the already-partitioned data;
    // it still guarantees the correct element at position k_high.
    std::nth_element(data, data + k_high, data + N);
    float v_high = data[k_high];

    float weight = idx - static_cast<float>(k_low);
    return (1.0 - weight) * v_low + weight * v_high;
}

bool percentile_2D(float* I, int N_images, int N, float q, float* percentiles)
{
    if (I == NULL || N_images < 1 || N < 1 || percentiles == NULL)
        return false;

    int num_threads = num_cpu_threads();

    float** images = (float**) malloc(sizeof(float*)*num_threads);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++)
    {
        int ind = omp_get_thread_num();
        images[ind] = calloc_aligned(sizeof(float)*size_t(N));
    }

    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < N_images; i++)
    {
        float* data = &I[uint64(i)*uint64(N)];
        float* data_copy = images[omp_get_thread_num()];
        memcpy(data_copy, data, sizeof(float)*size_t(N));
        percentiles[i] = percentile_1D(data_copy, uint64(N), q);
    }

    // clean up
    if (images != NULL)
    {
        for (int i = 0; i < num_threads; i++)
            free_aligned(images[i]);
        free(images);
    }

    return true;
}

template <typename T>
float* histogram(const T* I, int N_1, int N_2, int N_3, int& numBins, float& binSize, float* h, float* bins, bool include_zeros, float rangeMin, float rangeMax)
{
    return histogram(I, uint64(N_1) * uint64(N_2) * uint64(N_3), numBins, binSize, h, bins, include_zeros, rangeMin, rangeMax);
}

template <typename T>
float* histogram(const T* I, uint64 N, int& numBins, float& binSize, float* h, float* bins, bool include_zeros, float rangeMin, float rangeMax)
{
    // When rangeMax > rangeMin the histogram bins span the user-provided
    // [rangeMin, rangeMax] interval and values outside this interval are ignored.
    // Otherwise the bin range is derived from the data min/max (default behavior).
    const bool useCustomRange = (rangeMax > rangeMin);
    static_assert(std::is_arithmetic_v<T>, "histogram<T>: T must be an arithmetic type");

    if (I == NULL || N == 0)
    {
        binSize = 0.0f;
        if (h == NULL && numBins > 0)
            h = new float[numBins]();
        else if (h != NULL && numBins > 0)
            std::fill(h, h + numBins, 0.0f);
        if (bins != NULL && numBins > 0)
            std::fill(bins, bins + numBins, 0.0f);
        return h;
    }

    if (numBins <= 0)
        numBins = 256;

    int max_threads = num_cpu_threads();
    if (max_threads > N)
        max_threads = int(N);

    float* h_stack = (float*)calloc(size_t(numBins * max_threads), sizeof(float));

    // Compute min/max in float space (avoids needing minimum/maximum<T>).
    float x_min = std::numeric_limits<float>::infinity();
    float x_max = -std::numeric_limits<float>::infinity();

    uint64 max_chunk_size = uint64(ceil(double(N) / double(max_threads)));

    if (useCustomRange)
    {
        // The bin range is fixed by the caller; no need to scan the data.
        x_min = rangeMin;
        x_max = rangeMax;
    }
    else
    {
        std::vector<float> mins(size_t(max_threads), std::numeric_limits<float>::infinity());
        std::vector<float> maxes(size_t(max_threads), -std::numeric_limits<float>::infinity());

        omp_set_num_threads(max_threads);
        #pragma omp parallel for
        for (int i = 0; i < max_threads; i++)
        {
            uint64 ind_offset = uint64(i) * max_chunk_size;

            float cur_min = std::numeric_limits<float>::infinity();
            float cur_max = -std::numeric_limits<float>::infinity();

            for (uint64 j = 0; j < max_chunk_size; j++)
            {
                uint64 ind = ind_offset + j;
                if (ind < N)
                {
                    float v = static_cast<float>(I[ind]);
                    cur_min = std::min(cur_min, v);
                    cur_max = std::max(cur_max, v);
                }
            }

            mins[size_t(i)] = cur_min;
            maxes[size_t(i)] = cur_max;
        }

        for (int i = 0; i < max_threads; i++)
        {
            x_min = std::min(x_min, mins[size_t(i)]);
            x_max = std::max(x_max, maxes[size_t(i)]);
        }
    }

	// Bin width (avoid naming it 'T' which would shadow the template parameter).
	float binWidth = 0.0f;
	if (numBins > 1)
		binWidth = (x_max - x_min) / float(numBins - 1);

	binSize = binWidth;

    if (bins != NULL)
    {
		for (int i = 0; i < numBins; i++)
			bins[i] = x_min + binWidth * float(i);
    }

    // If all values are identical, put everything in the first bin.
	if (binWidth == 0.0f)
    {
        if (h == NULL)
            h = new float[numBins]();
        else
            std::fill(h, h + numBins, 0.0f);

        h[0] = float(N);

        free(h_stack);
        return h;
    }

    omp_set_num_threads(max_threads);
    #pragma omp parallel for
    for (int i = 0; i < max_threads; i++)
    {
        uint64 ind_offset = uint64(i) * max_chunk_size;
        float* h_local = &h_stack[i * numBins];

        for (uint64 j = 0; j < max_chunk_size; j++)
        {
            uint64 ind = ind_offset + j;
            if (ind < N)
            {
                const T raw = I[ind];
                if (!include_zeros && raw == T(0))
                    continue;

					float curVal = static_cast<float>(raw);

					// With a user-specified range, ignore values that fall outside [x_min, x_max].
					if (useCustomRange && (curVal < x_min || curVal > x_max))
						continue;

					int bin = std::max(0, std::min(numBins - 1, int(0.5f + (curVal - x_min) / binWidth)));
                h_local[bin] += 1.0f;
            }
        }
    }

    if (h == NULL)
        h = new float[numBins];

    for (int i = 0; i < max_threads; i++)
    {
        for (int j = 0; j < numBins; j++)
        {
            if (i == 0)
                h[j] = h_stack[i * numBins + j];
            else
                h[j] += h_stack[i * numBins + j];
        }
    }
    free(h_stack);

    return h;
}

// Explicit instantiations (add more here as needed)
template float* histogram<float>(const float*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<float>(const float*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<double>(const double*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<double>(const double*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<std::uint8_t>(const std::uint8_t*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<std::uint8_t>(const std::uint8_t*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<std::int16_t>(const std::int16_t*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<std::int16_t>(const std::int16_t*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<std::uint16_t>(const std::uint16_t*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<std::uint16_t>(const std::uint16_t*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<std::int32_t>(const std::int32_t*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<std::int32_t>(const std::int32_t*, uint64, int&, float&, float*, float*, bool, float, float);

template float* histogram<std::uint32_t>(const std::uint32_t*, int, int, int, int&, float&, float*, float*, bool, float, float);
template float* histogram<std::uint32_t>(const std::uint32_t*, uint64, int&, float&, float*, float*, bool, float, float);

std::vector<float> initKMeansPlusPlus1DFromHistogram(const std::vector<float>& bin_centers, const std::vector<float>& counts, int k)
{
    if (k <= 0) throw std::invalid_argument("k must be positive.");
    if (bin_centers.size() != counts.size())
        throw std::invalid_argument("bin_centers and counts must have same length.");
    const int m = static_cast<int>(bin_centers.size());
    if (m == 0) throw std::invalid_argument("Histogram is empty.");

    unsigned seed = 42;

    // total weight must be > 0
    float total_w = 0.0;
    for (float c : counts) {
        if (c < 0.0) throw std::invalid_argument("counts must be non-negative.");
        total_w += c;
    }
    if (total_w == 0.0) throw std::invalid_argument("All counts are zero.");

    k = std::min(k, m);

    std::mt19937 rng(seed);

    std::vector<float> centers;
    centers.reserve(k);

    // 1) First center: sample a bin index proportional to counts
    {
        std::discrete_distribution<int> first_dist(counts.begin(), counts.end());
        int idx = first_dist(rng);
        centers.push_back(bin_centers[idx]);
    }

    // 2) Remaining centers: sample proportional to counts * D(x)^2
    std::vector<float> weights(m, 0.0); // per-bin selection weight
    for (int c = 1; c < k; ++c) {
        // Update weights = count * squared distance to nearest chosen center
        float sumw = 0.0;
        for (int i = 0; i < m; ++i) {
            if (counts[i] == 0.0) {
                weights[i] = 0.0;
                continue;
            }
            float x = bin_centers[i];
            float best = std::numeric_limits<float>::infinity();
            for (float ctr : centers) {
                float d = x - ctr;
                float d2 = d * d;
                if (d2 < best) best = d2;
            }
            weights[i] = counts[i] * best;
            sumw += weights[i];
        }

        int next_idx;
        if (sumw == 0.0) {
            // All occupied bins coincide with current centers (or only one occupied bin).
            // Fall back to sampling by counts again.
            std::discrete_distribution<int> fallback(counts.begin(), counts.end());
            next_idx = fallback(rng);
        } else {
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            next_idx = dist(rng);
        }

        centers.push_back(bin_centers[next_idx]);
    }

    return centers;
}

bool kmeans(float* I, int N_1, int N_2, int N_3, float* means, int K)
{
    if (I == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || means == NULL || K < 1)
        return false;
    uint64 N = uint64(N_1) * uint64(N_2) * uint64(N_3);
    if (K == 1)
    {
        means[0] = sum(I, N) / float(N);
        return true;
    }

    int numBins = int(pow(2.0, 20.0));
    float T;

    float* bins = new float[numBins];
    float* hist = histogram(I, N_1, N_2, N_3, numBins, T, NULL, bins, false);

    float x_min = bins[0];
    //float total_sum = sum(hist, uint64(numBins))-hist[0];
    float total_sum = sum(hist, uint64(numBins));
    float total_sum_over_numClasses = total_sum / float(K+1);

    /*
    printf("volume size: %d, %d, %d\n", N_1, N_2, N_3);
    printf("total_sum = %f, total_sum_over_numClasses = %f\n", total_sum, total_sum_over_numClasses);
    printf("max bin: %f\n", (numBins-1)*T);
    printf("binSize = %e, numBins = %d\n", T, numBins);
    //*/

    /* Initialize means by dividing histogram into equal pieces
    for (int k = 0; k < K; k++)
        means[k] = 0.0;
    double cur_sum = 0.0;
    means[0] = bins[0];
    int k_0 = 1;
    for (int bin = 0; bin < numBins; bin++)
    {
        cur_sum += hist[bin];
        if (cur_sum > total_sum_over_numClasses)
        {
            //printf("cur_sum[%d] = %f (bin = %d)\n", k_0, cur_sum, bin);
            means[k_0] = bins[bin];
            cur_sum = 0.0;
            k_0 += 1;
            if (k_0 >= K)
                break;
        }
    }
    if (means[K-1] == 0.0)
        means[K-1] = bins[numBins-1];
    //*/

    /* Initial means by dividing the histogram equally
    for (int k = 0; k < K; k++)
        means[k] = bins[int(((float(k)+0.5)/float(K))*numBins)];
    //*/

    std::vector<float> bin_centers(bins, bins + numBins);
    std::vector<float> hist_counts(hist, hist + numBins);
    std::vector<float> means_vec = initKMeansPlusPlus1DFromHistogram(bin_centers, hist_counts, K);
    std::sort(means_vec.begin(), means_vec.end());
    for (int k = 0; k < K; k++)
        means[k] = means_vec[k];

    /*
    printf("initial guesses: ");
    for (int k = 0; k < K; k++)
        printf("%f ", means[k]);
    printf("\n");
    //*/

    float* new_means = new float[K];
    float* counts = new float[K];

    int numIter = 50;
    for (int n = 0; n < numIter; n++)
    {
        for (int k = 0; k < K; k++)
        {
            new_means[k] = 0.0;
            counts[k] = 0.0;
        }
        for (int bin = 0; bin < numBins; bin++)
        {
            if (hist[bin] > 0.0)
            {
                float x = x_min + T*bin;
                float minDist = fabs(x - means[0]);
                int ind_min = 0;
                for (int k = 1; k < K; k++)
                {
                    if (fabs(x-means[k]) < minDist)
                    {
                        minDist = fabs(x-means[k]);
                        ind_min = k;
                    }
                }
                counts[ind_min] += hist[bin];
                new_means[ind_min] += x*hist[bin];
            }
        }
        for (int k = 0; k < K; k++)
        {
            if (counts[k] > 0.0)
                means[k] = new_means[k] / counts[k];
        }

        /*
        printf("iteration %d guesses: ", n);
        for (int k = 0; k < K; k++)
            printf("%f ", means[k]);
        printf("\n");
        //*/
    }

    delete [] new_means;
    delete [] counts;
    delete [] hist;
    return true;
}

bool Otsu(float* I, int N_1, int N_2, int N_3, float* thresholds, int K)
{
    if (I == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || thresholds == NULL || K < 1 || K > 4)
        return false;
    uint64 N = uint64(N_1) * uint64(N_2) * uint64(N_3);

    int numBins = int(pow(2.0, 10.0));
    float T;

    float* bins = new float[numBins];
    float* hist = histogram(I, N_1, N_2, N_3, numBins, T, NULL, bins);

    // Normalize histogram
    float accum = sum(hist, numBins);
    //printf("accum = %f\n", accum);
    for (int i = 0; i < numBins; i++)
        hist[i] = hist[i] / accum;

    // Set up lookup tables
    float* P = (float*) calloc(numBins*numBins, sizeof(float));
    float* S = (float*) calloc(numBins*numBins, sizeof(float));
    float* H = (float*) calloc(numBins*numBins, sizeof(float));

    // diagonal
    for (int i=1; i < numBins; ++i)
    {
        P[i*numBins+i] = hist[i];
        S[i*numBins+i] = float(i)*hist[i];
    }
    // calculate first row (row 0 is all zero)
    for (int i=1; i < numBins-1; ++i)
    {
        P[1*numBins+i+1] = P[1*numBins+i] + hist[i+1];
        S[1*numBins+i+1] = S[1*numBins+i] + float(i+1)*hist[i+1];
    }
    // using row 1 to calculate others
    for (int i=2; i < numBins; i++)
    {
        for (int j=i+1; j < numBins; j++)
        {
            P[i*numBins+j] = P[1*numBins+j] - P[1*numBins+i-1];
            S[i*numBins+j] = S[1*numBins+j] - S[1*numBins+i-1];
        }
    }
    // now calculate H[i][j]
    for (int i=1; i < numBins; ++i)
    {
        for (int j=i+1; j < numBins; j++)
        {
            if (P[i*numBins+j] != 0.0)
                H[i*numBins+j] = (S[i*numBins+j]*S[i*numBins+j])/P[i*numBins+j];
            else
                H[i*numBins+j] = 0.0;
        }
    }

    vector<int> binEdges;
    for (int i = 0; i < K; i++)
        binEdges.push_back(0);
    
    float maxSig = 0.0;
    switch(K)
    {
    case 1:
        for (int i = 1; i < numBins-K; i++) // t1
        {
            float Sq = H[1*numBins+i] + H[(i+1)*numBins+numBins-1];
            if (maxSig < Sq)
            {
                binEdges[0] = i;
                maxSig = Sq;
            }
        }
        break;
    case 2:
        for (int i = 1; i < numBins-K; i++) // t1
        {
            for (int j = i+1; j < numBins-K +1; j++) // t2
            {
                float Sq = H[1*numBins+i] + H[(i+1)*numBins+j] + H[(j+1)*numBins+numBins-1];
                if (maxSig < Sq)
                {
                    binEdges[0] = i;
                    binEdges[1] = j;
                    maxSig = Sq;
                }
            }
        }
        break;
    case 3:
        for (int i = 1; i < numBins-K; i++) // t1
        {
            for (int j = i+1; j < numBins-K +1; j++) // t2
            {
                for (int k = j+1; k < numBins-K + 2; k++) // t3
                {
                    float Sq = H[1*numBins+i] + H[(i+1)*numBins+j] + H[(j+1)*numBins+k] + H[(k+1)*numBins+numBins-1];
                    if (maxSig < Sq)
                    {
                        binEdges[0] = i;
                        binEdges[1] = j;
                        binEdges[2] = k;
                        maxSig = Sq;
                    }
                }
            }
        }
        break;
    case 4:
        for (int i = 1; i < numBins-K; i++) // t1
        {
            for (int j = i+1; j < numBins-K +1; j++) // t2
            {
                for (int k = j+1; k < numBins-K + 2; k++) // t3
                {
                    for (int m = k+1; m < numBins-K + 3; m++) // t4
                    {
                        float Sq = H[1*numBins+i] + H[(i+1)*numBins+j] + H[(j+1)*numBins+k] + H[(k+1)*numBins+m] + H[(m+1)*numBins+numBins-1];
                        if (maxSig < Sq)
                        {
                            binEdges[0] = i;
                            binEdges[1] = j;
                            binEdges[2] = k;
                            binEdges[3] = m;
                            maxSig = Sq;
                        }
                    }
                }
            }
        }
        break;
    }

    for (int i = 0; i < K; i++)
    {
        //printf("%d -> %f\n", binEdges[i], bins[binEdges[i]]);
        thresholds[i] = bins[binEdges[i]];
    }

    // Clean Up
    free(P);
    free(S);
    free(H);
    delete [] bins;
    delete [] hist;
    return true;
}

bool calculate_centroid(float* I, int N_1, int N_2, int N_3, float threshold, float* centroid)
{
    if (I == NULL || N_1 < 1 || N_2 < 1 || N_3 < 1 || centroid == NULL)
        return false;

    int max_threads = min(N_1, num_cpu_threads());

    double* centroids = new double[N_1*3];
    double* slice_mass = new double[N_1];
    uint64 img_sz = uint64(N_2) * uint64(N_3);

    omp_set_num_threads(max_threads);
	#pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        double x = double(i);
        float* aSlice = &I[uint64(i)*img_sz];
        double centroid_local[3] = {0.0, 0.0, 0.0};
        double accum = 0.0;
        for (int j = 0; j < N_2; j++)
        {
            double y = double(j);
            for (int k = 0; k < N_3; k++)
            {
                double z = double(k);
                double curVal = aSlice[j*N_3 + k];
                if (isnan(threshold) == false)
                {
                    if (curVal > threshold)
                        curVal = 1.0;
                    else
                    {
                        curVal = 0.0;
                        continue;
                    }
                }
                accum += curVal;
                centroid_local[0] += x*curVal;
                centroid_local[1] += y*curVal;
                centroid_local[2] += z*curVal;
            }
        }

        slice_mass[i] = accum;
        centroids[3*i + 0] = centroid_local[0];
        centroids[3*i + 1] = centroid_local[1];
        centroids[3*i + 2] = centroid_local[2];
    }

    for (int i = 0; i < 3; i++)
        centroid[i] = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N_1; i++)
    {
        accum += slice_mass[i];
        centroid[0] += centroids[3*i + 0];
        centroid[1] += centroids[3*i + 1];
        centroid[2] += centroids[3*i + 2];
    }
    bool retVal = true;
    if (accum > 0.0)
    {
        for (int i = 0; i < 3; i++)
            centroid[i] = centroid[i] / accum;
    }
    else
    {
        retVal = false;
        for (int i = 0; i < 3; i++)
            centroid[i] = 0.0;
    }

    delete [] centroids;
    delete [] slice_mass;
    return retVal;
}

bool calculate_covariance(float* I, int N_1, int N_2, int N_3, float threshold, float* cov, float* centroid)
{
    if (I == NULL || N_1 < 1 || N_2 < 1 || N_3 < 1 || cov == NULL || centroid == NULL)
        return false;

    //if (isnan(threshold) == false)
    //    printf("using threshold = %f\n", threshold);

    // Calculate Centroid
    if (calculate_centroid(I, N_1, N_2, N_3, threshold, centroid) == false)
        return false;

    // Calculate Covariance Matrix
    int max_threads = min(N_1, num_cpu_threads());

    double* variances = new double[N_1*3];
    double* covariances = new double[N_1*3];
    double* slice_mass = new double[N_1];
    uint64 img_sz = uint64(N_2) * uint64(N_3);

    omp_set_num_threads(max_threads);
	#pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        double x = double(i) - centroid[0];
        float* aSlice = &I[uint64(i)*img_sz];
        double variance_local[3] = {0.0, 0.0, 0.0};
        double covariance_local[3] = {0.0, 0.0, 0.0};
        double accum = 0.0;
        for (int j = 0; j < N_2; j++)
        {
            double y = double(j) - centroid[1];
            for (int k = 0; k < N_3; k++)
            {
                double z = double(k) - centroid[2];
                double curVal = aSlice[j*N_3 + k];
                if (isnan(threshold) == false)
                {
                    if (curVal > threshold)
                        curVal = 1.0;
                    else
                    {
                        curVal = 0.0;
                        continue;
                    }
                }
                accum += curVal;
                variance_local[0] += (x*curVal) * (x*curVal);
                variance_local[1] += (y*curVal) * (y*curVal);
                variance_local[2] += (z*curVal) * (z*curVal);
                covariance_local[0] += (x*curVal) * (y*curVal);
                covariance_local[1] += (x*curVal) * (z*curVal);
                covariance_local[2] += (y*curVal) * (z*curVal);
            }
        }

        slice_mass[i] = accum;
        variances[3*i + 0] = variance_local[0];
        variances[3*i + 1] = variance_local[1];
        variances[3*i + 2] = variance_local[2];

        covariances[3*i + 0] = covariance_local[0];
        covariances[3*i + 1] = covariance_local[1];
        covariances[3*i + 2] = covariance_local[2];
    }

    for (int i = 0; i < 9; i++)
        cov[i] = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N_1; i++)
    {
        accum += slice_mass[i];
        cov[0*3 + 0] += variances[3*i + 0];
        cov[1*3 + 1] += variances[3*i + 1];
        cov[2*3 + 2] += variances[3*i + 2];

        cov[0*3 + 1] += covariances[3*i + 0];
        cov[0*3 + 2] += covariances[3*i + 1];
        cov[1*3 + 2] += covariances[3*i + 2];

        cov[1*3 + 0] += covariances[3*i + 0];
        cov[2*3 + 0] += covariances[3*i + 1];
        cov[2*3 + 1] += covariances[3*i + 2];
    }
    bool retVal = true;
    if (accum > 0.0)
    {
        for (int i = 0; i < 9; i++)
            cov[i] = cov[i] / accum;
    }
    else
    {
        retVal = false;
        for (int i = 0; i < 9; i++)
            cov[i] = 0.0;
    }

    // Clean up temporary memory
    delete [] variances;
    delete [] covariances;
    delete [] slice_mass;

    return retVal;
}

bool sum_first_dimension(float* I, int N_1, int N_2, int N_3, float* sums)
{
    return sum_dimension(I, N_1, N_2, N_3, sums, 0);
}

bool sum_dimension(float* I, int N_1, int N_2, int N_3, float* sums, int axis)
{
    if (I == NULL || N_1 < 1 || N_2 < 1 || N_3 < 1 || sums == NULL)
        return false;
    if (axis < 0 || axis > 2)
        return false;

    if (axis == 0)
    {
        // sum over the first dimension; output is N_2 x N_3
        uint64 out_sz = uint64(N_2)*uint64(N_3);

        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for
        for (long long jk = 0; jk < (long long)out_sz; jk++)
        {
            float cur_sum = 0.0;
            for (int i = 0; i < N_1; i++)
                cur_sum += I[uint64(i)*out_sz + jk];
            sums[jk] = cur_sum;
        }
    }
    else if (axis == 1)
    {
        // sum over the second dimension; output is N_1 x N_3
        uint64 proj_sz = uint64(N_2)*uint64(N_3);

        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for
        for (int i = 0; i < N_1; i++)
        {
            for (int k = 0; k < N_3; k++)
            {
                float cur_sum = 0.0;
                for (int j = 0; j < N_2; j++)
                    cur_sum += I[uint64(i)*proj_sz + uint64(j)*uint64(N_3) + k];
                sums[uint64(i)*uint64(N_3) + k] = cur_sum;
            }
        }
    }
    else // axis == 2
    {
        // sum over the third dimension; output is N_1 x N_2
        uint64 proj_sz = uint64(N_2)*uint64(N_3);

        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for
        for (int i = 0; i < N_1; i++)
        {
            for (int j = 0; j < N_2; j++)
            {
                float cur_sum = 0.0;
                uint64 row_off = uint64(i)*proj_sz + uint64(j)*uint64(N_3);
                for (int k = 0; k < N_3; k++)
                    cur_sum += I[row_off + k];
                sums[uint64(i)*uint64(N_2) + j] = cur_sum;
            }
        }
    }
    return true;
}
