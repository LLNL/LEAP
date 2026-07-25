////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ routines for some CPU-based computations
////////////////////////////////////////////////////////////////////////////////
#include <omp.h>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include "cpu_utils.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <unistd.h>
#endif

float getAvailableSystemMemory()
{
#if defined(__linux__)
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == NULL)
        return 0.0f;

    char line[256];
    long availableKb = -1;
    long freeSwapKb = -1;
    long hugeTotalPages = -1;
    long hugeFreePages = -1;
    long hugePageSize = -1;

    while (fgets(line, sizeof(line), meminfo))
    {
        long val;
        if (sscanf(line, "MemAvailable: %ld kB", &val) == 1)
            availableKb = val;
        else if (sscanf(line, "SwapFree: %ld kB", &val) == 1)
            freeSwapKb = val;
        else if (sscanf(line, "HugePages_Total: %ld", &val) == 1)
            hugeTotalPages = val;
        else if (sscanf(line, "HugePages_Free: %ld", &val) == 1)
            hugeFreePages = val;
        else if (sscanf(line, "Hugepagesize: %ld kB", &val) == 1)
            hugePageSize = val;

        if (availableKb != -1 && freeSwapKb != -1 &&
            hugeTotalPages != -1 && hugeFreePages != -1 && hugePageSize != -1)
            break;
    }
    fclose(meminfo);

    if (hugeTotalPages > 0 && hugeTotalPages != -1)
    {
        availableKb = hugeFreePages * hugePageSize;
        freeSwapKb = 0;
    }

    if (availableKb <= 0)
        return 0.0f;
    return float(double(availableKb) / (1024.0 * 1024.0));
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return float(double(status.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0));
    return 0.0f;
#elif defined(__APPLE__) && defined(__MACH__)
    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS)
    {
        long long available = ((long long)vm_stat.free_count + (long long)vm_stat.inactive_count)
                              * (long long)sysconf(_SC_PAGESIZE);
        return float(double(available) / (1024.0 * 1024.0 * 1024.0));
    }
    return 0.0f;
#else
    return 0.0f;
#endif
}

/*
#ifdef WIN32
#include <shlobj.h>
#include <direct.h>
//#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pwd.h>
#endif
//*/

using namespace std;

int max_threads = max(1, omp_get_num_procs());

int optimalFFTsize(int N)
{
    // returns smallest number = 2^(n+1)*3^m such that 2^(n+1)*3^m >= N and n,m >= 0
    if (N <= 2)
        return 2;

    double c1 = log2(double(N) / 2.0) / log2(3);
    double c2 = 1.0 / log2(3);
    //2^x*3^y = N ==> y = c1-c2*x
    double xbar = log2(double(N) / 2.0);
    int x, y;
    int minValue = int(pow(2, int(ceil(xbar)) + 1));
    int newValue;
    for (x = 0; x < int(ceil(xbar)); x++)
    {
        y = int(ceil(c1 - c2 * double(x)));
        newValue = int(pow(2, x + 1) * pow(3, y));
        if (newValue < minValue && y >= 0)
            minValue = newValue;
    }

    //printf("%d\n", minValue);

    return minValue;
}

/*
double getAvailableGBofMemory()
{
    return double(getPhysicalMemorySize()) / pow(2.0, 30);
}

size_t getPhysicalMemorySize()
{
    // Returns the size of physical memory (RAM) in bytes.

#if defined(_WIN32) && (defined(__CYGWIN__) || defined(__CYGWIN32__))
    // Cygwin under Windows. ------------------------------------
    // New 64-bit MEMORYSTATUSEX isn't available.  Use old 32.bit
    MEMORYSTATUS status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatus(&status);
    return (size_t)status.dwTotalPhys;
#elif defined(_WIN32)
    // Windows. -------------------------------------------------
    // Use new 64-bit MEMORYSTATUSEX, not old 32-bit MEMORYSTATUS
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    // UNIX variants. -------------------------------------------
    // Prefer sysctl() over sysconf() except sysctl() HW_REALMEM and HW_PHYSMEM
#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_MEMSIZE)
    mib[1] = HW_MEMSIZE;            // OSX. ---------------------
#elif defined(HW_PHYSMEM64)
    mib[1] = HW_PHYSMEM64;          // NetBSD, OpenBSD. ---------
#endif
    int64_t size = 0;               // 64-bit
    size_t len = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0)
        return (size_t)size;
    return 0L;			// Failed?
#elif defined(_SC_AIX_REALMEM)
    // AIX. -----------------------------------------------------
    return (size_t)sysconf(_SC_AIX_REALMEM) * (size_t)1024L;
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    // FreeBSD, Linux, OpenBSD, and Solaris. --------------------
    return (size_t)sysconf(_SC_PHYS_PAGES) *
        (size_t)sysconf(_SC_PAGESIZE);
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    // Legacy. --------------------------------------------------
    return (size_t)sysconf(_SC_PHYS_PAGES) *
        (size_t)sysconf(_SC_PAGE_SIZE);
#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM))
    // DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX. --------
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_REALMEM)
    mib[1] = HW_REALMEM;		// FreeBSD. -----------------
#elif defined(HW_PYSMEM)
    mib[1] = HW_PHYSMEM;		// Others. ------------------
#endif
    unsigned int size = 0;		// 32-bit
    size_t len = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0)
        return (size_t)size;
    return 0L;			// Failed?
#endif // sysctl and sysconf variants
#else
    return 0L;			// Unknown OS.
#endif

    return 0L;
}
//*/

float* getSlice(float* f, int i, parameters* params)
{
    if (params->volumeDimensionOrder == parameters::XYZ)
        return &f[uint64(i) * uint64(params->numZ) * uint64(params->numY)];
    else
        return &f[uint64(i) * uint64(params->numX) * uint64(params->numY)];
}

float* getProjection(float* g, int i, parameters* params)
{
    return &g[uint64(i) * uint64(params->numRows) * uint64(params->numCols)];
}

float tex3D(float* f, int iz, int iy, int ix, parameters* params)
{
	if (0 <= ix && ix < params->numX && 0 <= iy && iy < params->numY && 0 <= iz && iz < params->numZ)
	{
		if (params->volumeDimensionOrder == parameters::XYZ)
			return f[uint64(ix) * uint64(params->numZ * params->numY) + uint64(iy * params->numZ + iz)];
		else
			return f[uint64(iz) * uint64(params->numY * params->numX) + uint64(iy * params->numX + ix)];
	}
	else
		return 0.0;
}

float tex3D_rev(float* f, float ix, float iy, float iz, parameters* params)
{
    return tex3D(f, iz, iy, ix, params);
}

float tex3D(float* f, float iz, float iy, float ix, parameters* params)
{
    if (0.0 <= ix && ix <= params->numX-1 && 0.0 <= iy && iy <= params->numY-1 && 0.0 <= iz && iz <= params->numZ-1)
    {
        int ix_lo = int(ix);
        int ix_hi = min(ix_lo + 1, params->numX - 1);
        float dx = ix - float(ix_lo);

        int iy_lo = int(iy);
        int iy_hi = min(iy_lo + 1, params->numY - 1);
        float dy = iy - float(iy_lo);

        int iz_lo = int(iz);
        int iz_hi = min(iz_lo + 1, params->numZ - 1);
        float dz = iz - float(iz_lo);

        if (params->volumeDimensionOrder == parameters::XYZ)
        {
            float* xSlice_lo = &f[uint64(ix_lo) * uint64(params->numZ * params->numY)];
            float* xSlice_hi = &f[uint64(ix_hi) * uint64(params->numZ * params->numY)];

            float partA = (1.0 - dy) * ((1.0 - dz) * xSlice_lo[iy_lo * params->numZ + iz_lo] + dz * xSlice_lo[iy_lo * params->numZ + iz_hi]) + dy * ((1.0 - dz) * xSlice_lo[iy_hi * params->numZ + iz_lo] + dz * xSlice_lo[iy_hi * params->numZ + iz_hi]);
            float partB = (1.0 - dy) * ((1.0 - dz) * xSlice_hi[iy_lo * params->numZ + iz_lo] + dz * xSlice_hi[iy_lo * params->numZ + iz_hi]) + dy * ((1.0 - dz) * xSlice_hi[iy_hi * params->numZ + iz_lo] + dz * xSlice_hi[iy_hi * params->numZ + iz_hi]);

            return (1.0 - dx) * partA + dx * partB;
        }
        else
        {
            float* zSlice_lo = &f[uint64(iz_lo) * uint64(params->numY * params->numX)];
            float* zSlice_hi = &f[uint64(iz_hi) * uint64(params->numY * params->numX)];

            float partA = (1.0 - dy) * ((1.0 - dx) * zSlice_lo[iy_lo * params->numX + ix_lo] + dx * zSlice_lo[iy_lo * params->numX + ix_hi]) + dy * ((1.0 - dx) * zSlice_lo[iy_hi * params->numX + ix_lo] + dx * zSlice_lo[iy_hi * params->numX + ix_hi]);
            float partB = (1.0 - dy) * ((1.0 - dx) * zSlice_hi[iy_lo * params->numX + ix_lo] + dx * zSlice_hi[iy_lo * params->numX + ix_hi]) + dy * ((1.0 - dx) * zSlice_hi[iy_hi * params->numX + ix_lo] + dx * zSlice_hi[iy_hi * params->numX + ix_hi]);

            return (1.0 - dz) * partA + dz * partB;
        }
    }
    else
        return 0.0;
}

float* reorder_ZYX_to_XYZ(float* f, parameters* params, int sliceStart, int sliceEnd)
{
    if (sliceStart < 0)
        sliceStart = 0;
    if (sliceEnd < 0)
        sliceEnd = params->numZ - 1;
    int numZ_new = (sliceEnd - sliceStart + 1);
    float* f_XYZ = (float*)malloc(sizeof(float) * uint64(params->numX * params->numY) * uint64(numZ_new));
    int num_threads = num_cpu_threads();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int ix = 0; ix < params->numX; ix++)
    {
        float* xSlice_out = &f_XYZ[uint64(ix) * uint64(numZ_new * params->numY)];
        for (int iy = 0; iy < params->numY; iy++)
        {
            float* zLine_out = &xSlice_out[iy * numZ_new];
            for (int iz = sliceStart; iz <= sliceEnd; iz++)
            {
                zLine_out[iz - sliceStart] = f[uint64(iz) * uint64(params->numX * params->numY) + uint64(iy * params->numX + ix)];
            }
        }
    }
    return f_XYZ;
}

float innerProduct_cpu(float* x, float* y, int N_1, int N_2, int N_3)
{
    float* accums = new float[N_1];

    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        double accum_local = 0.0;
        float* x_slice = &x[uint64(i) * uint64(N_2 * N_3)];
        float* y_slice = &y[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                accum_local += x_slice[j * N_3 + k] * y_slice[j * N_3 + k];
        }
        accums[i] = accum_local;
    }
    float accum = 0.0;
    for (int i = 0; i < N_1; i++)
        accum += accums[i];
    delete[] accums;
    return accum;
}

bool sub_cpu(float* x, float* y, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* x_slice = &x[uint64(i) * uint64(N_2 * N_3)];
        float* y_slice = &y[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                x_slice[j * N_3 + k] -= y_slice[j * N_3 + k];
        }
    }
    return true;
}

bool scalarAdd_cpu(float* x, float c, float* y, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* x_slice = &x[uint64(i) * uint64(N_2 * N_3)];
        float* y_slice = &y[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                x_slice[j * N_3 + k] += c * y_slice[j * N_3 + k];
        }
    }
    return true;
}

bool equal_cpu(float* f_out, float* f_in, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* f_out_slice = &f_out[uint64(i) * uint64(N_2 * N_3)];
        float* f_in_slice = &f_in[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                f_out_slice[j * N_3 + k] = f_in_slice[j * N_3 + k];
        }
    }
    return true;
}

bool equal_cpu(float* f_out, float c, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* f_out_slice = &f_out[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                f_out_slice[j * N_3 + k] = c;
        }
    }
    return true;
}

bool scale_cpu(float* f, float c, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* aSlice = &f[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
                aSlice[j * N_3 + k] *= c;
        }
    }
    return true;
}

bool clip_cpu(float* f, int N_1, int N_2, int N_3, float clipVal)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* aSlice = &f[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
            {
                if (aSlice[j * N_3 + k] < clipVal)
                    aSlice[j * N_3 + k] = clipVal;
            }
        }
    }
    return true;
}

bool replaceZeros_cpu(float* f, int N_1, int N_2, int N_3, float newVal)
{
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        float* aSlice = &f[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
            {
                if (aSlice[j * N_3 + k] == 0.0)
                    aSlice[j * N_3 + k] = newVal;
            }
        }
    }
    return true;
}

float sum_cpu(float* f, int N_1, int N_2, int N_3)
{
    double* sums = new double[N_1];
    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for
    for (int i = 0; i < N_1; i++)
    {
        double sum = 0.0;
        float* aSlice = &f[uint64(i) * uint64(N_2 * N_3)];
        for (int j = 0; j < N_2; j++)
        {
            for (int k = 0; k < N_3; k++)
            {
                sum += double(aSlice[j * N_3 + k]);
            }
        }
        sums[i] = sum;
    }

    double retVal = 0.0;
    for (int i = 0; i < N_1; i++)
        retVal += sums[i];
    delete[] sums;

    return float(retVal);
}

bool windowFOV_cpu(float* f, parameters* params)
{
    if (f == NULL || params == NULL)
        return false;
    else
    {
        float rFOVsq = params->rFOV() * params->rFOV();
        if (params->volumeDimensionOrder == parameters::XYZ)
        {
            omp_set_num_threads(num_cpu_threads());
            #pragma omp parallel for
            for (int ix = 0; ix < params->numX; ix++)
            {
                float x = ix * params->voxelWidth + params->x_0();
                for (int iy = 0; iy < params->numY; iy++)
                {
                    float y = iy * params->voxelWidth + params->y_0();
                    if (x * x + y * y > rFOVsq)
                    {
                        float* zLine = &f[uint64(ix) * uint64(params->numY * params->numZ) + uint64(iy * params->numZ)];
                        for (int iz = 0; iz < params->numZ; iz++)
                            zLine[iz] = 0.0;
                    }
                }
            }
        }
        else // ZYX
        {
            omp_set_num_threads(num_cpu_threads());
            #pragma omp parallel for
            for (int iz = 0; iz < params->numZ; iz++)
            {
                float* zSlice = &f[uint64(iz) * uint64(params->numX * params->numY)];
                for (int iy = 0; iy < params->numY; iy++)
                {
                    float y = iy * params->voxelWidth + params->y_0();
                    for (int ix = 0; ix < params->numX; ix++)
                    {
                        float x = ix * params->voxelWidth + params->x_0();
                        if (x * x + y * y > rFOVsq)
                            zSlice[iy * params->numX + ix] = 0.0;
                    }
                }
            }
        }
        return true;
    }
}

float* rotateAroundAxis(float* theAxis, float phi, float* aVec)
{
    // assume vectors in R^3 and theAxis is a unit vector
    // reference: http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/, section 5.2

    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float axis_dot_vector = theAxis[0] * aVec[0] + theAxis[1] * aVec[1] + theAxis[2] * aVec[2];

    float u = theAxis[0];
    float v = theAxis[1];
    float w = theAxis[2];

    float x = aVec[0];
    float y = aVec[1];
    float z = aVec[2];

    aVec[0] = u * axis_dot_vector * (1.0 - cos_phi) + x * cos_phi + (-w * y + v * z) * sin_phi;
    aVec[1] = v * axis_dot_vector * (1.0 - cos_phi) + y * cos_phi + (w * x - u * z) * sin_phi;
    aVec[2] = w * axis_dot_vector * (1.0 - cos_phi) + z * cos_phi + (-v * x + u * y) * sin_phi;

    return aVec;
}

// Bitcast float->uint32_t without UB (safe under fast-math)
static inline uint32_t float_bits(float x) noexcept
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

// Returns true if x is NaN or +/-Inf (IEEE-754 binary32)
static inline bool is_nan_or_inf_bits(float x) noexcept
{
    // exponent all ones => NaN or Inf
    return (float_bits(x) & 0x7f800000u) == 0x7f800000u;
}

bool has_nan_or_inf_omp_fastmath(const float* __restrict a, std::size_t n)
{
    std::atomic<bool> found{false};

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        // Static partition: [begin, end)
        const std::size_t begin = (n * (std::size_t)tid) / (std::size_t)nt;
        const std::size_t end   = (n * (std::size_t)(tid + 1)) / (std::size_t)nt;

        for (std::size_t i = begin; i < end; ++i) {
            if (found.load(std::memory_order_relaxed)) break;

            if (is_nan_or_inf_bits(a[i])) {
                found.store(true, std::memory_order_relaxed);
                break;
            }
        }
    }

    return found.load(std::memory_order_relaxed);
}

bool has_nan(float* x, int N_1, int N_2, int N_3)
{
    if (x == NULL)
    {
        printf("has_nan: data is NULL\n");
        return true;
    }
    else if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
    {
        printf("has_nan: data has zero-length dimension\n");
        return true;
    }
    else
    {
        return has_nan_or_inf_omp_fastmath(x, size_t(N_1) * size_t(N_2) * size_t(N_3));
        /*
        for (int i = 0; i < N_1; i++)
        {
            float* data2D = &x[uint64(i)*uint64(N_2)*uint64(N_3)];
            for (int j = 0; j < N_2; j++)
            {
                for (int k = 0; k < N_3; k++)
                {
                    if (isnan(data2D[j*N_3 + k]))
                    {
                        printf("data[%d,%d,%d] is nan\n", i, j, k);
                        return true;
                    }
                }
            }
        }
        return false;
        //*/
    }
}

bool replace_nan(float* x, int N_1, int N_2, int N_3, float newValue)
{
    if (x == NULL)
        return false;
    else if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
    {
        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_1; i++)
        {
            float* data2D = &x[uint64(i)*uint64(N_2)*uint64(N_3)];
            for (int j = 0; j < N_2; j++)
            {
                for (int k = 0; k < N_3; k++)
                {
                    if (isnan(data2D[j*N_3 + k]))
                    {
                        data2D[j*N_3 + k] = newValue;
                    }
                }
            }
        }
        return true;
    }
}

bool bounding_box(float* x, int N_1, int N_2, int N_3, int boundary_type, int* AABB)
{
    if (x == NULL || AABB == NULL)
        return false;
    else if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
    {
        int* AABB_slices = new int[N_1*4];
        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_1; i++)
        {
            int* AABB_cur = &AABB_slices[i*4];
            AABB_cur[0] = -1;
            AABB_cur[1] = -1;
            AABB_cur[2] = -1;
            AABB_cur[3] = -1;
            bool found_value = false;
            float* data2D = &x[uint64(i)*uint64(N_2)*uint64(N_3)];
            for (int j = 0; j < N_2; j++)
            {
                for (int k = 0; k < N_3; k++)
                {
                    float curVal = data2D[j*N_3+k];
                    if (boundary_type == 0)
                    {
                        if (curVal > 0.0)
                        {
                            if (found_value)
                            {
                                AABB_cur[0] = min(AABB_cur[0], j);
                                AABB_cur[1] = max(AABB_cur[1], j);
                                AABB_cur[2] = min(AABB_cur[2], k);
                                AABB_cur[3] = max(AABB_cur[3], k);
                            }
                            else
                            {
                                AABB_cur[0] = j;
                                AABB_cur[1] = j;
                                AABB_cur[2] = k;
                                AABB_cur[3] = k;
                            }
                            found_value = true;
                        }

                    }
                    else if (boundary_type == 1)
                    {
                        if (curVal < 0.0)
                        {
                            if (found_value)
                            {
                                AABB_cur[0] = min(AABB_cur[0], j);
                                AABB_cur[1] = max(AABB_cur[1], j);
                                AABB_cur[2] = min(AABB_cur[2], k);
                                AABB_cur[3] = max(AABB_cur[3], k);
                            }
                            else
                            {
                                AABB_cur[0] = j;
                                AABB_cur[1] = j;
                                AABB_cur[2] = k;
                                AABB_cur[3] = k;
                            }
                            found_value = true;
                        }
                    }
                    else //if (boundary_type == 2)
                    {
                        if (isnan(curVal))
                        {
                            if (found_value)
                            {
                                AABB_cur[0] = min(AABB_cur[0], j);
                                AABB_cur[1] = max(AABB_cur[1], j);
                                AABB_cur[2] = min(AABB_cur[2], k);
                                AABB_cur[3] = max(AABB_cur[3], k);
                            }
                            else
                            {
                                AABB_cur[0] = j;
                                AABB_cur[1] = j;
                                AABB_cur[2] = k;
                                AABB_cur[3] = k;
                            }
                            found_value = true;
                        }
                    }
                }
            }
        }

        AABB[0] = -1;
        AABB[1] = -1;
        AABB[2] = -1;
        AABB[3] = -1;
        AABB[4] = -1;
        AABB[5] = -1;
        for (int i = 0; i < N_1; i++)
        {
            int* AABB_cur = &AABB_slices[i*4];
            if (AABB_cur[0] != -1 && AABB_cur[1] != -1 && AABB_cur[2] != -1 && AABB_cur[3] != -1)
            {
                if (AABB[0] == -1 || AABB[1] == -1 || AABB[2] == -1 || AABB[3] == -1 || AABB[4] == -1 || AABB[5] == -1)
                {
                    // not set yet
                    AABB[0] = i;
                    AABB[1] = i;
                    AABB[2] = AABB_cur[0];
                    AABB[3] = AABB_cur[1];
                    AABB[4] = AABB_cur[2];
                    AABB[5] = AABB_cur[3];
                }
                else
                {
                    AABB[0] = min(AABB[0], i);
                    AABB[1] = max(AABB[1], i);
                    AABB[2] = min(AABB[2], AABB_cur[0]);
                    AABB[3] = max(AABB[3], AABB_cur[1]);
                    AABB[4] = min(AABB[4], AABB_cur[2]);
                    AABB[5] = max(AABB[5], AABB_cur[3]);
                }
            }
        }

        delete [] AABB_slices;

        if (AABB[0] == -1 || AABB[1] == -1 || AABB[2] == -1 || AABB[3] == -1 || AABB[4] == -1 || AABB[5] == -1)
            return false;
        else
           return true;
    }
}

bool step_function(float* x, int N_1, int N_2, int N_3, float scale, float shift)
{
    if (x == NULL)
        return false;
    else if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
    {
        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_1; i++)
        {
            float* data2D = &x[uint64(i)*uint64(N_2)*uint64(N_3)];
            for (int j = 0; j < N_2; j++)
            {
                for (int k = 0; k < N_3; k++)
                {
                    if (scale * data2D[j*N_3+k] + shift > 0.0)
                        data2D[j*N_3+k] = 1.0;
                    else
                        data2D[j*N_3+k] = 0.0;
                }
            }
        }
        return true;
    }
}

bool dirac_function(float* x, int N_1, int N_2, int N_3, float scale, float shift)
{
    if (x == NULL)
        return false;
    else if (N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
        return false;
    else
    {
        omp_set_num_threads(num_cpu_threads());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N_1; i++)
        {
            float* data2D = &x[uint64(i)*uint64(N_2)*uint64(N_3)];
            for (int j = 0; j < N_2; j++)
            {
                for (int k = 0; k < N_3; k++)
                {
                    if (scale * data2D[j*N_3+k] + shift == 0.0)
                        data2D[j*N_3+k] = 1.0;
                    else
                        data2D[j*N_3+k] = 0.0;
                }
            }
        }
        return true;
    }
}

float findMedian(std::vector<float>& nums)
{
    int n = nums.size();
    int mid = n / 2;

    std::nth_element(nums.begin(), nums.begin() + mid, nums.end());

    if (n % 2 == 1)
        return nums[mid];
    else
    {
        float upper = nums[mid];
        std::nth_element(nums.begin(), nums.begin() + mid - 1, nums.begin() + mid);
        float lower = nums[mid - 1];
        return (lower + upper) / 2.0f;
    }
}

bool badPixelCorrection_cpu(float* g, int N_1, int N_2, int N_3, float* badPixelMap, int w)
{
    if (g == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0 || badPixelMap == NULL)
        return false;
    w = max(1, min(w, 3));

    omp_set_num_threads(num_cpu_threads());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_1; i++)
    {
        float* x = &g[uint64(i)*uint64(N_2)*uint64(N_3)];
        for (int ind = 0; ind < N_2*N_3; ind++)
        {
            if (badPixelMap[ind] > 0.0)
            {
                vector<float> neighbors;
                //ind = j*N_3 + k;
                int k = ind % N_3;
                int j = (ind-k) / N_3;
                for (int dj = -w; dj <= w; dj++)
                {
                    int jj = j + dj;
                    if (jj < 0 || jj >= N_2)
                        continue;
                    for (int dk = -w; dk <= w; dk++)
                    {
                        int kk = k + dk;
                        if (kk < 0 || kk >= N_3)
                            continue;
                        if (badPixelMap[jj*N_3 + kk] == 0.0)
                            neighbors.push_back(x[jj*N_3 + kk]);
                    }
                }
                if (neighbors.size() == 0)
                {
                    for (int dj = -w-1; dj <= w+1; dj++)
                    {
                        int jj = j + dj;
                        if (jj < 0 || jj >= N_2)
                            continue;
                        for (int dk = -w-1; dk <= w+1; dk++)
                        {
                            int kk = k + dk;
                            if (kk < 0 || kk >= N_3)
                                continue;
                            if (badPixelMap[jj*N_3 + kk] == 0.0)
                                neighbors.push_back(x[jj*N_3 + kk]);
                        }
                   }
                }
                if (neighbors.size() > 0)
                    x[ind] = findMedian(neighbors);
            }
        }
    }
    return true;
}

int num_cpu_threads()
{
    //return max(1, min(MAX_CPU_THREADS, omp_get_num_procs()-1));
    //return max(1, min(MAX_CPU_THREADS, omp_get_num_procs()));
    return max(1, min(max_threads, omp_get_num_procs()));
}

char swapEndian(char x)
{
    x = bswap(x);
    return x;
}

short swapEndian(short x)
{
    return (x << 8) | ((x >> 8) & 0xFF);
}

unsigned short swapEndian(unsigned short x)
{
    return (x << 8) | (x >> 8);
}

int swapEndian(int x)
{
    x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0xFF00FF);
    return (x << 16) | ((x >> 16) & 0xFFFF);
}

unsigned int swapEndian(unsigned int x)
{
    x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0xFF00FF);
    return (x << 16) | (x >> 16);
}

float swapEndian(float x)
{
    x = bswap(x);
    return x;
}

double swapEndian(double x)
{
    x = bswap(x);
    return x;
}

template <typename T>
T bswap(T val)
{
    T retVal;
    char* pVal = (char*)&val;
    char* pRetVal = (char*)&retVal;
    int size = sizeof(T);
    for (int i = 0; i < size; i++)
    {
        pRetVal[size - 1 - i] = pVal[i];
    }

    return retVal;
}

float* malloc_aligned(size_t num_bytes, int alignment)
{
    if (alignment != 16 && alignment != 32 && alignment != 64)
        return nullptr;
    if (num_bytes <= 0)
        return nullptr;
    float* data = NULL;
    #if defined(_MSC_VER) || defined(_WIN32)
        data = (float*) _aligned_malloc(num_bytes, alignment);
        if (!data)
            return nullptr;
        else
            return data;
    #else
        int result = posix_memalign((void**)&data, alignment, num_bytes);
        if (result != 0 || !data)
            return nullptr;
        else
            return data;
    #endif
}

float* calloc_aligned(size_t num_bytes, int alignment)
{
    float* retVal = malloc_aligned(num_bytes, alignment);
    if (retVal != NULL)
        memset(retVal, 0, num_bytes);
    return retVal;
}

bool free_aligned(float* data)
{
    if (data != NULL)
    {
        #if defined(_MSC_VER) || defined(_WIN32)
            _aligned_free(data);
        #else
            free(data);
        #endif
        return true;
    }
    else
        return false;
}

void unpack01_from_float(float packed, float& a, float& b)
{
    uint32_t bits;
    std::memcpy(&bits, &packed, sizeof(bits)); // safe bit-cast

    uint32_t ai = bits >> 16;
    uint32_t bi = bits & 0xffffu;

    a = float(ai) / 65535.0f;
    b = float(bi) / 65535.0f;
}

static inline float half_to_float(uint16_t h)
{
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0)
    {
        if (mant == 0)
            bits = sign; // +/- zero
        else
        {
            // subnormal half -> normalized float
            exp = 1;
            while ((mant & 0x400u) == 0)
            {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13); // 112 = 127 - 15
        }
    }
    else if (exp == 0x1Fu)
        bits = sign | 0x7F800000u | (mant << 13); // Inf / NaN
    else
        bits = sign | ((exp + 112u) << 23) | (mant << 13);

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void unpack_half2_from_float(float packed, float& a, float& b)
{
    uint32_t bits;
    std::memcpy(&bits, &packed, sizeof(bits)); // safe bit-cast

    a = half_to_float(uint16_t(bits >> 16));
    b = half_to_float(uint16_t(bits & 0xffffu));
}

#ifdef __USE_CPU
// CPU-build stubs for the GPU helpers declared in cuda_utils.h.
// In GPU builds these come from cuda_utils.cu; cuda_utils.cu is not compiled
// in CPU-only builds, so callers in shared .cpp translation units would
// otherwise fail to link.
#include <vector>
#include "cuda_utils.h"

int numberOfGPUs()
{
    return 0;
}

float getAvailableGPUmemory(std::vector<int> /*whichGPUs*/)
{
    return 0.0;
}

float getAvailableGPUmemory(int /*whichGPU*/)
{
    return 0.0;
}

bool physically_shared_memory(int /*whichGPU*/)
{
    return false;
}
#endif
