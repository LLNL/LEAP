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
#include "cpu_utils.h"

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
    int num_threads = omp_get_num_procs();
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

    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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

bool scale_cpu(float* f, float c, int N_1, int N_2, int N_3)
{
    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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
    omp_set_num_threads(omp_get_num_procs());
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
            omp_set_num_threads(omp_get_num_procs());
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
            omp_set_num_threads(omp_get_num_procs());
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
