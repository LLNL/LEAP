#include <omp.h>
#include <stdlib.h>
#include "cpu_utils.h"

using namespace std;

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
