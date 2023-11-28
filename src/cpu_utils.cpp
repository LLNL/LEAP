#include <omp.h>
#include "cpu_utils.h"

using namespace std;

float tex3D(float* f, int iz, int iy, int ix, parameters* params)
{
	if (0 <= ix && ix < params->numX && 0 <= iy && iy < params->numY && 0 <= iz && iz < params->numZ)
	{
		if (params->volumeDimensionOrder == parameters::XYZ)
			return f[ix * params->numZ * params->numY + iy * params->numZ + iz];
		else
			return f[iz * params->numY * params->numX + iy * params->numX + ix];
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
            float* xSlice_lo = &f[ix_lo * params->numZ * params->numY];
            float* xSlice_hi = &f[ix_hi * params->numZ * params->numY];

            float partA = (1.0 - dy) * ((1.0 - dz) * xSlice_lo[iy_lo * params->numZ + iz_lo] + dz * xSlice_lo[iy_lo * params->numZ + iz_hi]) + dy * ((1.0 - dz) * xSlice_lo[iy_hi * params->numZ + iz_lo] + dz * xSlice_lo[iy_hi * params->numZ + iz_hi]);
            float partB = (1.0 - dy) * ((1.0 - dz) * xSlice_hi[iy_lo * params->numZ + iz_lo] + dz * xSlice_hi[iy_lo * params->numZ + iz_hi]) + dy * ((1.0 - dz) * xSlice_hi[iy_hi * params->numZ + iz_lo] + dz * xSlice_hi[iy_hi * params->numZ + iz_hi]);

            return (1.0 - dx) * partA + dx * partB;
        }
        else
        {
            float* zSlice_lo = &f[iz_lo * params->numY * params->numX];
            float* zSlice_hi = &f[iz_hi * params->numY * params->numX];

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
    float* f_XYZ = (float*)malloc(sizeof(float) * params->numX * params->numY * numZ_new);
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int ix = 0; ix < params->numX; ix++)
    {
        float* xSlice_out = &f_XYZ[ix * numZ_new * params->numY];
        for (int iy = 0; iy < params->numY; iy++)
        {
            float* zLine_out = &xSlice_out[iy * numZ_new];
            for (int iz = sliceStart; iz <= sliceEnd; iz++)
            {
                zLine_out[iz - sliceStart] = f[iz * params->numX * params->numY + iy * params->numX + ix];
            }
        }
    }
    return f_XYZ;
}
