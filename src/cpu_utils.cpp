#include <omp.h>
#include "cpu_utils.h"

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
