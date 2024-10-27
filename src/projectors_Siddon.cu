////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for Siddon projector (deprecated)
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors_Siddon.cuh"
#include "cuda_utils.h"
//using namespace std;

__device__ float projectLine(cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float4 pos, float4 traj)
{
	float val = 0.0;
	if (fabs(traj.x) >= fabs(traj.y) && fabs(traj.x) >= fabs(traj.z))
	{
		float x_bottom = startVals_f.x - 0.5f*T_f.x;
		float t_bottom = (x_bottom - pos.x) / traj.x;

		float y_low = (pos.y + t_bottom * traj.y - startVals_f.y) / T_f.y;
		float z_low = (pos.z + t_bottom * traj.z - startVals_f.z) / T_f.z;

		float y_inc = (traj.y / traj.x) * (T_f.y / T_f.x);
		float z_inc = (traj.z / traj.x) * (T_f.z / T_f.x);

		if (y_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy + 1, ix)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy + 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy + 1, ix)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy + 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy + 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else if (y_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy - 1, ix)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy - 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy - 1, ix)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy - 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy - 1, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else // y_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc))*tex3D<float>(f, iz + 1, iy, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc))*tex3D<float>(f, iz - 1, iy, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D<float>(f, iz, iy, ix);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + y_inc * y_inc + z_inc * z_inc)*T_f.x;
	}
	else if (fabs(traj.y) >= fabs(traj.x) && fabs(traj.y) >= fabs(traj.z))
	{
		float y_bottom = startVals_f.y - 0.5f*T_f.y;
		float t_bottom = (y_bottom - pos.y) / traj.y;

		float x_low = (pos.x + t_bottom * traj.x - startVals_f.x) / T_f.x;
		float z_low = (pos.z + t_bottom * traj.z - startVals_f.z) / T_f.z;

		float x_inc = (traj.x / traj.y) * (T_f.x / T_f.y);
		float z_inc = (traj.z / traj.y) * (T_f.z / T_f.y);

		if (x_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix + 1)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix + 1)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix - 1)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz + 1, iy, ix - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz, iy, ix - 1)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc)))*tex3D<float>(f, iz - 1, iy, ix - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc))*tex3D<float>(f, iz + 1, iy, ix);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc))*tex3D<float>(f, iz - 1, iy, ix);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D<float>(f, iz, iy, ix);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + z_inc * z_inc)*T_f.y;
	}
	else
	{
		float z_bottom = startVals_f.z - 0.5f*T_f.z;
		float t_bottom = (z_bottom - pos.z) / traj.z;

		float x_low = (pos.x + t_bottom * traj.x - startVals_f.x) / T_f.x;
		float y_low = (pos.y + t_bottom * traj.y - startVals_f.y) / T_f.y;

		float x_inc = (traj.x / traj.z) * (T_f.x / T_f.z);
		float y_inc = (traj.y / traj.z) * (T_f.y / T_f.z);

		if (x_inc > 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix + 1)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy + 1, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy + 1, ix + 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix + 1)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy - 1, ix)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy - 1, ix + 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else //if (y_inc == 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix + 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix - 1)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy + 1, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy + 1, ix - 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy, ix - 1)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy - 1, ix)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc)))*tex3D<float>(f, iz, iy - 1, ix - 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc))*tex3D<float>(f, iz, iy, ix - 1);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.y; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy + 1, ix);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy, ix)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc))*tex3D<float>(f, iz, iy - 1, ix);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += tex3D<float>(f, iz, iy, ix);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + y_inc * y_inc)*T_f.z;
	}

	return val;
}

__device__ float projectLine_ZYX(cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float4 pos, float4 traj)
{
	float val = 0.0;
	if (fabs(traj.x) >= fabs(traj.y) && fabs(traj.x) >= fabs(traj.z))
	{
		float x_bottom = startVals_f.x - 0.5f * T_f.x;
		float t_bottom = (x_bottom - pos.x) / traj.x;

		float y_low = (pos.y + t_bottom * traj.y - startVals_f.y) / T_f.y;
		float z_low = (pos.z + t_bottom * traj.z - startVals_f.z) / T_f.z;

		float y_inc = (traj.y / traj.x) * (T_f.y / T_f.x);
		float z_inc = (traj.z / traj.x) * (T_f.z / T_f.x);

		if (y_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy + 1, iz)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz + 1)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy + 1, iz + 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy + 1, iz)
						+ max(0.0f, min(1.0f, min((iy + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz - 1)
						+ max(0.0f, min(1.0f, min((iy + 1 + 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 1 - 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy + 1, iz - 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy + 1, iz);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else if (y_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy - 1, iz)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz + 1)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy - 1, iz + 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy - 1, iz)
						+ max(0.0f, min(1.0f, min((iy - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz - 1)
						+ max(0.0f, min(1.0f, min((iy - 1 - 0.5f - y_low) / y_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((iy - 1 + 0.5f - y_low) / y_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy - 1, iz - 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy - 1, iz);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		else // y_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz + 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz - 1);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int ix = 0; ix < N_f.x; ix++)
				{
					int iy = int(y_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D<float>(f, ix, iy, iz);

					y_low += y_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + y_inc * y_inc + z_inc * z_inc) * T_f.x;
	}
	else if (fabs(traj.y) >= fabs(traj.x) && fabs(traj.y) >= fabs(traj.z))
	{
		float y_bottom = startVals_f.y - 0.5f * T_f.y;
		float t_bottom = (y_bottom - pos.y) / traj.y;

		float x_low = (pos.x + t_bottom * traj.x - startVals_f.x) / T_f.x;
		float z_low = (pos.z + t_bottom * traj.z - startVals_f.z) / T_f.z;

		float x_inc = (traj.x / traj.y) * (T_f.x / T_f.y);
		float z_inc = (traj.z / traj.y) * (T_f.z / T_f.y);

		if (x_inc > 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix + 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz + 1)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix + 1, iy, iz + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix + 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz - 1)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix + 1, iy, iz - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else //if (z_inc == 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix + 1, iy, iz);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix - 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz + 1)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz + 1 + 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 1 - 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix - 1, iy, iz + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix - 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix, iy, iz - 1)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iz - 1 - 0.5f - z_low) / z_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iz - 1 + 0.5f - z_low) / z_inc))) * tex3D<float>(f, ix - 1, iy, iz - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix - 1, iy, iz);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (z_inc > 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz + 0.5f - z_low) / z_inc) - max(0.0f, (iz - 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iz + 1 + 0.5f - z_low) / z_inc) - max(0.0f, (iz + 1 - 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz + 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else if (z_inc < 0.0f)
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += max(0.0f, min(1.0f, (iz - 0.5f - z_low) / z_inc) - max(0.0f, (iz + 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iz - 1 - 0.5f - z_low) / z_inc) - max(0.0f, (iz - 1 + 0.5f - z_low) / z_inc)) * tex3D<float>(f, ix, iy, iz - 1);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
			else
			{
				for (int iy = 0; iy < N_f.y; iy++)
				{
					int ix = int(x_low + 0.5f);
					int iz = int(z_low + 0.5f);

					val += tex3D<float>(f, ix, iy, iz);

					x_low += x_inc;
					z_low += z_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + z_inc * z_inc) * T_f.y;
	}
	else
	{
		float z_bottom = startVals_f.z - 0.5f * T_f.z;
		float t_bottom = (z_bottom - pos.z) / traj.z;

		float x_low = (pos.x + t_bottom * traj.x - startVals_f.x) / T_f.x;
		float y_low = (pos.y + t_bottom * traj.y - startVals_f.y) / T_f.y;

		float x_inc = (traj.x / traj.z) * (T_f.x / T_f.z);
		float y_inc = (traj.y / traj.z) * (T_f.y / T_f.z);

		if (x_inc > 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					// when ray starts to enter the (iy)th sample, find the voxel that the ray lies within
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix + 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy + 1, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix + 1, iy + 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix + 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy - 1, iz)
						+ max(0.0f, min(1.0f, min((ix + 1 + 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 1 - 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix + 1, iy - 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else //if (y_inc == 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix + 0.5f - x_low) / x_inc) - max(0.0f, (ix - 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (ix + 1 + 0.5f - x_low) / x_inc) - max(0.0f, (ix + 1 - 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix + 1, iy, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else if (x_inc < 0.0f)
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix - 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy + 1, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy + 1 + 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 1 - 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix - 1, iy + 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix - 1, iy, iz)
						+ max(0.0f, min(1.0f, min((ix - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix, iy - 1, iz)
						+ max(0.0f, min(1.0f, min((ix - 1 - 0.5f - x_low) / x_inc, (iy - 1 - 0.5f - y_low) / y_inc)) - max(0.0f, max((ix - 1 + 0.5f - x_low) / x_inc, (iy - 1 + 0.5f - y_low) / y_inc))) * tex3D<float>(f, ix - 1, iy - 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f); // ?

					val += max(0.0f, min(1.0f, (ix - 0.5f - x_low) / x_inc) - max(0.0f, (ix + 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (ix - 1 - 0.5f - x_low) / x_inc) - max(0.0f, (ix - 1 + 0.5f - x_low) / x_inc)) * tex3D<float>(f, ix - 1, iy, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		else // x_inc == 0.0f
		{
			if (y_inc > 0.0f)
			{
				for (int iz = 0; iz < N_f.y; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy + 0.5f - y_low) / y_inc) - max(0.0f, (iy - 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iy + 1 + 0.5f - y_low) / y_inc) - max(0.0f, (iy + 1 - 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy + 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else if (y_inc < 0.0f)
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += max(0.0f, min(1.0f, (iy - 0.5f - y_low) / y_inc) - max(0.0f, (iy + 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy, iz)
						+ max(0.0f, min(1.0f, (iy - 1 - 0.5f - y_low) / y_inc) - max(0.0f, (iy - 1 + 0.5f - y_low) / y_inc)) * tex3D<float>(f, ix, iy - 1, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
			else
			{
				for (int iz = 0; iz < N_f.z; iz++)
				{
					int ix = int(x_low + 0.5f);
					int iy = int(y_low + 0.5f);

					val += tex3D<float>(f, ix, iy, iz);

					x_low += x_inc;
					y_low += y_inc;
				}
			}
		}
		val = val * sqrt(1.0 + x_inc * x_inc + y_inc * y_inc) * T_f.z;
	}

	return val;
}

__global__ void fanBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float rFOVsq, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	const float x = i * T_f.x + startVals_f.x;
	const float y = j * T_f.y + startVals_f.y;
	const float z = k * T_f.z + startVals_f.z;

	uint64 ind;
	if (volumeDimensionOrder == 0)
		ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
	else
		ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

	if (x * x + y * y > rFOVsq)
	{
		f[ind] = 0.0;
		return;
	}

	const float x_width = 0.5f * T_f.x;
	const float y_width = 0.5f * T_f.y;
	const float z_width = 0.5f * T_f.z;

	const float voxelToMagnifiedDetectorPixelRatio_u = T_f.x / (R / D * T_g.z);
	//const float voxelToMagnifiedDetectorPixelRatio_v = T_f.z / T_g.y;

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f * voxelToMagnifiedDetectorPixelRatio_u);
	//const int searchWidth_v = 1 + int(0.5f * voxelToMagnifiedDetectorPixelRatio_v);

	float val = 0.0f;
	for (int iphi = 0; iphi < N_g.x; iphi++)
	{
		const float cos_phi = cos(phis[iphi]);
		const float sin_phi = sin(phis[iphi]);
		float4 pos;
		pos.x = x - R * cos_phi;
		pos.y = y - R * sin_phi;
		pos.z = z - 0.0f;

		const float v_denom = R - x * cos_phi - y * sin_phi;
		const int u_arg_mid = int(0.5 + (D * (y * cos_phi - x * sin_phi) / v_denom - startVals_g.z) / T_g.z);
		const int iv = int(0.5 + (z - startVals_g.y) / T_g.y);

		//const float v = iv * T_g.y + startVals_g.y;
		for (int iu = u_arg_mid - searchWidth_u; iu <= u_arg_mid + searchWidth_u; iu++)
		{
			const float u = iu * T_g.z + startVals_g.z;

			float4 traj;
			const float trajLength_inv = rsqrt(D * D + u * u);
			traj.x = (-D * cos_phi - u * sin_phi) * trajLength_inv;
			traj.y = (-D * sin_phi + u * cos_phi) * trajLength_inv;
			traj.z = 0.0f;

			float t_max = 1.0e16;
			float t_min = -1.0e16;
			if (traj.x != 0.0f)
			{
				const float t_a = (pos.x + x_width) / traj.x;
				const float t_b = (pos.x - x_width) / traj.x;
				t_max = min(t_max, max(t_b, t_a));
				t_min = max(t_min, min(t_b, t_a));
			}
			if (traj.y != 0.0f)
			{
				const float t_a = (pos.y + y_width) / traj.y;
				const float t_b = (pos.y - y_width) / traj.y;
				t_max = min(t_max, max(t_b, t_a));
				t_min = max(t_min, min(t_b, t_a));
			}
			if (traj.z != 0.0f)
			{
				const float t_a = (pos.z + z_width) / traj.z;
				const float t_b = (pos.z - z_width) / traj.z;
				t_max = min(t_max, max(t_b, t_a));
				t_min = max(t_min, min(t_b, t_a));
			}

			val += max(0.0f, t_max - t_min) * tex3D<float>(g, iu, iv, iphi);
		}
	}
	f[ind] = val;
}

__global__ void fanBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;

	float v = j * T_g.y + startVals_g.y;
	float u = k * T_g.z + startVals_g.z;

	float cos_phi = cos(phis[i]);
	float sin_phi = sin(phis[i]);
	float4 sourcePos;
	sourcePos.x = R * cos_phi;
	sourcePos.y = R * sin_phi;
	sourcePos.z = v;

	float4 traj;
	traj.x = -D * cos_phi - u * sin_phi;
	traj.y = -D * sin_phi + u * cos_phi;
	traj.z = 0.0;

	if (volumeDimensionOrder == 0)
		g[i * N_g.y * N_g.z + j * N_g.z + k] = projectLine(f, N_f, T_f, startVals_f, sourcePos, traj);
	else
		g[i * N_g.y * N_g.z + j * N_g.z + k] = projectLine_ZYX(f, N_f, T_f, startVals_f, sourcePos, traj);
}

__global__ void coneBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float rFOVsq, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	const float x = i * T_f.x + startVals_f.x;
	const float y = j * T_f.y + startVals_f.y;
	const float z = k * T_f.z + startVals_f.z;

	uint64 ind;
	if (volumeDimensionOrder == 0)
		ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
	else
		ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

	if (x * x + y * y > rFOVsq)
	{
		f[ind] = 0.0;
		return;
	}

	const float x_width = 0.5f*T_f.x;
	const float y_width = 0.5f*T_f.y;
	const float z_width = 0.5f*T_f.z;

	const float voxelToMagnifiedDetectorPixelRatio_u = T_f.x / (R / D * T_g.z);
	const float voxelToMagnifiedDetectorPixelRatio_v = T_f.z / (R / D * T_g.y);

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	const int searchWidth_v = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_v);

	float val = 0.0f;
	for (int iphi = 0; iphi < N_g.x; iphi++)
	{
		const float cos_phi = cos(phis[iphi]);
		const float sin_phi = sin(phis[iphi]);
		float4 pos;
		pos.x = x - R * cos_phi;
		pos.y = y - R * sin_phi;
		pos.z = z - 0.0f;

		const float v_denom = R - x * cos_phi - y * sin_phi;
		const int u_arg_mid = int(0.5 + (D*(y*cos_phi - x * sin_phi) / v_denom - startVals_g.z) / T_g.z);
		const int v_arg_mid = int(0.5 + (D*z / v_denom - startVals_g.y) / T_g.y);

		for (int iv = v_arg_mid - searchWidth_v; iv <= v_arg_mid + searchWidth_v; iv++)
		{
			const float v = iv * T_g.y + startVals_g.y;
			for (int iu = u_arg_mid - searchWidth_u; iu <= u_arg_mid + searchWidth_u; iu++)
			{
				const float u = iu * T_g.z + startVals_g.z;

				float4 traj;
				const float trajLength_inv = rsqrt(D*D + u * u + v * v);
				traj.x = (-D * cos_phi - u * sin_phi) * trajLength_inv;
				traj.y = (-D * sin_phi + u * cos_phi) * trajLength_inv;
				traj.z = v * trajLength_inv;

				float t_max = 1.0e16;
				float t_min = -1.0e16;
				if (traj.x != 0.0f)
				{
					const float t_a = (pos.x + x_width) / traj.x;
					const float t_b = (pos.x - x_width) / traj.x;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}
				if (traj.y != 0.0f)
				{
					const float t_a = (pos.y + y_width) / traj.y;
					const float t_b = (pos.y - y_width) / traj.y;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}
				if (traj.z != 0.0f)
				{
					const float t_a = (pos.z + z_width) / traj.z;
					const float t_b = (pos.z - z_width) / traj.z;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}

				val += max(0.0f, t_max - t_min)*tex3D<float>(g, iu, iv, iphi);
			}
		}
	}
	f[ind] = val;
}

__global__ void coneBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;

	float v = j * T_g.y + startVals_g.y;
	float u = k * T_g.z + startVals_g.z;

	float cos_phi = cos(phis[i]);
	float sin_phi = sin(phis[i]);
	float4 sourcePos;
	sourcePos.x = R * cos_phi;
	sourcePos.y = R * sin_phi;
	sourcePos.z = 0.0f;

	float4 traj;
	traj.x = -D * cos_phi - u * sin_phi;
	traj.y = -D * sin_phi + u * cos_phi;
	traj.z = v;

	if (volumeDimensionOrder == 0)
		g[i*N_g.y*N_g.z + j * N_g.z + k] = projectLine(f, N_f, T_f, startVals_f, sourcePos, traj);
	else
		g[i * N_g.y * N_g.z + j * N_g.z + k] = projectLine_ZYX(f, N_f, T_f, startVals_f, sourcePos, traj);
}

__global__ void parallelBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float rFOVsq, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	const float x = i * T_f.x + startVals_f.x;
	const float y = j * T_f.y + startVals_f.y;
	const float z = k * T_f.z + startVals_f.z;

	uint64 ind;
	if (volumeDimensionOrder == 0)
		ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
	else
		ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

	if (x * x + y * y > rFOVsq)
	{
		f[ind] = 0.0;
		return;
	}

	const float x_width = 0.5f*T_f.x;
	const float y_width = 0.5f*T_f.y;
	//const float z_width = 0.5f*T_f.z;

	const float voxelToMagnifiedDetectorPixelRatio_u = T_f.x / T_g.z;
	//const float voxelToMagnifiedDetectorPixelRatio_v = T_f.z / T_g.y;

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	//const int searchWidth_v = 0;

	float val = 0.0f;
	for (int iphi = 0; iphi < N_g.x; iphi++)
	{
		const float cos_phi = cos(phis[iphi]);
		const float sin_phi = sin(phis[iphi]);

		float4 traj;
		traj.x = cos_phi;
		traj.y = sin_phi;
		//traj.z = 0.0;

		const int u_arg_mid = int(0.5 + (y*cos_phi - x * sin_phi - startVals_g.z) / T_g.z);
		const int v_arg_mid = int(0.5 + (z - startVals_g.y) / T_g.y);
		const int iv = v_arg_mid;

		for (int iu = u_arg_mid - searchWidth_u; iu <= u_arg_mid + searchWidth_u; iu++)
		{
			const float u = iu * T_g.z + startVals_g.z;
			float4 pos;
			pos.x = x + u * sin_phi;
			pos.y = y - u * cos_phi;
			//pos.z = z - v; // does not matter

			float t_max = 1.0e16;
			float t_min = -1.0e16;
			if (traj.x != 0.0f)
			{
				const float t_a = (pos.x + x_width) / traj.x;
				const float t_b = (pos.x - x_width) / traj.x;
				t_max = min(t_max, max(t_b, t_a));
				t_min = max(t_min, min(t_b, t_a));
			}
			if (traj.y != 0.0f)
			{
				const float t_a = (pos.y + y_width) / traj.y;
				const float t_b = (pos.y - y_width) / traj.y;
				t_max = min(t_max, max(t_b, t_a));
				t_min = max(t_min, min(t_b, t_a));
			}

			val += max(0.0f, t_max - t_min)*tex3D<float>(g, iu, iv, iphi);
		}
	}
	f[ind] = val;
}

__global__ void parallelBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* phis, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;

	float v = j * T_g.y + startVals_g.y;
	float u = k * T_g.z + startVals_g.z;

	float cos_phi = cos(phis[i]);
	float sin_phi = sin(phis[i]);
	float4 sourcePos;
	sourcePos.x = -u * sin_phi;
	sourcePos.y = u * cos_phi;
	sourcePos.z = v;

	float4 traj;
	traj.x = -cos_phi;
	traj.y = -sin_phi;
	traj.z = 0.0;

	if (volumeDimensionOrder == 0)
		g[i*N_g.y*N_g.z + j * N_g.z + k] = projectLine(f, N_f, T_f, startVals_f, sourcePos, traj);
	else
		g[i * N_g.y * N_g.z + j * N_g.z + k] = projectLine_ZYX(f, N_f, T_f, startVals_f, sourcePos, traj);
}

__global__ void modularBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	uint64 ind;
	if (volumeDimensionOrder == 0)
		ind = uint64(i) * uint64(N_f.y * N_f.z) + uint64(j * N_f.z + k);
	else
		ind = uint64(k) * uint64(N_f.y * N_f.x) + uint64(j * N_f.x + i);

	const float x = i * T_f.x + startVals_f.x;
	const float y = j * T_f.y + startVals_f.y;
	const float z = k * T_f.z + startVals_f.z;

	const float x_width = 0.5f*T_f.x;
	const float y_width = 0.5f*T_f.y;
	const float z_width = 0.5f*T_f.z;

	const float R = sqrt(sourcePositions[0] * sourcePositions[0] + sourcePositions[1] * sourcePositions[1] + sourcePositions[2] * sourcePositions[2]);
	const float D = R + sqrt(moduleCenters[0] * moduleCenters[0] + moduleCenters[1] * moduleCenters[1] + moduleCenters[2] * moduleCenters[2]);

	const float voxelToMagnifiedDetectorPixelRatio_u = T_f.x / (R / D * T_g.z);
	const float voxelToMagnifiedDetectorPixelRatio_v = T_f.z / (R / D * T_g.y);

	//int searchWidth_u = max(1, int(voxelToMagnifiedDetectorPixelRatio_u));
	//int searchWidth_v = max(1, int(voxelToMagnifiedDetectorPixelRatio_v));
	const int searchWidth_u = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_u);
	const int searchWidth_v = 1 + int(0.5f*voxelToMagnifiedDetectorPixelRatio_v);

	float val = 0.0f;
	for (int iphi = 0; iphi < N_g.x; iphi++)
	{
		float* sourcePosition = &sourcePositions[3 * iphi];
		float* moduleCenter = &moduleCenters[3 * iphi];
		float* v_vec = &rowVectors[3 * iphi];
		float* u_vec = &colVectors[3 * iphi];

		float4 pos;
		pos.x = x - sourcePosition[0];
		pos.y = y - sourcePosition[1];
		pos.z = z - sourcePosition[2];

		float4 c_minus_p;
		c_minus_p.x = moduleCenter[0] - sourcePosition[0];
		c_minus_p.y = moduleCenter[1] - sourcePosition[1];
		c_minus_p.z = moduleCenter[2] - sourcePosition[2];

		float4 v_cross_u;
		v_cross_u.x = v_vec[1] * u_vec[2] - v_vec[2] * u_vec[1];
		v_cross_u.y = v_vec[2] * u_vec[0] - v_vec[0] * u_vec[2];
		v_cross_u.z = v_vec[0] * u_vec[1] - v_vec[1] * u_vec[0];

		const float lineLength = (pos.x * v_cross_u.x + pos.y * v_cross_u.y + pos.z * v_cross_u.z) / (c_minus_p.x * v_cross_u.x + c_minus_p.y * v_cross_u.y + c_minus_p.z * v_cross_u.z);
		const float u_arg = (pos.x * u_vec[0] + pos.y * u_vec[1] + pos.z * u_vec[2]) / lineLength - (c_minus_p.x * u_vec[0] + c_minus_p.y * u_vec[1] + c_minus_p.z * u_vec[2]);
		const float v_arg = (pos.x * v_vec[0] + pos.y * v_vec[1] + pos.z * v_vec[2]) / lineLength - (c_minus_p.x * v_vec[0] + c_minus_p.y * v_vec[1] + c_minus_p.z * v_vec[2]);

		const int u_arg_mid = int(0.5 + (u_arg - startVals_g.z) / T_g.z);
		const int v_arg_mid = int(0.5 + (v_arg - startVals_g.y) / T_g.y);

		for (int iv = v_arg_mid - searchWidth_v; iv <= v_arg_mid + searchWidth_v; iv++)
		{
			const float v = iv * T_g.y + startVals_g.y;
			for (int iu = u_arg_mid - searchWidth_u; iu <= u_arg_mid + searchWidth_u; iu++)
			{
				const float u = iu * T_g.z + startVals_g.z;

				float4 traj;
				traj.x = c_minus_p.x + u * u_vec[0] + v * v_vec[0];
				traj.y = c_minus_p.y + u * u_vec[1] + v * v_vec[1];
				traj.z = c_minus_p.z + u * u_vec[2] + v * v_vec[2];
				const float trajMag_inv = rsqrt(traj.x * traj.x + traj.y * traj.y + traj.z * traj.z);
				traj.x *= trajMag_inv;
				traj.y *= trajMag_inv;
				traj.z *= trajMag_inv;

				float t_max = 1.0e16;
				float t_min = -1.0e16;
				if (traj.x != 0.0f)
				{
					const float t_a = (pos.x + x_width) / traj.x;
					const float t_b = (pos.x - x_width) / traj.x;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}
				if (traj.y != 0.0f)
				{
					const float t_a = (pos.y + y_width) / traj.y;
					const float t_b = (pos.y - y_width) / traj.y;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}
				if (traj.z != 0.0f)
				{
					const float t_a = (pos.z + z_width) / traj.z;
					const float t_b = (pos.z - z_width) / traj.z;
					t_max = min(t_max, max(t_b, t_a));
					t_min = max(t_min, min(t_b, t_a));
				}

				val += max(0.0f, t_max - t_min)*tex3D<float>(g, iu, iv, iphi);
			}
		}
	}
	f[ind] = val;
}

__global__ void modularBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float* sourcePositions, float* moduleCenters, float* rowVectors, float* colVectors, int volumeDimensionOrder)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;

	float v = j * T_g.y + startVals_g.y;
	float u = k * T_g.z + startVals_g.z;

	float4 sourcePos;
	sourcePos.x = sourcePositions[3 * i + 0];
	sourcePos.y = sourcePositions[3 * i + 1];
	sourcePos.z = sourcePositions[3 * i + 2];

	float4 traj;
	traj.x = moduleCenters[3 * i + 0] + v * rowVectors[3 * i + 0] + u * colVectors[3 * i + 0] - sourcePos.x;
	traj.y = moduleCenters[3 * i + 1] + v * rowVectors[3 * i + 1] + u * colVectors[3 * i + 1] - sourcePos.y;
	traj.z = moduleCenters[3 * i + 2] + v * rowVectors[3 * i + 2] + u * colVectors[3 * i + 2] - sourcePos.z;

	if (volumeDimensionOrder == 0)
		g[i*N_g.y*N_g.z + j * N_g.z + k] = projectLine(f, N_f, T_f, startVals_f, sourcePos, traj);
	else
		g[i * N_g.y * N_g.z + j * N_g.z + k] = projectLine_ZYX(f, N_f, T_f, startVals_f, sourcePos, traj);
}


//#########################################################################################
//#########################################################################################
bool project_Siddon(float*& g, float* f, parameters* params, bool data_on_cpu)
{
	if (params == NULL)
		return false;
	if (params->geometry == parameters::MODULAR)
		return project_modular(g, f, params, data_on_cpu);

	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate projection data on GPU
	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else
		dev_g = g;

	float* dev_phis = copyAngleArrayToGPU(params);

	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	bool useLinearInterpolation = false;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = NULL;
	/*
	if (data_on_cpu)
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	else
		dev_f = f;
	d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));
	//*/
	//*
	if (data_on_cpu)
		d_data_array = loadTexture_from_cpu(d_data_txt, f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));
	else
		d_data_array = loadTexture(d_data_txt, f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));
	//*/

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_g);
	dim3 dimGrid = setGridSize(N_g, dimBlock);
	if (params->geometry == parameters::CONE)
		coneBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, dev_phis, params->volumeDimensionOrder);
	else if (params->geometry == parameters::FAN)
		fanBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, dev_phis, params->volumeDimensionOrder);
	else if (params->geometry == parameters::PARALLEL)
		parallelBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_phis, params->volumeDimensionOrder);

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (data_on_cpu)
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	else
		g = dev_g;

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_Siddon(float* g, float*& f, parameters* params, bool data_on_cpu)
{
	if (params == NULL)
		return false;
	if (params->geometry == parameters::MODULAR)
		return backproject_modular(g, f, params, data_on_cpu);

	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate volume data on GPU
	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	float* dev_phis = copyAngleArrayToGPU(params);

	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

	float rFOVsq = params->rFOV() * params->rFOV();

	if (data_on_cpu)
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	else
		dev_g = g;

	bool useLinearInterpolation = false;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		dev_g = 0;
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else
		dev_f = f;

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_f);
	dim3 dimGrid = setGridSize(N_f, dimBlock);
	if (params->geometry == parameters::CONE)
		coneBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, rFOVsq, dev_phis, params->volumeDimensionOrder);
	else if (params->geometry == parameters::FAN)
		fanBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, rFOVsq, dev_phis, params->volumeDimensionOrder);
	else if (params->geometry == parameters::PARALLEL)
		parallelBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}
	if (data_on_cpu)
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	else
		f = dev_f;

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool project_fan(float*& g, float* f, parameters* params, bool data_on_cpu)
{
	return project_Siddon(g, f, params, data_on_cpu);
}

bool backproject_fan(float* g, float*& f, parameters* params, bool data_on_cpu)
{
	return backproject_Siddon(g, f, params, data_on_cpu);
}

bool project_cone(float *&g, float *f, parameters* params, bool data_on_cpu)
{
	return project_Siddon(g, f, params, data_on_cpu);
}

bool backproject_cone(float* g, float *&f, parameters* params, bool data_on_cpu)
{
	return backproject_Siddon(g, f, params, data_on_cpu);
}

bool project_parallel(float *&g, float* f, parameters* params, bool data_on_cpu)
{
	return project_Siddon(g, f, params, data_on_cpu);
}

bool backproject_parallel(float* g, float *&f, parameters* params, bool data_on_cpu)
{
	return backproject_Siddon(g, f, params, data_on_cpu);
}

bool project_modular(float *&g, float* f, parameters* params, bool data_on_cpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate projection data on GPU
	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N_g.x * N_g.y * N_g.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else
		dev_g = g;

	float *dev_sourcePositions = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_sourcePositions, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_sourcePositions, params->sourcePositions, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(sourcePositions) failed!\n");

	float *dev_moduleCenters = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_moduleCenters, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_moduleCenters, params->moduleCenters, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(moduleCenters) failed!\n");

	float *dev_rowVectors = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_rowVectors, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_rowVectors, params->rowVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(rowVectors) failed!\n");

	float *dev_colVectors = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_colVectors, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_colVectors, params->colVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(colVectors) failed!\n");

	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = NULL;
	/*
	if (data_on_cpu)
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	else
		dev_f = f;
	d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
	//*/
	//*
	if (data_on_cpu)
		d_data_array = loadTexture_from_cpu(d_data_txt, f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
	else
		d_data_array = loadTexture(d_data_txt, f, N_f, false, false, bool(params->volumeDimensionOrder == 1));
	//*/

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_g);
	dim3 dimGrid = setGridSize(N_g, dimBlock);
	modularBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (data_on_cpu)
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	else
		g = dev_g;

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_sourcePositions);
	cudaFree(dev_moduleCenters);
	cudaFree(dev_rowVectors);
	cudaFree(dev_colVectors);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_modular(float* g, float *&f, parameters* params, bool data_on_cpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	// Allocate volume data on GPU
	int4 N_f; float4 T_f; float4 startVal_f;
	setVolumeGPUparams(params, N_f, T_f, startVal_f);

	if (data_on_cpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N_f.x * N_f.y * N_f.z * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else
		dev_f = f;

	float *dev_sourcePositions = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_sourcePositions, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_sourcePositions, params->sourcePositions, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(sourcePositions) failed!\n");

	float *dev_moduleCenters = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_moduleCenters, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_moduleCenters, params->moduleCenters, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(moduleCenters) failed!\n");

	float *dev_rowVectors = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_rowVectors, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_rowVectors, params->rowVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(rowVectors) failed!\n");

	float *dev_colVectors = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_colVectors, 3 * params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_colVectors, params->colVectors, 3 * params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(colVectors) failed!\n");

	int4 N_g; float4 T_g; float4 startVal_g;
	setProjectionGPUparams(params, N_g, T_g, startVal_g, false);

	if (data_on_cpu)
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	else
		dev_g = g;

	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, false);

	// Call Kernel
	dim3 dimBlock = setBlockSize(N_f);
	dim3 dimGrid = setGridSize(N_f, dimBlock);
	modularBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (data_on_cpu)
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	else
		f = dev_f;

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_sourcePositions);
	cudaFree(dev_moduleCenters);
	cudaFree(dev_rowVectors);
	cudaFree(dev_colVectors);

	if (data_on_cpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}
