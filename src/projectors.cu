////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// cuda module for projector
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "projectors.h"
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

	int ind;
	if (volumeDimensionOrder == 0)
		ind = i * N_f.y * N_f.z + j * N_f.z + k;
	else
		ind = k * N_f.y * N_f.x + j * N_f.x + i;

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

		const float v = iv * T_g.y + startVals_g.y;
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

	int ind;
	if (volumeDimensionOrder == 0)
		ind = i * N_f.y * N_f.z + j * N_f.z + k;
	else
		ind = k * N_f.y * N_f.x + j * N_f.x + i;

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

	int ind;
	if (volumeDimensionOrder == 0)
		ind = i * N_f.y * N_f.z + j * N_f.z + k;
	else
		ind = k * N_f.y * N_f.x + j * N_f.x + i;

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

	int ind;
	if (volumeDimensionOrder == 0)
		ind = i * N_f.y * N_f.z + j * N_f.z + k;
	else
		ind = k * N_f.y * N_f.x + j * N_f.x + i;

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

__global__ void AbelConeBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float tan_beta = sin_beta / cos_beta;
	const float sec_beta = 1.0f / cos_beta;

	const float r_unbounded = T_f.y*float(j) + startVals_f.y;
	const float r = fabs(r_unbounded);
	const float r_inner = r - 0.5f*T_f.y;
	const float r_outer = r + 0.5f*T_f.y;

	const float disc_shift_inner = (r_inner*r_inner - tau * tau)*sec_beta*sec_beta; // r_inner^2
	const float disc_shift_outer = (r_outer*r_outer - tau * tau)*sec_beta*sec_beta; // r_outer^2

	const float z = T_f.z*float(k) + startVals_f.z;
	const float Tv_inv = 1.0f / T_g.y;

	const float Z = (R + z * sin_beta)*sec_beta; // nominal value: R
	const float z_slope = (z + R * sin_beta)*sec_beta; // nominal value: 0

	int iu_min;
	int iu_max;
	if (r_unbounded < 0.0)
	{
		// left half
		iu_min = 0;
		iu_max = int(-startVals_g.z / T_g.z);
	}
	else
	{
		// right half
		iu_min = int(ceil(-startVals_g.z / T_g.z));
		iu_max = N_g.z - 1;
	}

	float curVal = 0.0;
	for (int iu = iu_min; iu <= iu_max; iu++)
	{
		const float u = fabs(T_g.z*float(iu) + startVals_g.z);
		float disc_outer = u * u*(r_outer*r_outer - Z * Z) + 2.0f*Z*sec_beta*tau*u + disc_shift_outer; // u^2*(r^2 - R^2) + r^2
		if (disc_outer > 0.0f)
		{
			const float b_ti = Z * sec_beta + tau * u;
			const float a_ti_inv = 1.0f / (u*u + sec_beta * sec_beta);
			float disc_inner = u * u*(r_inner*r_inner - Z * Z) + 2.0f*Z*sec_beta*tau*u + disc_shift_inner; // disc_outer > disc_inner
			if (disc_inner > 0.0f)
			{
				disc_inner = sqrt(disc_inner);
				disc_outer = sqrt(disc_outer);
				const float t_1st_low = (b_ti - disc_outer)*a_ti_inv;
				const float t_1st_high = (b_ti - disc_inner)*a_ti_inv;
				const float v_1st_arg = 2.0f*z_slope / (t_1st_low + t_1st_high) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_1st_arg * v_1st_arg) * (t_1st_high - t_1st_low) * tex3D<float>(g, float(iu) + 0.5f, (v_1st_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);

				const float t_2nd_low = (b_ti + disc_inner)*a_ti_inv;
				const float t_2nd_high = (b_ti + disc_outer)*a_ti_inv;
				const float v_2nd_arg = 2.0f*z_slope / (t_2nd_low + t_2nd_high) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_2nd_arg * v_2nd_arg) * (t_2nd_high - t_2nd_low) * tex3D<float>(g, float(iu) + 0.5f, (v_2nd_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
			else
			{
				disc_outer = sqrt(disc_outer);
				const float v_arg = z_slope / (b_ti*a_ti_inv) - tan_beta;
				curVal += sqrt(1.0f + tan_beta * tan_beta + u * u + v_arg * v_arg) * 2.0f*disc_outer*a_ti_inv * tex3D<float>(g, float(iu) + 0.5f, (v_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
		}
	}
	f[j * N_f.z + k] = curVal;
}

__device__ float AbelConeProjectorKernel_left(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int j, int k)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	const float v = float(j)*T_g.y + startVals_g.y;
	const float u = fabs(float(k)*T_g.z + startVals_g.z);

	const float X = cos_beta - v * sin_beta;

	const float sec_sq_plus_u_sq = X * X + u * u;
	const float b_ti = X * R*cos_beta + u * tau;
	const float a_ti_inv = 1.0f / sec_sq_plus_u_sq;
	const float disc_ti_shift = -(u*R*cos_beta - tau * X)*(u*R*cos_beta - tau * X);

	if (disc_ti_shift > 0.0f || fabs(sec_sq_plus_u_sq) < 1.0e-8)
		return 0.0f;

	float curVal = 0.0;
	const int N_r = int(0.5 + 0.5*N_f.y);

	const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	const float r_max = (float)(N_f.y - 1)*T_f.y + startVals_f.y;
	const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);

	const float z_shift = (-R * sin_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta + v * cos_beta) / T_f.z;

	int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
		int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		if (r_absoluteMinimum < r_max)
		{
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - rInd_min_minus) + 0.5f, 0.5f);
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - rInd_min_minus) + 0.5f, 0.5f);
		}
	}

	for (int ir = rInd_min; ir < N_r; ir++)
	{
		r_prev = (float)ir*T_f.y;
		const float r_next = (float)(ir + 1)*T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next*sec_sq_plus_u_sq);

		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - ir) + 0.5f, 0.5f);
		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - ir) + 0.5f, 0.5f);

		// update radius and sqrt for t calculation
		//r_prev = r_next;
		disc_sqrt_prev = disc_sqrt_next;
	}
	return curVal * sqrt(1.0f + u * u + v * v);
}

__device__ float AbelConeProjectorKernel_right(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau, int j, int k)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0;
		cos_beta = 1.0;
	}

	const float v = float(j)*T_g.y + startVals_g.y;
	const float u = fabs(float(k)*T_g.z + startVals_g.z);

	const float X = cos_beta - v * sin_beta;

	const float sec_sq_plus_u_sq = X * X + u * u;
	const float b_ti = X * R*cos_beta + u * tau;
	const float a_ti_inv = 1.0f / sec_sq_plus_u_sq;
	const float disc_ti_shift = -(u*R*cos_beta - tau * X)*(u*R*cos_beta - tau * X);

	if (disc_ti_shift > 0.0f || fabs(sec_sq_plus_u_sq) < 1.0e-8)
		return 0.0f;

	float curVal = 0.0;
	const int N_r = int(0.5 + 0.5*N_f.y);

	const int rInd_floor = N_r; // first valid index
	const float r_max = (float)(N_f.y - 1)*T_f.y + startVals_f.y;
	const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);

	const float z_shift = (-R * sin_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta + v * cos_beta) / T_f.z;

	int rInd_min = (int)ceil((sqrt(-disc_ti_shift / sec_sq_plus_u_sq)) / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev*sec_sq_plus_u_sq);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = sqrt(-disc_ti_shift / sec_sq_plus_u_sq);
		int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		if (r_absoluteMinimum < r_max)
		{
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + rInd_min_minus) + 0.5f, 0.5f);
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + rInd_min_minus) + 0.5f, 0.5f);
		}
	}

	for (int ir = rInd_min; ir < N_r; ir++)
	{
		r_prev = (float)ir*T_f.y;
		const float r_next = (float)(ir + 1)*T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next*sec_sq_plus_u_sq);

		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + ir) + 0.5f, 0.5f);
		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + ir) + 0.5f, 0.5f);

		// update radius and sqrt for t calculation
		//r_prev = r_next;
		disc_sqrt_prev = disc_sqrt_next;
	}
	return curVal * sqrt(1.0f + u * u + v * v);
	//return 0.0f;
}

__global__ void AbelConeProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float R, float D, float axisOfSymmetry, float tau)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;
	else if (float(k)*T_g.z + startVals_g.z < 0.0)
		g[j * N_g.z + k] = AbelConeProjectorKernel_left(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, R, D, axisOfSymmetry, tau, j, k);
	else
		g[j * N_g.z + k] = AbelConeProjectorKernel_right(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, R, D, axisOfSymmetry, tau, j, k);
}

__device__ float AbelParallelBeamProjectorKernel_left(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int j, int k)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float sec_beta = 1.0f / cos_beta;

	const float v = (float)j*T_g.y + startVals_g.y;
	const float u = fabs((float)k*T_g.z + startVals_g.z);

	const float b_ti = v*sin_beta;
	const float a_ti_inv = sec_beta;
	const float disc_ti_shift = -u*u;

	float curVal = 0.0;
	const int N_r = int(0.5 + 0.5*N_f.y);

	const int rInd_floor = int(0.5 - startVals_f.y / T_f.y); // first valid index
	const float r_max = (float)(N_f.y-1)*T_f.y+ startVals_f.y;
	const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);

	const float z_shift = (v*cos_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta) / T_f.z;

	int rInd_min = (int)ceil(u/T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev*r_prev < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev*r_prev);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = u;
		int rInd_min_minus = max(0, min(N_r-1, (int)(ceil(r_absoluteMinimum/T_f.y-1.0f))));
		if (r_absoluteMinimum < r_max)
		{
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - rInd_min_minus) + 0.5f, 0.5f);
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - rInd_min_minus) + 0.5f, 0.5f);
		}
	}

	for (int ir = rInd_min; ir < N_r; ir++)
	{
		r_prev = ir*T_f.y;
		disc_sqrt_prev = sqrt(disc_ti_shift + r_prev*r_prev);
		const float r_next = (ir+1)*T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next*r_next);

		curVal += (disc_sqrt_next-disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - ir) + 0.5f, 0.5f);
		curVal += (disc_sqrt_next-disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r - 1 - ir) + 0.5f, 0.5f);
	}
	return curVal;
}

__device__ float AbelParallelBeamProjectorKernel_right(int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry, int j, int k)
{
	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float sec_beta = 1.0f / cos_beta;

	const float v = (float)j*T_g.y + startVals_g.y;
	const float u = fabs((float)k*T_g.z + startVals_g.z);

	const float b_ti = v * sin_beta;
	const float a_ti_inv = sec_beta;
	const float disc_ti_shift = -u * u;

	float curVal = 0.0;
	const int N_r = int(0.5 + 0.5*N_f.y);

	const int rInd_floor = N_r; // first valid index
	const float r_max = (float)(N_f.y - 1)*T_f.y + startVals_f.y;
	const float r_min = fabs((float)(rInd_floor)*T_f.y + startVals_f.y);

	const float z_shift = (v*cos_beta - startVals_f.z) / T_f.z;
	const float z_slope = (sin_beta) / T_f.z;

	int rInd_min = (int)ceil(u / T_f.y);
	float r_prev = (float)(rInd_min)*T_f.y;
	if (disc_ti_shift + r_prev * r_prev < 0.0f)
	{
		rInd_min = rInd_min + 1;
		r_prev = (float)(rInd_min)*T_f.y;
	}
	float disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev);

	// Go back one sample and check
	if (rInd_min >= 1)
	{
		const float r_absoluteMinimum = u;
		int rInd_min_minus = max(0, min(N_r - 1, (int)(ceil(r_absoluteMinimum / T_f.y - 1.0f))));
		if (r_absoluteMinimum < r_max)
		{
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + rInd_min_minus) + 0.5f, 0.5f);
			curVal += max(0.0f, disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + rInd_min_minus) + 0.5f, 0.5f);
		}
	}

	for (int ir = rInd_min; ir < N_r; ir++)
	{
		r_prev = ir * T_f.y;
		disc_sqrt_prev = sqrt(disc_ti_shift + r_prev * r_prev);
		const float r_next = (ir + 1)*T_f.y;
		const float disc_sqrt_next = sqrt(disc_ti_shift + r_next * r_next);

		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti - 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + ir) + 0.5f, 0.5f);
		curVal += (disc_sqrt_next - disc_sqrt_prev)*a_ti_inv*tex3D<float>(f, (b_ti + 0.5f*(disc_sqrt_next + disc_sqrt_prev))*a_ti_inv*z_slope + z_shift + 0.5f, (float)(N_r + ir) + 0.5f, 0.5f);
	}
	return curVal;
}

__global__ void AbelParallelBeamProjectorKernel(float* g, int4 N_g, float4 T_g, float4 startVals_g, cudaTextureObject_t f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_g.x || j >= N_g.y || k >= N_g.z)
		return;
	else if (float(k)*T_g.z + startVals_g.z < 0.0)
		g[j * N_g.z + k] = AbelParallelBeamProjectorKernel_left(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, axisOfSymmetry, j, k);
	else
		g[j * N_g.z + k] = AbelParallelBeamProjectorKernel_right(N_g, T_g, startVals_g, f, N_f, T_f, startVals_f, axisOfSymmetry, j, k);
}

__global__ void AbelParallelBeamBackprojectorKernel(cudaTextureObject_t g, int4 N_g, float4 T_g, float4 startVals_g, float* f, int4 N_f, float4 T_f, float4 startVals_f, float axisOfSymmetry)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	const int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i >= N_f.x || j >= N_f.y || k >= N_f.z)
		return;

	float cos_beta = cos(axisOfSymmetry);
	float sin_beta = sin(axisOfSymmetry);
	if (fabs(sin_beta) < 1.0e-4)
	{
		sin_beta = 0.0f;
		cos_beta = 1.0f;
	}
	const float tan_beta = sin_beta/cos_beta;
	const float sec_beta = 1.0f/cos_beta;

	const float r_unbounded = T_f.y*(float)j + startVals_f.y;
	const float r = fabs(r_unbounded);
	const float r_inner = r - 0.5f*T_f.y;
	const float r_outer = r + 0.5f*T_f.y;

	const float disc_shift_inner = r_inner*r_inner; // r_inner^2
	const float disc_shift_outer = r_outer*r_outer; // r_outer^2

	const float z = T_f.z*(float)k + startVals_f.z;
	const float Tv_inv = 1.0f/T_g.y;

	int iu_min;
	int iu_max;
	if (r_unbounded < 0.0)
	{
		// left half
		iu_min = 0;
		iu_max = int(-startVals_g.z / T_g.z);
	}
	else
	{
		// right half
		iu_min = int(ceil(-startVals_g.z / T_g.z));
		iu_max = N_g.z - 1;
	}

	float curVal = 0.0;
	for (int iu = iu_min; iu <= iu_max; iu++)
	{
		const float u = fabs(T_g.z*(float)iu + startVals_g.z);
		float disc_outer = disc_shift_outer - u*u; // u^2*(r^2 - R^2) + r^2
		if (disc_outer > 0.0f)
		{
			const float b_ti = z*tan_beta;
			const float a_ti_inv = cos_beta;
			float disc_inner = disc_shift_inner - u*u; // disc_outer > disc_inner
			if (disc_inner > 0.0f)
			{
				disc_inner = sqrt(disc_inner);
				disc_outer = sqrt(disc_outer);
				const float t_1st_low = (b_ti-disc_outer)*a_ti_inv;
				const float t_1st_high = (b_ti-disc_inner)*a_ti_inv;
				const float v_1st_arg = -0.5f*(t_1st_low+t_1st_high)*tan_beta + z*sec_beta;
				curVal += (t_1st_high - t_1st_low) * tex3D<float>(g, float(iu) + 0.5f, (v_1st_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);

				const float t_2nd_low = (b_ti+disc_inner)*a_ti_inv;
				const float t_2nd_high = (b_ti+disc_outer)*a_ti_inv;
				const float v_2nd_arg = -0.5f*(t_2nd_low+t_2nd_high)*tan_beta + z*sec_beta;
				curVal += (t_2nd_high - t_2nd_low) * tex3D<float>(g, float(iu) + 0.5f, (v_2nd_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
			else
			{
				 disc_outer = sqrt(disc_outer);
				const float v_arg = -(b_ti*a_ti_inv)*tan_beta + z*sec_beta;
				curVal += 2.0f*disc_outer*a_ti_inv * tex3D<float>(g, float(iu) + 0.5f, (v_arg - startVals_g.y) * Tv_inv + 0.5f, 0.5f);
			}
		}
	}
	f[j * N_f.z + k] = curVal * sqrt(1.0f + tan_beta * tan_beta);
}


//#########################################################################################
//#########################################################################################
bool project_fan(float*& g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate planogram data on GPU
	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	int N = N_g.x * N_g.y * N_g.z;
	if (cpu_to_gpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else
	{
		dev_g = g;
	}

	float* dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	if (cpu_to_gpu)
	{
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	}
	else
	{
		dev_f = f;
	}

	bool useLinearInterpolation = false;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_g.x) / double(dimBlock.x))), int(ceil(double(N_g.y) / double(dimBlock.y))), int(ceil(double(N_g.z) / double(dimBlock.z))));
	fanBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, dev_phis, params->volumeDimensionOrder);
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu)
	{
		//printf("pulling projections off GPU...\n");
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	}
	else
	{
		g = dev_g;
	}

	/*
	float maxVal_g = g[0];
	for (int i = 0; i < N; i++)
		maxVal_g = std::max(maxVal_g, g[i]);
	printf("max g: %f\n", maxVal_g);
	float maxVal_f = f[0];
	for (int i = 0; i < N_f.x*N_f.y*N_f.z; i++)
		maxVal_f = std::max(maxVal_f, f[i]);
	printf("max f: %f\n", maxVal_f);
	//*/

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_fan(float* g, float*& f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate volume data on GPU
	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	int N = N_f.x * N_f.y * N_f.z;
	if (cpu_to_gpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else
	{
		dev_f = f;
	}

	float* dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	float rFOVsq = params->rFOV() * params->rFOV();

	if (cpu_to_gpu)
	{
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	}
	else
	{
		dev_g = g;
	}

	bool useLinearInterpolation = false;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_f.x) / double(dimBlock.x))), int(ceil(double(N_f.y) / double(dimBlock.y))), int(ceil(double(N_f.z) / double(dimBlock.z))));
	fanBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, rFOVsq, dev_phis, params->volumeDimensionOrder);
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}
	if (cpu_to_gpu)
	{
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	}
	else
	{
		f = dev_f;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu)
	{
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool project_cone(float *&g, float *f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate planogram data on GPU
	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	if (params->isSymmetric())
	{
		T_g.y = T_g.y / params->sdd;
		T_g.z = T_g.z / params->sdd;
		startVal_g.y = startVal_g.y / params->sdd;
		startVal_g.z = startVal_g.z / params->sdd;
	}

	int N = N_g.x * N_g.y * N_g.z;
	if (cpu_to_gpu)
	{
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else
	{
		dev_g = g;
	}

	float *dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	if (cpu_to_gpu) {
		//printf("copying volume to GPU...\n");
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	}
	else {
		dev_f = f;
	}

	bool useLinearInterpolation = false;
	if (params->isSymmetric())
		useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_g.x) / double(dimBlock.x))), int(ceil(double(N_g.y) / double(dimBlock.y))), int(ceil(double(N_g.z) / double(dimBlock.z))));
	if (params->isSymmetric())
	{
		//printf("about to call: AbelConeProjectorKernel...\n");
		AbelConeProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, params->axisOfSymmetry*PI / 180.0, 0.0);
	}
	else
	{
		coneBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->sod, params->sdd, dev_phis, params->volumeDimensionOrder);
	}
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu) {
		//printf("pulling projections off GPU...\n");
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	}
	else {
		g = dev_g;
	}

	/*
	float maxVal_g = g[0];
	for (int i = 0; i < N; i++)
		maxVal_g = std::max(maxVal_g, g[i]);
	printf("max g: %f\n", maxVal_g);
	float maxVal_f = f[0];
	for (int i = 0; i < N_f.x*N_f.y*N_f.z; i++)
		maxVal_f = std::max(maxVal_f, f[i]);
	printf("max f: %f\n", maxVal_f);
	//*/

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_cone(float* g, float *&f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate volume data on GPU
	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	int N = N_f.x * N_f.y * N_f.z;
	if (cpu_to_gpu) {
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else {
		dev_f = f;
	}

	float *dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	float rFOVsq = params->rFOV()*params->rFOV();

	if (params->isSymmetric())
	{
		T_g.y = T_g.y / params->sdd;
		T_g.z = T_g.z / params->sdd;
		startVal_g.y = startVal_g.y / params->sdd;
		startVal_g.z = startVal_g.z / params->sdd;
	}

	if (cpu_to_gpu) {
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	}
	else {
		dev_g = g;
	}

	bool useLinearInterpolation = false;
	if (params->isSymmetric())
		useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_f.x) / double(dimBlock.x))), int(ceil(double(N_f.y) / double(dimBlock.y))), int(ceil(double(N_f.z) / double(dimBlock.z))));
	if (params->isSymmetric())
	{
		AbelConeBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, params->axisOfSymmetry*PI / 180.0, 0.0);
	}
	else
	{
		coneBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->sod, params->sdd, rFOVsq, dev_phis, params->volumeDimensionOrder);
	}
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}
	if (cpu_to_gpu) {
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	}
	else {
		f = dev_f;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool project_parallel(float *&g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate planogram data on GPU
	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	int N = N_g.x * N_g.y * N_g.z;
	if (cpu_to_gpu) {
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else {
		dev_g = g;
	}

	float *dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	if (cpu_to_gpu) {
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	}
	else {
		dev_f = f;
	}

	bool useLinearInterpolation = false;
	if (params->isSymmetric())
		useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, useLinearInterpolation, bool(params->volumeDimensionOrder == 1));

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_g.x) / double(dimBlock.x))), int(ceil(double(N_g.y) / double(dimBlock.y))), int(ceil(double(N_g.z) / double(dimBlock.z))));
	if (params->isSymmetric())
	{
		AbelParallelBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, params->axisOfSymmetry*PI/180.0);
	}
	else
	{
		parallelBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_phis, params->volumeDimensionOrder);
	}
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu) {
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	}
	else {
		g = dev_g;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_parallel(float* g, float *&f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate volume data on GPU
	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	int N = N_f.x * N_f.y * N_f.z;
	if (cpu_to_gpu) {
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else {
		dev_f = f;
	}

	float *dev_phis = 0;
	if (cudaSuccess != cudaMalloc((void**)&dev_phis, params->numAngles * sizeof(float)))
		fprintf(stderr, "cudaMalloc failed!\n");
	if (cudaMemcpy(dev_phis, params->phis, params->numAngles * sizeof(float), cudaMemcpyHostToDevice))
		fprintf(stderr, "cudaMemcpy(phis) failed!\n");

	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	float rFOVsq = params->rFOV()*params->rFOV();

	if (cpu_to_gpu) {
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	}
	else {
		dev_g = g;
	}

	bool useLinearInterpolation = false;
	if (params->isSymmetric())
		useLinearInterpolation = true;
	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, useLinearInterpolation);

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_f.x) / double(dimBlock.x))), int(ceil(double(N_f.y) / double(dimBlock.y))), int(ceil(double(N_f.z) / double(dimBlock.z))));
	if (params->isSymmetric())
	{
		AbelParallelBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, params->axisOfSymmetry*PI/180.0);
	}
	else
	{
		parallelBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, rFOVsq, dev_phis, params->volumeDimensionOrder);
	}
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu) {
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	}
	else {
		f = dev_f;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_phis);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool project_modular(float *&g, float* f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate planogram data on GPU
	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	int N = N_g.x * N_g.y * N_g.z;
	if (cpu_to_gpu) {
		if ((cudaStatus = cudaMalloc((void**)&dev_g, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(projections) failed!\n");
		}
	}
	else {
		dev_g = g;
	}

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

	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	if (cpu_to_gpu) {
		dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
	}
	else {
		dev_f = f;
	}

	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_f, N_f, false, false, bool(params->volumeDimensionOrder == 1));

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_g.x) / double(dimBlock.x))), int(ceil(double(N_g.y) / double(dimBlock.y))), int(ceil(double(N_g.z) / double(dimBlock.z))));
	modularBeamProjectorKernel <<< dimGrid, dimBlock >>> (dev_g, N_g, T_g, startVal_g, d_data_txt, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu) {
		pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
	}
	else {
		g = dev_g;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_sourcePositions);
	cudaFree(dev_moduleCenters);
	cudaFree(dev_rowVectors);
	cudaFree(dev_colVectors);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}

bool backproject_modular(float* g, float *&f, parameters* params, bool cpu_to_gpu)
{
	if (g == NULL || f == NULL || params == NULL || params->allDefined() == false)
		return false;

	cudaSetDevice(params->whichGPU);
	cudaError_t cudaStatus;

	float* dev_g = 0;
	float* dev_f = 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Allocate volume data on GPU
	int4 N_f; N_f.x = params->numX; N_f.y = params->numY; N_f.z = params->numZ;
	float4 T_f; T_f.x = params->voxelWidth; T_f.y = params->voxelWidth; T_f.z = params->voxelHeight;
	float4 startVal_f; startVal_f.x = params->x_0(); startVal_f.y = params->y_0(); startVal_f.z = params->z_0();

	int N = N_f.x * N_f.y * N_f.z;
	if (cpu_to_gpu) {
		if ((cudaStatus = cudaMalloc((void**)&dev_f, N * sizeof(float))) != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc(volume) failed!\n");
		}
	}
	else {
		dev_f = f;
	}

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

	int4 N_g; N_g.x = params->numAngles; N_g.y = params->numRows; N_g.z = params->numCols;
	float4 T_g; T_g.x = params->T_phi(); T_g.y = params->pixelHeight; T_g.z = params->pixelWidth;
	float4 startVal_g; startVal_g.x = params->phis[0]; startVal_g.y = params->v_0(); startVal_g.z = params->u_0();

	if (cpu_to_gpu) {
		dev_g = copyProjectionDataToGPU(g, params, params->whichGPU);
	}
	else {
		dev_g = g;
	}

	cudaTextureObject_t d_data_txt = NULL;
	cudaArray* d_data_array = loadTexture(d_data_txt, dev_g, N_g, false, false);

	//* call kernel: FIXME!
	dim3 dimBlock(8, 8, 8); // best so far
	dim3 dimGrid(int(ceil(double(N_f.x) / double(dimBlock.x))), int(ceil(double(N_f.y) / double(dimBlock.y))), int(ceil(double(N_f.z) / double(dimBlock.z))));
	modularBeamBackprojectorKernel <<< dimGrid, dimBlock >>> (d_data_txt, N_g, T_g, startVal_g, dev_f, N_f, T_f, startVal_f, dev_sourcePositions, dev_moduleCenters, dev_rowVectors, dev_colVectors, params->volumeDimensionOrder);
	//*/

	// pull result off GPU
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "kernel failed!\n");
		fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
		fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
	}

	if (cpu_to_gpu) {
		pullVolumeDataFromGPU(f, params, dev_f, params->whichGPU);
	}
	else {
		f = dev_f;
	}

	// Clean up
	cudaFreeArray(d_data_array);
	cudaDestroyTextureObject(d_data_txt);
	cudaFree(dev_sourcePositions);
	cudaFree(dev_moduleCenters);
	cudaFree(dev_rowVectors);
	cudaFree(dev_colVectors);

	if (cpu_to_gpu) {
		if (dev_g != 0)
			cudaFree(dev_g);
		if (dev_f != 0)
			cudaFree(dev_f);
	}

	return true;
}
