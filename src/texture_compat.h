////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Texture backend abstraction.
//
// LEAP samples volumes and projections through CUDA texture objects, relying on
// the texture units for hardware (bi/tri)linear interpolation and boundary
// handling. Some accelerators (notably AMD CDNA APUs such as the MI300A) expose
// no texture memory. This header provides a drop-in software emulation of the
// texture fetches so the exact same kernels run on those devices.
//
// Selection is entirely compile-time:
//   * default (NVIDIA / any GPU with texture memory): TEX_DATA is a
//     cudaTextureObject_t and TEX3D/TEX1D expand to the hardware tex3D/tex1D
//     intrinsics. The generated device code is byte-for-byte identical to
//     calling tex3D<float> directly -- there is no wrapper and no overhead.
//   * __USE_NOTEX: TEX_DATA becomes a small handle that carries a global-memory
//     pointer plus the sampling metadata (extents, addressing mode, filter
//     mode). TEX3D/TEX1D perform the interpolation in software.
//
// The handle stores its extents in "extent order" (component 0 varies fastest,
// matching the width axis of the cudaArray the hardware path allocates). The
// loaders in cuda_utils.cu are responsible for populating the handle so that
// TEX3D(t, x, y, z) reads data[z*n1*n0 + y*n0 + x] -- the same element the
// hardware path returns for tex3D(t, x, y, z).
////////////////////////////////////////////////////////////////////////////////
#ifndef __TEXTURE_COMPAT_H
#define __TEXTURE_COMPAT_H

#pragma once

#ifndef __USE_CPU
#include "cuda_runtime.h"

#ifndef __USE_NOTEX
//==============================================================================
// Hardware texture path (default). Zero-cost aliases over the CUDA intrinsics.
//==============================================================================
typedef cudaTextureObject_t TEX_DATA;
typedef cudaArray*          TEX_ARRAY;

#define TEX3D(tex, x, y, z) tex3D<float>((tex), (x), (y), (z))
#define TEX1D(tex, x)       tex1D<float>((tex), (x))

#else
//==============================================================================
// Software emulation path (__USE_NOTEX): sample from global memory.
//==============================================================================

// Addressing (extrapolation) and filtering modes, mirroring cudaTextureDesc.
enum texAddressMode { TEX_BORDER = 0, TEX_CLAMP = 1 }; // border -> 0 outside, clamp -> edge value
enum texFilterMode  { TEX_NEAREST = 0, TEX_LINEAR = 1 };

// Software texture handle. Passed by value into kernels exactly where a
// cudaTextureObject_t used to be. Deliberately small (a pointer + 5 ints) so it
// costs a handful of registers.
struct softTexture
{
    const float* data; // device pointer, extent-order layout (axis 0 fastest)
    int n0;            // extent of the fastest-varying axis (cudaArray width)
    int n1;            // extent of the middle axis
    int n2;            // extent of the slowest-varying axis (cudaArray depth); 1 for 1D
    int address;       // texAddressMode
    int filter;        // texFilterMode
    int owns;          // 1 if freeTexture should cudaFree(data), 0 when aliasing caller memory
};

typedef softTexture TEX_DATA;
typedef float*      TEX_ARRAY; // owns the backing buffer, or NULL when aliasing caller memory

#define TEX3D(tex, x, y, z) leap_tex3D((tex), (x), (y), (z))
#define TEX1D(tex, x)       leap_tex1D((tex), (x))

#ifndef __LEAP_MIN
#define __LEAP_MIN(a, b) ((a) < (b) ? (a) : (b))
#define __LEAP_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Read-only cache load on device; plain dereference when parsed by the host
// compiler (__CUDA_ARCH__ is only defined during device compilation).
template <typename T>
__host__ __device__ __forceinline__ T leap_ldg(const T* ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// Per-axis linear sampling factors matching CUDA's non-normalized linear filter:
// the texel center of index i sits at coordinate i + 0.5, so the lower neighbor
// is floor(coord - 0.5).
struct leapAxisWeights { int i0; int i1; float w0; float w1; };

__host__ __device__ __forceinline__ leapAxisWeights leap_axis_linear(float coord, int n, int address)
{
    const float c = coord - 0.5f;
    const float f = floorf(c);
    int i0 = (int)f;
    int i1 = i0 + 1;
    float a = c - f;             // fractional distance in [0, 1)
    float w0 = 1.0f - a;
    float w1 = a;

    if (address == TEX_CLAMP)
    {
        i0 = __LEAP_MIN(__LEAP_MAX(i0, 0), n - 1);
        i1 = __LEAP_MIN(__LEAP_MAX(i1, 0), n - 1);
    }
    else // TEX_BORDER: out-of-range neighbors contribute 0 (per-neighbor, as in hardware)
    {
        if (i0 < 0 || i0 >= n) { w0 = 0.0f; i0 = __LEAP_MIN(__LEAP_MAX(i0, 0), n - 1); }
        if (i1 < 0 || i1 >= n) { w1 = 0.0f; i1 = __LEAP_MIN(__LEAP_MAX(i1, 0), n - 1); }
    }

    leapAxisWeights w;
    w.i0 = i0; w.i1 = i1; w.w0 = w0; w.w1 = w1;
    return w;
}

// Per-axis nearest sampling. Returns the (clamped-for-safe-load) index and sets
// inRange=false for border mode when the coordinate falls outside the volume.
__host__ __device__ __forceinline__ int leap_axis_nearest(float coord, int n, int address, bool& inRange)
{
    int i = (int)floorf(coord);
    if (address == TEX_CLAMP)
    {
        inRange = true;
        return __LEAP_MIN(__LEAP_MAX(i, 0), n - 1);
    }
    inRange = (i >= 0 && i < n);
    return __LEAP_MIN(__LEAP_MAX(i, 0), n - 1);
}

//------------------------------------------------------------------------------
// 3D fetch. (x, y, z) index axes (0, 1, 2) i.e. (fastest, middle, slowest),
// exactly like tex3D<float>(tex, x, y, z) against the hardware cudaArray.
//------------------------------------------------------------------------------
__host__ __device__ __forceinline__ float leap_tex3D(const softTexture& tex, float x, float y, float z)
{
    if (isnan(x) || isnan(y) || isnan(z))
        return 0.0f;

    const float* data = tex.data;
    const int n0 = tex.n0, n1 = tex.n1, n2 = tex.n2;
    const long long s1 = (long long)n0;       // stride of axis 1
    const long long s2 = (long long)n0 * n1;  // stride of axis 2

    if (tex.filter == TEX_LINEAR)
    {
        const leapAxisWeights ax = leap_axis_linear(x, n0, tex.address);
        const leapAxisWeights ay = leap_axis_linear(y, n1, tex.address);
        const leapAxisWeights az = leap_axis_linear(z, n2, tex.address);

        const long long b0 = az.i0 * s2, b1 = az.i1 * s2;
        const long long r00 = b0 + ay.i0 * s1, r10 = b0 + ay.i1 * s1;
        const long long r01 = b1 + ay.i0 * s1, r11 = b1 + ay.i1 * s1;

        const float c000 = leap_ldg(&data[r00 + ax.i0]);
        const float c100 = leap_ldg(&data[r00 + ax.i1]);
        const float c010 = leap_ldg(&data[r10 + ax.i0]);
        const float c110 = leap_ldg(&data[r10 + ax.i1]);
        const float c001 = leap_ldg(&data[r01 + ax.i0]);
        const float c101 = leap_ldg(&data[r01 + ax.i1]);
        const float c011 = leap_ldg(&data[r11 + ax.i0]);
        const float c111 = leap_ldg(&data[r11 + ax.i1]);

        // Interpolate along axis 0, then 1, then 2 using fused multiply-adds.
        const float c00 = fmaf(ax.w1, c100, ax.w0 * c000);
        const float c10 = fmaf(ax.w1, c110, ax.w0 * c010);
        const float c01 = fmaf(ax.w1, c101, ax.w0 * c001);
        const float c11 = fmaf(ax.w1, c111, ax.w0 * c011);

        const float c0 = fmaf(ay.w1, c10, ay.w0 * c00);
        const float c1 = fmaf(ay.w1, c11, ay.w0 * c01);

        return fmaf(az.w1, c1, az.w0 * c0);
    }
    else
    {
        bool vx, vy, vz;
        const int i = leap_axis_nearest(x, n0, tex.address, vx);
        const int j = leap_axis_nearest(y, n1, tex.address, vy);
        const int k = leap_axis_nearest(z, n2, tex.address, vz);
        if (!(vx && vy && vz))
            return 0.0f;
        return leap_ldg(&data[k * s2 + j * s1 + i]);
    }
}

//------------------------------------------------------------------------------
// 1D fetch, matching tex1D<float>(tex, x).
//------------------------------------------------------------------------------
__host__ __device__ __forceinline__ float leap_tex1D(const softTexture& tex, float x)
{
    if (isnan(x))
        return 0.0f;

    const float* data = tex.data;
    const int n0 = tex.n0;

    if (tex.filter == TEX_LINEAR)
    {
        const leapAxisWeights ax = leap_axis_linear(x, n0, tex.address);
        return fmaf(ax.w1, leap_ldg(&data[ax.i1]), ax.w0 * leap_ldg(&data[ax.i0]));
    }
    else
    {
        bool vx;
        const int i = leap_axis_nearest(x, n0, tex.address, vx);
        return vx ? leap_ldg(&data[i]) : 0.0f;
    }
}

#endif // __USE_NOTEX
#endif // __USE_CPU

#endif // __TEXTURE_COMPAT_H
