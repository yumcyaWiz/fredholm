#pragma once

#include "helper_math.h"

static __forceinline__ __device__ float3 world_to_local(const float3& v,
                                                        const float3& t,
                                                        const float3& n,
                                                        const float3& b)
{
    return make_float3(dot(v, t), dot(v, n), dot(v, b));
}

static __forceinline__ __device__ float3 local_to_world(const float3& v,
                                                        const float3& t,
                                                        const float3& n,
                                                        const float3& b)
{
    return make_float3(v.x * t.x + v.y * n.x + v.z * b.x,
                       v.x * t.y + v.y * n.y + v.z * b.y,
                       v.x * t.z + v.y * n.z + v.z * b.z);
}

static __forceinline__ __device__ float length2(const float3& v)
{
    return dot(v, v);
}

static __forceinline__ __device__ float3 square(const float3& v)
{
    return make_float3(v.x * v.x, v.y * v.y, v.z * v.z);
}

static __forceinline__ __device__ float3 sqrtf(const float3& v)
{
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

static __forceinline__ __device__ float3 sinf(const float3& v)
{
    return make_float3(sinf(v.x), sinf(v.y), sinf(v.z));
}

static __forceinline__ __device__ float3 cosf(const float3& v)
{
    return make_float3(cosf(v.x), cosf(v.y), cosf(v.z));
}

static __forceinline__ float3 tanf(const float3& v)
{
    return make_float3(tanf(v.x), tanf(v.y), tanf(v.z));
}

static __forceinline__ __device__ float3 atan2f(const float3& v1,
                                                const float3& v2)
{
    return make_float3(atan2f(v1.x, v2.x), atan2f(v1.y, v2.y),
                       atan2f(v1.z, v2.z));
}

static __forceinline__ __device__ bool isnan(const float3& v)
{
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

static __forceinline__ __device__ bool isinf(const float3& v)
{
    return isinf(v.x) || isinf(v.y) || isinf(v.z);
}

static __forceinline__ __device__ float deg_to_rad(float deg)
{
    return deg * M_PIf / 180.0f;
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float rgb_to_luminance(const float3& rgb)
{
    return dot(rgb, make_float3(0.2126729f, 0.7151522f, 0.0721750f));
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float3 rgb_to_xyz(const float3& rgb)
{
    return make_float3(
        dot(rgb, make_float3(0.4887180f, 0.3106803f, 0.2006017f)),
        dot(rgb, make_float3(0.1762044f, 0.8129847f, 0.0108109f)),
        dot(rgb, make_float3(0.0000000f, 0.0102048f, 0.9897952f)));
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float3 xyz_to_rgb(const float3& xyz)
{
    return make_float3(dot(xyz, make_float3(2.3706743, -0.9000405, -0.4706338)),
                       dot(xyz, make_float3(-0.5138850, 1.4253036, 0.0885814)),
                       dot(xyz, make_float3(0.0052982, -0.0146949, 1.0093968)));
}

static __forceinline__ __device__ float2 cartesian_to_spherical(const float3& w)
{
    float2 ret;
    ret.x = acosf(clamp(w.y, -1.0f, 1.0f));
    ret.y = atan2f(w.z, w.x);
    if (ret.y < 0) ret.y += 2.0f * M_PIf;
    return ret;
}

template <typename T>
static __forceinline__ __device__ int binary_search(T* values, int size,
                                                    float value)
{
    int idx_min = 0;
    int idx_max = size - 1;
    while (idx_max >= idx_min)
    {
        const int idx_mid = (idx_min + idx_max) / 2;
        const T mid = values[idx_mid];
        if (value < mid) { idx_max = idx_mid - 1; }
        else if (value > mid) { idx_min = idx_mid + 1; }
        else { return idx_mid; }
    }
    return idx_max;
}