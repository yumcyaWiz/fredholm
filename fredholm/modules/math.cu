#pragma once

#include "sutil/vec_math.h"

// Duff, T., Burgess, J., Christensen, P., Hery, C., Kensler, A., Liani, M., &
// Villemin, R. (2017). Building an orthonormal basis, revisited. JCGT, 6(1).
static __forceinline__ __device__ void orthonormal_basis(const float3& normal,
                                                         float3& tangent,
                                                         float3& bitangent)
{
  float sign = copysignf(1.0f, normal.z);
  const float a = -1.0f / (sign + normal.z);
  const float b = normal.x * normal.y * a;
  tangent = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b,
                        -sign * normal.x);
  bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

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

static __forceinline__ __device__ float3 sqrt(const float3& v)
{
  return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
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
  while (idx_max > idx_min) {
    const int idx_mid = idx_min + (idx_max - idx_min) / 2;
    const T mid = values[idx_mid];
    if (value < mid) {
      idx_max = idx_mid - 1;
    } else if (value > mid) {
      idx_min = idx_mid + 1;
    } else {
      return idx_mid;
    }
  }
  return idx_min;
}