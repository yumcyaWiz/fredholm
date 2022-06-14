#include "sutil/vec_math.h"

static __forceinline__ __device__ void orthonormal_basis(const float3& n,
                                                         float3& t, float3& b)
{
  if (abs(n.y) < 0.9f) {
    t = normalize(cross(n, make_float3(0, 1, 0)));
  } else {
    t = normalize(cross(n, make_float3(0, 0, -1)));
  }
  b = normalize(cross(t, n));
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