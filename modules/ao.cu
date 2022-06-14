#include <optix.h>

#include "sampling.cu"
#include "shared.h"
#include "sutil/vec_math.h"

#define RAY_EPS 0.01

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

extern "C" {
__constant__ LaunchParams params;
}

enum class RayType : unsigned int {
  RAY_TYPE_RADIANCE = 0,
  RAY_TYPE_SHADOW = 1,
  RAY_TYPE_COUNT
};

struct RadiancePayload {
  float3 origin;
  float3 direction;
  float3 throughput = make_float3(1);
  RNGState rng;
  bool done = false;
};

struct ShadowPayload {
  float visibility = 0.0f;
};

// upper-32bit + lower-32bit -> 64bit
static __forceinline__ __device__ void* unpack_ptr(unsigned int i0,
                                                   unsigned int i1)
{
  const unsigned long long uptr =
      static_cast<unsigned long long>(i0) << 32 | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

// 64bit -> upper-32bit + lower-32bit
static __forceinline__ __device__ void pack_ptr(void* ptr, unsigned int& i0,
                                                unsigned int& i1)
{
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

// u0, u1 is upper-32bit, lower-32bit of ptr of Payload
template <typename Payload>
static __forceinline__ __device__ Payload* get_payload_ptr()
{
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<Payload*>(unpack_ptr(u0, u1));
}

// trace radiance ray
static __forceinline__ __device__ void trace_radiance(
    OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction,
    float tmin, float tmax, RadiancePayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE), u0, u1);
}

// trace shadow ray
static __forceinline__ __device__ void trace_shadow(
    OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction,
    float tmin, float tmax, ShadowPayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             static_cast<unsigned int>(RayType::RAY_TYPE_SHADOW),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_SHADOW), u0, u1);
}

static __forceinline__ __device__ void sample_ray_pinhole_camera(
    float2 uv, float3& origin, float3& direction)
{
  const float3 p_sensor =
      params.cam_origin + uv.x * params.cam_right + uv.y * params.cam_up;
  const float3 p_pinhole = params.cam_origin + params.cam_forward;

  origin = params.cam_origin;
  direction = normalize(p_pinhole - p_sensor);
}

extern "C" __global__ void __raygen__rg()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const float2 uv = make_float2((2.0f * idx.x - dim.x) / dim.x,
                                (2.0f * idx.y - dim.y) / dim.y);
  float3 ray_origin, ray_direction;
  sample_ray_pinhole_camera(uv, ray_origin, ray_direction);

  RadiancePayload payload;
  // TODO: use hash to set more nice seed
  payload.rng.state = idx.x + params.width * idx.y;
  payload.done = false;

  int depth = 0;
  while (!payload.done) {
    trace_radiance(params.gas_handle, ray_origin, ray_direction, 0.0f, 1e9f,
                   &payload);

    if (payload.done || depth > 3) { break; }

    // advance ray
    ray_origin = payload.origin;
    ray_direction = payload.direction;

    depth++;
  }

  // write radiance to frame buffer
  params.framebuffer[idx.x + params.width * idx.y] =
      make_float4(payload.throughput, 1.0f);
}

extern "C" __global__ void __miss__radiance()
{
  const MissSbtRecordData* sbt =
      reinterpret_cast<MissSbtRecordData*>(optixGetSbtDataPointer());

  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();
  payload->throughput *= sbt->bg_color;
  payload->done = true;
}

extern "C" __global__ void __miss__shadow()
{
  ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
  payload->visibility = 1.0f;
}

extern "C" __global__ void __closesthit__radiance()
{
  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());

  // compute face normal
  const int prim_idx = optixGetPrimitiveIndex();
  const float3 v0 = sbt->vertices[3 * prim_idx + 0];
  const float3 v1 = sbt->vertices[3 * prim_idx + 1];
  const float3 v2 = sbt->vertices[3 * prim_idx + 2];
  const float3 n = normalize(cross(v1 - v0, v2 - v0));

  // compute tangent space basis
  // TODO: compute normal
  // const float3 n = make_float3(0, 1, 0);
  // float3 t, b;
  // orthonormal_basis(n, t, b);

  // sample shadow ray direction
  // const float3 wi = sample_cosine_weighted_hemisphere(frandom(&payload->rng),
  //                                                     frandom(&payload->rng));
  // const float3 wi_world = local_to_world(wi, t, n, b);
  const float3 wi_world = make_float3(0, 1, 0);

  // trace shadow ray
  const float3 shadow_ray_origin =
      optixGetWorldRayOrigin() +
      optixGetRayTmax() * optixGetWorldRayDirection();
  const float3 shadow_ray_direction = wi_world;
  ShadowPayload shadow_payload;
  trace_shadow(params.gas_handle, shadow_ray_origin, shadow_ray_direction,
               RAY_EPS, 1e9f, &shadow_payload);

  // multiply visibility
  // payload->throughput *= shadow_payload.visibility;
  // payload->throughput *= sbt->material.base_color;
  payload->throughput *= 0.5f * (n + 1.0f);
  payload->done = true;
}

extern "C" __global__ void __closesthit__shadow()
{
  ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
  payload->visibility = 0.0f;
}