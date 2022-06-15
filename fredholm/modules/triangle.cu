#include <optix.h>

#include "shared.h"
#include "sutil/vec_math.h"

extern "C" {
__constant__ LaunchParams params;
}

struct RadiancePayload {
  float3 radiance;
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
static __forceinline__ __device__ RadiancePayload* get_payload_ptr()
{
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<RadiancePayload*>(unpack_ptr(u0, u1));
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
  payload.radiance = make_float3(0.0f);

  unsigned int u0, u1;
  pack_ptr(&payload, u0, u1);
  optixTrace(params.gas_handle, ray_origin, ray_direction, 0.0f, 1e9f, 0.0f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);

  params.framebuffer[idx.x + idx.y * params.width] =
      make_float4(payload.radiance, 1.0f);
}

extern "C" __global__ void __miss__ms()
{
  RadiancePayload* payload = get_payload_ptr();
  payload->radiance = make_float3(0.0f);
}

extern "C" __global__ void __closesthit__ch()
{
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());

  RadiancePayload* payload = get_payload_ptr();
  payload->radiance = sbt->material.base_color;
}