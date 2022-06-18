#include <optix.h>

#include "math.cu"
#include "sampling.cu"
#include "shared.h"
#include "sutil/vec_math.h"

#define RAY_EPS 0.001f

using namespace fredholm;

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
  float3 radiance = make_float3(0);

  RNGState rng;

  bool done = false;
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
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    RadiancePayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE), u0, u1);
}

static __forceinline__ __device__ bool has_emission(const Material& material)
{
  return (material.emission_color.x > 0 || material.emission_color.y > 0 ||
          material.emission_color.z > 0);
}

static __forceinline__ __device__ void sample_ray_pinhole_camera(
    const float2& uv, float3& origin, float3& direction)
{
  const float3 p_sensor = params.camera.origin + uv.x * params.camera.right +
                          uv.y * params.camera.up;
  const float3 p_pinhole =
      params.camera.origin + params.camera.f * params.camera.forward;

  origin = p_sensor;
  direction = normalize(p_pinhole - p_sensor);
}

static __forceinline__ __device__ void fill_surface_info(
    const float3* vertices, const uint3* indices, const float3* normals,
    const float2* texcoords, const float3& ray_origin,
    const float3& ray_direction, float ray_tmax, const float2& barycentric,
    uint prim_idx, SurfaceInfo& info)
{
  info.x = ray_origin + ray_tmax * ray_direction;
  info.barycentric = barycentric;

  const uint3 idx = indices[prim_idx];
  const float3 v0 = vertices[idx.x];
  const float3 v1 = vertices[idx.y];
  const float3 v2 = vertices[idx.z];
  info.n_g = normalize(cross(v1 - v0, v2 - v0));

  const float3 n0 = normals[idx.x];
  const float3 n1 = normals[idx.y];
  const float3 n2 = normals[idx.z];
  info.n_s = (1.0f - info.barycentric.x - info.barycentric.y) * n0 +
             info.barycentric.x * n1 + info.barycentric.y * n2;

  const float2 tex0 = texcoords[idx.x];
  const float2 tex1 = texcoords[idx.y];
  const float2 tex2 = texcoords[idx.z];
  info.texcoord = (1.0f - info.barycentric.x - info.barycentric.y) * tex0 +
                  info.barycentric.x * tex1 + info.barycentric.y * tex2;

  orthonormal_basis(info.n_s, info.tangent, info.bitangent);
}

extern "C" __global__ void __raygen__rg()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  const uint image_idx = idx.x + params.width * idx.y;

  // set RNG state
  RadiancePayload payload;
  payload.rng.state = params.rng_states[image_idx].state;
  payload.rng.inc = params.rng_states[image_idx].inc;

  for (int spp = 0; spp < params.n_samples; ++spp) {
    // generate initial ray from camera
    const float2 uv =
        make_float2((2.0f * (idx.x + frandom(payload.rng)) - dim.x) / dim.x,
                    (2.0f * (idx.y + frandom(payload.rng)) - dim.y) / dim.y);
    sample_ray_pinhole_camera(uv, payload.origin, payload.direction);

    // start ray tracing from the camera
    payload.radiance = make_float3(0);
    payload.throughput = make_float3(1);
    payload.done = false;
    for (int depth = 0; depth < params.max_depth; ++depth) {
      trace_radiance(params.gas_handle, payload.origin, payload.direction, 0.0f,
                     1e9f, &payload);

      if (payload.done) { break; }
    }

    // accumulate contribution
    params.accumulation[image_idx] += make_float4(payload.radiance, 1.0f);
    params.sample_count[image_idx] += 1;
  }

  // save RNG state for next sampling
  params.rng_states[image_idx].state = payload.rng.state;
  params.rng_states[image_idx].inc = payload.rng.inc;

  // take average
  float3 radiance = make_float3(params.accumulation[image_idx]);
  radiance /= params.sample_count[image_idx];

  // gamma correction
  radiance.x = pow(radiance.x, 1.0f / 2.2f);
  radiance.y = pow(radiance.y, 1.0f / 2.2f);
  radiance.z = pow(radiance.z, 1.0f / 2.2f);

  // write radiance to frame buffer
  params.framebuffer[image_idx] = make_float4(radiance, 1.0f);
}

extern "C" __global__ void __miss__radiance()
{
  const MissSbtRecordData* sbt =
      reinterpret_cast<MissSbtRecordData*>(optixGetSbtDataPointer());

  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();
  payload->radiance += payload->throughput * sbt->bg_color;
  payload->done = true;
}

extern "C" __global__ void __miss__shadow() {}

extern "C" __global__ void __closesthit__radiance()
{
  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());

  // Le
  if (has_emission(sbt->material)) {
    payload->radiance += payload->throughput * sbt->material.emission_color;
    payload->done = true;
    return;
  }

  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float ray_tmax = optixGetRayTmax();
  const float2 barycentric = optixGetTriangleBarycentrics();
  const uint prim_idx = optixGetPrimitiveIndex();

  SurfaceInfo surf_info;
  fill_surface_info(sbt->vertices, sbt->indices, sbt->normals, sbt->texcoords,
                    ray_origin, ray_direction, ray_tmax, barycentric, prim_idx,
                    surf_info);

  // sample next ray direction
  const float3 wi = sample_cosine_weighted_hemisphere(frandom(payload->rng),
                                                      frandom(payload->rng));
  const float3 wi_world =
      local_to_world(wi, surf_info.tangent, surf_info.n_s, surf_info.bitangent);

  // update payload
  payload->throughput *= sbt->material.base_color;

  // advance ray
  payload->origin = surf_info.x + RAY_EPS * surf_info.n_s;
  payload->direction = wi_world;
}

extern "C" __global__ void __closesthit__shadow() {}