#include <optix.h>

#include "math.cu"
#include "sampling.cu"
#include "shared.h"
#include "sutil/vec_math.h"

#define RAY_EPS 0.001f

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

  float3 radiance = make_float3(0);

  // warm up rng
  // TODO: use some hash function to set more nice seed
  RadiancePayload payload;
  payload.rng.state = idx.x + params.width * idx.y;
  for (int i = 0; i < 10; ++i) { frandom(payload.rng); }

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
    radiance += payload.radiance;
  }

  // take average
  radiance /= params.n_samples;

  // gamma correction
  radiance.x = pow(radiance.x, 1.0f / 2.2f);
  radiance.y = pow(radiance.y, 1.0f / 2.2f);
  radiance.z = pow(radiance.z, 1.0f / 2.2f);

  // write radiance to frame buffer
  params.framebuffer[idx.x + params.width * idx.y] =
      make_float4(radiance, 1.0f);
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

  // compute face normal
  // TODO: remove this calculation, store normals(ptr) in SBT
  const int prim_idx = optixGetPrimitiveIndex();
  const float3 v0 = sbt->vertices[3 * prim_idx + 0];
  const float3 v1 = sbt->vertices[3 * prim_idx + 1];
  const float3 v2 = sbt->vertices[3 * prim_idx + 2];
  const float3 n = normalize(cross(v1 - v0, v2 - v0));

  // compute tangent space basis
  float3 t, b;
  orthonormal_basis(n, t, b);

  // sample next ray direction
  const float3 wi = sample_cosine_weighted_hemisphere(frandom(payload->rng),
                                                      frandom(payload->rng));
  const float3 wi_world = local_to_world(wi, t, n, b);

  // update payload
  payload->throughput *= sbt->material.base_color;

  // advance ray
  payload->origin = optixGetWorldRayOrigin() +
                    optixGetRayTmax() * optixGetWorldRayDirection();
  payload->origin += RAY_EPS * n;
  payload->direction = wi_world;
}

extern "C" __global__ void __closesthit__shadow() {}