#pragma once
#include <cuda_runtime.h>
#include <optix.h>

namespace fredholm
{

enum class RayType : unsigned int {
  RAY_TYPE_RADIANCE = 0,
  RAY_TYPE_SHADOW = 1,
  RAY_TYPE_LIGHT = 2,
  RAY_TYPE_COUNT
};

struct CameraParams {
  float3 origin;
  float3 forward;
  float3 right;
  float3 up;
  float f;
};

struct RNGState {
  unsigned long long state = 0;
  unsigned long long inc = 1;
};

// similar to arnold standard surface
// https://autodesk.github.io/standard-surface/
struct Material {
  float3 base_color = make_float3(1, 1, 1);
  int base_color_texture_id = -1;

  float specular = 1.0f;
  float3 specular_color = make_float3(1, 1, 1);
  int specular_color_texture_id = -1;
  float specular_roughness = 0.2f;
  int specular_roughness_texture_id = -1;

  float metalness = 0;
  int metalness_texture_id = -1;

  float coat = 0;
  float3 coat_color = make_float3(1, 1, 1);
  float coat_roughness = 0.1;

  float transmission = 0;
  float3 transmission_color = make_float3(1, 1, 1);

  float emission = 0;
  float3 emission_color = make_float3(0, 0, 0);

  // int heightmap_texture_id = -1;
  int normalmap_texture_id = -1;
  int alpha_texture_id = -1;
};

struct AreaLight {
  float3 le;      // emission
  uint3 indices;  // indices of vertex array
};

struct DirectionalLight {
  float3 le;   // emission
  float3 dir;  // direction
};

struct SurfaceInfo {
  float t;             // ray tmax
  float3 x;            // shading position
  float3 n_g;          // geometric normal in world space
  float3 n_s;          // shading normal in world space
  float2 barycentric;  // barycentric coordinate
  float2 texcoord;     // texture coordinate
  float3 tangent;      // tangent vector in world space
  float3 bitangent;    // bitangent vector in world space
  bool is_entering;
};

struct ShadingParams {
  float3 base_color = make_float3(0, 0, 0);

  float specular = 1.0f;
  float3 specular_color = make_float3(0, 0, 0);
  float specular_roughness = 0.2f;

  float metalness = 0.0f;

  float coat = 0.0f;
  float3 coat_color = make_float3(1, 1, 1);
  float coat_roughness = 0.1f;

  float transmission = 0;
  float3 transmission_color = make_float3(1, 1, 1);
};

struct RenderLayer {
  float4* beauty;
  float4* position;
  float* depth;
  float4* normal;
  float4* texcoord;
  float4* albedo;
};

struct LaunchParams {
  RenderLayer render_layer;
  uint* sample_count;
  RNGState* rng_states;

  uint width;
  uint height;
  uint n_samples;
  uint max_depth;

  CameraParams camera;

  float3 bg_color;

  Material* materials;
  cudaTextureObject_t* textures;
  AreaLight* lights;
  uint n_lights;

  DirectionalLight* directional_light;
  cudaTextureObject_t ibl;

  OptixTraversableHandle ias_handle;
};

struct RayGenSbtRecordData {
};

struct MissSbtRecordData {
};

struct HitGroupSbtRecordData {
  float3* vertices;
  uint3* indices;
  float3* normals;
  float2* texcoords;
  uint* material_ids;
};

template <typename T>
struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using RayGenSbtRecord = SbtRecord<RayGenSbtRecordData>;
using MissSbtRecord = SbtRecord<MissSbtRecordData>;
using HitGroupSbtRecord = SbtRecord<HitGroupSbtRecordData>;

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static __forceinline__ __device__ __host__ uint pcg32_random_r(RNGState* rng)
{
  unsigned long long oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

}  // namespace fredholm