#pragma once
#include <cuda_runtime.h>
#include <optix.h>

namespace fredholm
{

enum class RayType : unsigned int {
  RAY_TYPE_RADIANCE = 0,
  RAY_TYPE_SHADOW = 1,
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

  int alpha_texture_id = -1;
};

struct SurfaceInfo {
  float t;             // ray tmax
  float3 x;            // shading position
  float3 wo;           // view direction in tangent space
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

  cudaTextureObject_t ibl;
  cudaTextureObject_t lut;

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

}  // namespace fredholm