#pragma once
#include <cuda_runtime.h>
#include <optix.h>

// similar to arnold standard surface
// https://autodesk.github.io/standard-surface/
// TODO: support texture input
struct Material {
  float base = 0.8;
  float3 base_color = make_float3(1, 1, 1);
  float diffuse_roughness = 0;

  float specular = 0;
  float metalness = 0;
  float3 specular_color = make_float3(0.2, 0.2, 0.2);
  float specular_roughness = 0.2;
  float specular_anisotropy = 0;
  float specular_rotation = 0;
  float specular_IOR = 1.5;

  float thin_film_thickness = 0;
  float thin_film_IOR = 1.5;

  float coat = 0;
  float3 coat_color = make_float3(1, 1, 1);
  float coat_roughness = 0.1;
  float coat_anisotropy = 0;
  float coat_rotation = 0;
  float coat_IOR = 1.5;

  float emission = 0;
  float3 emission_color = make_float3(1, 1, 1);

  float sheen = 0.8;
  float3 sheen_color = make_float3(1, 1, 1);
  float sheen_roughness = 0.3;
};

struct LaunchParams {
  float4* framebuffer;
  unsigned int width;
  unsigned int height;
  unsigned int n_samples;
  unsigned int max_depth;

  float3 cam_origin;
  float3 cam_forward;
  float3 cam_right;
  float3 cam_up;

  OptixTraversableHandle gas_handle;
};

struct RayGenSbtRecordData {
};

struct MissSbtRecordData {
  float3 bg_color = make_float3(0, 0, 0);
};

struct HitGroupSbtRecordData {
  Material material;
  float3* vertices;
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