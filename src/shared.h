#pragma once
#include <cuda_runtime.h>
#include <optix.h>

struct Params {
  float4* image;
  unsigned int image_width;
  unsigned int image_height;

  float3 cam_origin;
  float3 cam_forward, cam_right, cam_up;

  OptixTraversableHandle handle;
};

struct RayGenData {
};

struct MissData {
};

struct HitGroupData {
};