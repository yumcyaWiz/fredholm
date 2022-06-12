#pragma once
#include <cuda_runtime.h>
#include <optix.h>

struct Params {
  float4* framebuffer;
  unsigned int width;
  unsigned int height;

  float3 cam_origin;
  float3 cam_forward;
  float3 cam_right;
  float3 cam_up;

  OptixTraversableHandle gas_handle;
};

struct RayGenData {
};

struct MissData {
};

struct HitGroupData {
};