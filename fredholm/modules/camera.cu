#pragma once
#include "fredholm/shared.h"
#include "sampling.cu"
#include "sutil/vec_math.h"

using namespace fredholm;

static __forceinline__ __device__ void sample_ray_pinhole_camera(
    const CameraParams& params, const float2& uv, float3& origin,
    float3& direction, float& pdf)
{
  const float f = 1.0f / tanf(0.5f * params.fov);
  const float3 p_sensor =
      params.origin + uv.x * params.right + uv.y * params.up;
  const float3 p_pinhole = params.origin + f * params.forward;

  origin = p_pinhole;
  direction = normalize(p_pinhole - p_sensor);
  pdf = 1.0f / dot(direction, params.forward);
}

static __forceinline__ __device__ void sample_ray_thinlens_camera(
    const CameraParams& params, const float2& uv, const float2& u,
    float3& origin, float3& direction, float& pdf)
{
  const float f = 1.0f / tanf(0.5f * params.fov);

  const float b = params.focus;
  const float a = 1.0f / (1.0f + f - 1.0f / b);
  const float lens_radius = 2.0f * f / params.F;

  const float3 p_sensor =
      params.origin + uv.x * params.right + uv.y * params.up;
  const float3 p_lens_center = params.origin + f * params.forward;

  const float2 p_disk = lens_radius * sample_concentric_disk(u);
  const float3 p_lens =
      p_lens_center + params.right * p_disk.x + params.up * p_disk.y;
  const float3 sensor_to_lens = normalize(p_lens - p_sensor);

  const float3 sensor_to_lens_center = normalize(p_lens_center - p_sensor);
  const float3 p_object =
      p_sensor + ((a + b) / dot(sensor_to_lens_center, params.forward)) *
                     sensor_to_lens_center;

  origin = p_lens;
  direction = normalize(p_object - p_lens);
  const float pdf_area = 1.0f / (M_PIf * lens_radius * lens_radius);
  pdf = length2(p_lens - p_sensor) / dot(sensor_to_lens, params.forward) *
        pdf_area;
}