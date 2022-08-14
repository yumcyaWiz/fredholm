#pragma once
#include "sutil/vec_math.h"

static __forceinline__ __device__ float3 linear_to_srgb(const float3& rgb)
{
  float3 ret;
  ret.x = rgb.x < 0.0031308 ? 12.92 * rgb.x
                            : 1.055 * pow(rgb.x, 1.0f / 2.4f) - 0.055;
  ret.y = rgb.y < 0.0031308 ? 12.92 * rgb.y
                            : 1.055 * pow(rgb.y, 1.0f / 2.4f) - 0.055;
  ret.z = rgb.z < 0.0031308 ? 12.92 * rgb.z
                            : 1.055 * pow(rgb.z, 1.0f / 2.4f) - 0.055;
  return ret;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
static __forceinline__ __device__ float3 aces_tone_mapping(const float3& color)
{
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e),
               make_float3(0.0f), make_float3(1.0f));
}

// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
static __forceinline__ __device__ float compute_EV100(float aperture,
                                                      float shutter_time,
                                                      float ISO)
{
  return log2f(aperture * aperture / shutter_time * 100.0 / ISO);
}

static __forceinline__ __device__ float convert_EV100_to_exposure(float EV100)
{
  float maxLuminance = 1.2 * powf(2.0, EV100);
  return 1.0f / maxLuminance;
}

void __host__ post_process_launch(const float4* beauty_in,
                                  const float4* denoised_in, int width,
                                  int height, float ISO, float4* beauty_out,
                                  float4* denoised_out);

__global__ void post_process_kernel(const float4* beauty_in,
                                    const float4* denoised_in, int width,
                                    int height, float ISO, float4* beauty_out,
                                    float4* denoised_out);