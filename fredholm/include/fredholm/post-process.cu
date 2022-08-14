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

__global__ void post_process_kernel(const float4* beauty_in,
                                    const float4* denoised_in, int width,
                                    int height, float ISO, float4* beauty_out,
                                    float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  float3 color = make_float3(beauty_in[image_idx]);

  // beauty
  const float EV100 = compute_EV100(1.0f, 1.0f, ISO);
  const float exposure = convert_EV100_to_exposure(EV100);
  color *= exposure;
  color = aces_tone_mapping(color);
  color = linear_to_srgb(color);
  beauty_out[image_idx] = make_float4(color, 1.0f);

  // denoised
  color = make_float3(denoised_in[image_idx]);
  color *= exposure;
  color = aces_tone_mapping(color);
  color = linear_to_srgb(color);
  denoised_out[image_idx] = make_float4(color, 1.0f);
}