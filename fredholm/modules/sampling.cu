#pragma once

#include "fredholm/shared.h"
#include "sobol.cu"
#include "sutil/vec_math.h"

using namespace fredholm;

static __forceinline__ __device__ float funiform(PCGState& state)
{
  return pcg32_random_r(&state) * (1.0f / (1ULL << 32));
}

static __forceinline__ __device__ float sample_1d(SamplerState& state)
{
  return fsobol_owen(state.sobol_state);
}

static __forceinline__ __device__ float2 sample_2d(SamplerState& state)
{
  return make_float2(sample_1d(state), sample_1d(state));
}

static __forceinline__ __device__ float3 sampler_3d(SamplerState& state)
{
  return make_float3(sample_1d(state), sample_1d(state), sample_1d(state));
}

static __forceinline__ __device__ float4 sample_4d(SamplerState& state)
{
  return make_float4(sample_1d(state), sample_1d(state), sample_1d(state),
                     sample_1d(state));
}

static __forceinline__ __device__ float2 sample_uniform_disk(const float2& u)
{
  const float r = sqrtf(u.x);
  const float theta = 2.0f * M_PIf * u.y;
  return make_float2(r * cosf(theta), r * sinf(theta));
}

static __forceinline__ __device__ float3
sample_cosine_weighted_hemisphere(const float2& u)
{
  const float2 p_disk = sample_uniform_disk(u);

  float3 p;
  p.x = p_disk.x;
  p.z = p_disk.y;
  // Project up to hemisphere.
  p.y = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.z * p.z));

  return p;
}

static __forceinline__ __device__ float2 sample_triangle(const float2& u)
{
  const float su0 = sqrtf(u.x);
  return make_float2(1.0f - su0, u.y * su0);
}

// https://jcgt.org/published/0007/04/01/
static __device__ float3 sample_vndf(const float3& wo, const float2& alpha,
                                     const float2& u)
{
  const float3 Vh =
      normalize(make_float3(alpha.x * wo.x, wo.y, alpha.y * wo.z));

  const float lensq = Vh.x * Vh.x + Vh.z * Vh.z;
  const float3 T1 = lensq > 0 ? make_float3(Vh.z, 0, -Vh.x) / sqrtf(lensq)
                              : make_float3(0, 0, 1);
  const float3 T2 = cross(Vh, T1);

  const float r = sqrtf(u.x);
  const float phi = 2.0f * M_PI * u.y;
  const float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  const float s = 0.5f * (1.0f + Vh.y);
  t2 = (1.0f - s) * sqrtf(fmax(1.0f - t1 * t1, 0.0f)) + s * t2;
  const float3 Nh =
      t1 * T1 + t2 * T2 + sqrtf(fmax(1.0f - t1 * t1 - t2 * t2, 0.0f)) * Vh;
  const float3 Ne =
      normalize(make_float3(alpha.x * Nh.x, fmax(0.0f, Nh.y), alpha.y * Nh.z));

  return Ne;
}