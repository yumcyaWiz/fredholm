#pragma once

#include "fredholm/shared.h"
#include "sutil/vec_math.h"

using namespace fredholm;

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static __forceinline__ __device__ uint pcg32_random_r(RNGState* rng)
{
  unsigned long long oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static __forceinline__ __device__ float frandom(RNGState& rng)
{
  return pcg32_random_r(&rng) / static_cast<float>(0xffffffffu);
}

static __forceinline__ __device__ float3
sample_cosine_weighted_hemisphere(const float2& u)
{
  // Uniformly sample disk.
  const float r = sqrtf(u.x);
  const float phi = 2.0f * M_PIf * u.y;

  float3 p;
  p.x = r * cosf(phi);
  p.z = r * sinf(phi);
  // Project up to hemisphere.
  p.y = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.z * p.z));

  return p;
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