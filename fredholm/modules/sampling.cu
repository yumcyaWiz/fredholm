#pragma once

#include "blue-noise.cu"
#include "cmj.cu"
#include "fredholm/shared.h"
#include "math.cu"
#include "sobol.cu"
#include "sutil/vec_math.h"

#define DISCRETE_DISTRIBUTION_1D_MAX_SIZE 16

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
  // return make_float2(fsobol_owen(state.sobol_state),
  //                    fsobol_owen(state.sobol_state));
  return cmj_2d(state.cmj_state);
}

static __forceinline__ __device__ float3 sampler_3d(SamplerState& state)
{
  // return make_float3(fsobol_owen(state.sobol_state),
  //                    fsobol_owen(state.sobol_state),
  //                    fsobol_owen(state.sobol_state));
  return make_float3(cmj_2d(state.cmj_state), cmj_1d(state.cmj_state));
}

static __forceinline__ __device__ float4 sample_4d(SamplerState& state)
{
  // return make_float4(
  //     fsobol_owen(state.sobol_state), fsobol_owen(state.sobol_state),
  //     fsobol_owen(state.sobol_state), fsobol_owen(state.sobol_state));
  return make_float4(cmj_2d(state.cmj_state), cmj_2d(state.cmj_state));
}

static __forceinline__ __device__ float2 sample_uniform_disk(const float2& u)
{
  const float r = sqrtf(u.x);
  const float theta = 2.0f * M_PIf * u.y;
  return make_float2(r * cosf(theta), r * sinf(theta));
}

static __forceinline__ __device__ float2 sample_concentric_disk(const float2& u)
{
  const float2 u0 = 2.0f * u - 1.0f;
  if (u0.x == 0.0f && u0.y == 0.0f) return make_float2(0.0f);

  const float r = fabsf(u0.x) > fabsf(u0.y) ? u0.x : u0.y;
  const float theta = fabsf(u0.x) > fabsf(u0.y)
                          ? 0.25f * M_PIf * u0.y / u0.x
                          : 0.5f * M_PIf - 0.25f * M_PIf * u0.x / u0.y;
  return make_float2(r * cosf(theta), r * sinf(theta));
}

static __forceinline__ __device__ float3
sample_cosine_weighted_hemisphere(const float2& u)
{
  const float2 p_disk = sample_concentric_disk(u);

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

struct DiscreteDistribution1D {
  __device__ DiscreteDistribution1D() {}

  __device__ void init(const float* values, int size)
  {
    m_size = size;

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) { sum += values[i]; }

    // compute cdf
    m_cdf[0] = 0.0f;
    for (int i = 1; i < size + 1; ++i) {
      m_cdf[i] = m_cdf[i - 1] + values[i - 1] / sum;
    }
  }

  __device__ int sample(float u, float& pmf) const
  {
    const int idx = binary_search(m_cdf, m_size + 1, u);
    pmf = m_cdf[idx + 1] - m_cdf[idx];
    return idx;
  }

  __device__ float eval_pmf(int idx) const
  {
    return m_cdf[idx + 1] - m_cdf[idx];
  }

  float m_cdf[DISCRETE_DISTRIBUTION_1D_MAX_SIZE + 1];
  int m_size;
};