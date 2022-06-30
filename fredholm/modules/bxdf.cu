#pragma once

#include <cmath>

#include "sampling.cu"
#include "sutil/vec_math.h"

__forceinline__ __device__ float cos_theta(const float3& w) { return w.y; }

__forceinline__ __device__ float cos2_theta(const float3& w)
{
  return w.y * w.y;
}

__forceinline__ __device__ float abs_cos_theta(const float3& w)
{
  return fabs(w.y * w.y);
}

__forceinline__ __device__ float sin_theta(const float3& w)
{
  return sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float sin2_theta(const float3& w)
{
  return fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_sin_theta(const float3& w)
{
  return fabs(sin_theta(w));
}

__forceinline__ __device__ float tan_theta(const float3& w)
{
  return sin_theta(w) / cos_theta(w);
}

__forceinline__ __device__ float tan2_theta(const float3& w)
{
  return 1.0f / (w.y * w.y) - 1.0f;
}

__forceinline__ __device__ float abs_tan_theta(const float3& w)
{
  return fabs(tan_theta(w));
}

__forceinline__ __device__ float sin_phi(const float3& w)
{
  return w.z / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float sin2_phi(const float3& w)
{
  return w.z * w.z / fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_sin_phi(const float3& w)
{
  return fabs(sin_phi(w));
}

__forceinline__ __device__ float cos_phi(const float3& w)
{
  return w.x / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float cos2_phi(const float3& w)
{
  return w.x * w.x / fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_cos_phi(const float3& w)
{
  return fabs(cos_phi(w));
}

__forceinline__ __device__ float3 reflect(const float3& w, const float3& n)
{
  return -w + 2.0f * dot(w, n) * n;
}

class Lambert
{
 public:
  __device__ Lambert() {}
  __device__ Lambert(const float3& albedo) : m_albedo(albedo) {}

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    return m_albedo / M_PIf;
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    const float3 wi = sample_cosine_weighted_hemisphere(u);

    f = eval(wo, wi);
    pdf = abs_cos_theta(wi) / M_PIf;

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return abs_cos_theta(wi) / M_PIf;
  }

 private:
  float3 m_albedo;
};

struct FresnelDielectric {
  __device__ FresnelDielectric() {}
  __device__ FresnelDielectric(float n) : m_n(n) {}

  __device__ float eval(float cos) const
  {
    const float s2 = fmax(1.0f - cos * cos, 0.0f);
    const float t0 = sqrtf(fmax(1.0f - (s2 / (m_n * m_n)), 0.0f));
    const float t1 = m_n * t0;
    const float t2 = m_n * cos;

    const float rs = (cos - t1) / (cos + t1);
    const float rp = (t0 - t2) / (t0 + t2);

    return 0.5f * (rs * rs + rp * rp);
  }

  float m_n;
};

struct FresnelConductor {
  __device__ FresnelConductor() {}
  __device__ FresnelConductor(const float3& n, const float3& k) : m_n(n), m_k(k)
  {
  }

  __device__ float3 eval(float cos) const
  {
    const float c2 = cos * cos;
    const float3 two_eta_cos = 2.0f * m_n * cos;

    const float3 t0 = m_n * m_n + m_k * m_k;
    const float3 t1 = t0 * c2;
    const float3 Rs = (t0 - two_eta_cos + c2) / (t0 + two_eta_cos + c2);
    const float3 Rp = (t1 - two_eta_cos + 1.0f) / (t1 + two_eta_cos + 1.0f);

    return 0.5f * (Rp + Rs);
  }

  float3 m_n;
  float3 m_k;
};

class MicrofacetReflectionDielectric
{
 public:
  __device__ MicrofacetReflectionDielectric() {}
  __device__ MicrofacetReflectionDielectric(float ior, float roughness,
                                            float anisotropy)
      : m_fresnel(ior)
  {
    // Revisiting Physically Based Shading at Imageworks p.24
    m_alpha.x = roughness * roughness * (1.0f + anisotropy);
    m_alpha.y = roughness * roughness * (1.0f - anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float f = m_fresnel.eval(abs_cos_theta(wo));
    const float d = D(wh);
    const float g = G2(wo, wi);
    return make_float3(0.25f * (f * d * g) /
                       (abs_cos_theta(wo) * abs_cos_theta(wi)));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    // sample half-vector
    const float3 wh = sample_vndf(wo, m_alpha, u);

    // compute incident direction
    const float3 wi = reflect(wo, wh);

    // evaluate BxDF and pdf
    f = eval(wo, wi);
    pdf = eval_pdf(wo, wi);

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    return 0.25f * D_visible(wo, wh) / abs_cos_theta(wo);
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                    wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a0 = sqrtf(cos2_phi(w) * m_alpha.x * m_alpha.x +
                           sin2_phi(w) * m_alpha.y * m_alpha.y);
    const float a = 1.0f / (a0 * tan_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / (a * a)));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  FresnelDielectric m_fresnel;
  float2 m_alpha;
};

class MicrofacetReflectionConductor
{
 public:
  __device__ MicrofacetReflectionConductor() {}
  __device__ MicrofacetReflectionConductor(const float3& n, const float3& k,
                                           float roughness, float anisotropy)
      : m_fresnel(n, k)
  {
    // Revisiting Physically Based Shading at Imageworks p.24
    m_alpha.x = roughness * roughness * (1.0f + anisotropy);
    m_alpha.y = roughness * roughness * (1.0f - anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float3 f = m_fresnel.eval(abs_cos_theta(wo));
    const float d = D(wh);
    const float g = G2(wo, wi);
    return 0.25f * (f * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    // sample half-vector
    const float3 wh = sample_vndf(wo, m_alpha, u);

    // compute incident direction
    const float3 wi = reflect(wo, wh);

    // evaluate BxDF and pdf
    f = eval(wo, wi);
    pdf = eval_pdf(wo, wi);

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    return 0.25f * D_visible(wo, wh) / abs_cos_theta(wo);
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                    wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a0 = sqrtf(cos2_phi(w) * m_alpha.x * m_alpha.x +
                           sin2_phi(w) * m_alpha.y * m_alpha.y);
    const float a = 1.0f / (a0 * tan_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / (a * a)));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  FresnelConductor m_fresnel;
  float2 m_alpha;
};