#pragma once

#include "bxdf.cu"

class BSDF
{
 public:
  __device__ BSDF(const float3& base_color, float specular)
      : m_base_color(base_color), m_specular(specular)
  {
    m_lambert_brdf = Lambert(m_base_color);
    m_specular_brdf = MicrofacetReflectionDielectric(1.5f, 0.2f, 0.0f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_lambert_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    return lerp(diffuse, specular, m_specular);
  }

  __device__ float3 sample(const float3& wo, float u, const float2& v,
                           float3& f, float& pdf) const
  {
    if (u < m_specular) {
      const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
      pdf /= m_specular;
      return wi;
    } else {
      const float3 wi = m_lambert_brdf.sample(wo, v, f, pdf);
      pdf /= fmax(1.0f - m_specular, 0.0f);
      return wi;
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return (1.0f - m_specular) * m_lambert_brdf.eval_pdf(wo, wi) +
           m_specular * m_specular_brdf.eval_pdf(wo, wi);
  }

 private:
  float3 m_base_color;
  float m_specular;

  Lambert m_lambert_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
};