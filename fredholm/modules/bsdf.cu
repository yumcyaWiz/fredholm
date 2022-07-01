#pragma once

#include "bxdf.cu"
#include "fredholm/shared.h"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params)
      : m_base_color(shading_params.base_color),
        m_specular(shading_params.specular),
        m_specular_color(shading_params.specular_color)
  {
    m_lambert_brdf = Lambert(m_base_color);
    m_specular_brdf = MicrofacetReflectionDielectric(1.5f, 0.2f, 0.0f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_lambert_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    return diffuse + m_specular * m_specular_color * specular;
  }

  __device__ float3 sample(const float3& wo, float u, const float2& v,
                           float3& f, float& pdf) const
  {
    const float specular_color_avg =
        (m_specular_color.x + m_specular_color.y + m_specular_color.z) / 3;
    if (u < m_specular * specular_color_avg * 0.5f) {
      const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
      f *= m_specular * m_specular_color;
      pdf *= m_specular * specular_color_avg * 0.5f;
      return wi;
    } else {
      const float3 wi = m_lambert_brdf.sample(wo, v, f, pdf);
      pdf *= fmax(1.0f - m_specular * specular_color_avg * 0.5f, 0.0f);
      return wi;
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float specular_color_avg =
        (m_specular_color.x + m_specular_color.y + m_specular_color.z) / 3;
    return (1.0f - m_specular * specular_color_avg * 0.5f) *
               m_lambert_brdf.eval_pdf(wo, wi) +
           m_specular * specular_color_avg * 0.5f *
               m_specular_brdf.eval_pdf(wo, wi);
  }

 private:
  float3 m_base_color;

  float m_specular;
  float3 m_specular_color;

  Lambert m_lambert_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
};