#pragma once

#include "bxdf.cu"
#include "fredholm/shared.h"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params)
      : m_base_color(shading_params.base_color),
        m_specular(shading_params.specular),
        m_specular_color(shading_params.specular_color),
        m_metalness(shading_params.metalness)
  {
    m_lambert_brdf = Lambert(m_base_color);
    m_specular_brdf = MicrofacetReflectionDielectric(1.5f, 0.2f, 0.0f);

    float3 n, k;
    const float3 reflectivity =
        clamp(shading_params.base_color, make_float3(0), make_float3(0.99));
    const float3 edge_tint =
        clamp(shading_params.specular_color, make_float3(0), make_float3(0.99));
    artist_friendly_metallic_fresnel(shading_params.base_color,
                                     shading_params.specular_color, n, k);
    m_metal_brdf = MicrofacetReflectionConductor(n, k, 0.2f, 0.0f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_lambert_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    const float3 metal = m_metal_brdf.eval(wo, wi);
    return m_metalness * metal +
           (1.0f - m_metalness) *
               (diffuse + m_specular * m_specular_color * specular);
  }

  __device__ float3 sample(const float3& wo, const float2& u, const float2& v,
                           float3& f, float& pdf) const
  {
    // metal
    if (u.x < m_metalness) {
      const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
      f *= m_metalness;
      pdf *= m_metalness;
      return wi;
    }
    // specular + diffuse
    else {
      const float specular_color_luminance = rgb_to_luminance(m_specular_color);
      // specular
      if (u.y < m_specular * specular_color_luminance * 0.5f) {
        const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
        f *= (1.0f - m_metalness) * m_specular * m_specular_color;
        pdf *=
            (1.0f - m_metalness) * m_specular * specular_color_luminance * 0.5f;
        return wi;
      }
      // diffuse
      else {
        const float3 wi = m_lambert_brdf.sample(wo, v, f, pdf);
        f *= (1.0f - m_metalness);
        pdf *= (1.0f - m_metalness) *
               (1.0f - m_specular * specular_color_luminance * 0.5f);
        return wi;
      }
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float specular_color_luminance = rgb_to_luminance(m_specular_color);
    return (1.0f - m_specular * specular_color_luminance * 0.5f) *
               m_lambert_brdf.eval_pdf(wo, wi) +
           m_specular * specular_color_luminance * 0.5f *
               m_specular_brdf.eval_pdf(wo, wi);
  }

 private:
  float3 m_base_color;

  float m_specular;
  float3 m_specular_color;

  float m_metalness;

  Lambert m_lambert_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
  MicrofacetReflectionConductor m_metal_brdf;
};