#pragma once

#include <emmintrin.h>

#include "bxdf.cu"
#include "fredholm/shared.h"
#include "math.cu"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params)
      : m_params(shading_params)
  {
    m_coat_brdf =
        MicrofacetReflectionDielectric(1.5f, m_params.coat_roughness, 0.0f);

    m_specular_brdf =
        MicrofacetReflectionDielectric(1.5f, m_params.specular_roughness, 0.0f);

    float3 n, k;
    const float3 reflectivity =
        clamp(shading_params.base_color, make_float3(0), make_float3(0.99));
    const float3 edge_tint =
        clamp(shading_params.specular_color, make_float3(0), make_float3(0.99));
    artist_friendly_metallic_fresnel(reflectivity, edge_tint, n, k);
    m_metal_brdf =
        MicrofacetReflectionConductor(n, k, m_params.specular_roughness, 0.0f);

    m_diffuse_brdf = Lambert(m_params.base_color);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_diffuse_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    const float3 metal = m_metal_brdf.eval(wo, wi);
    const float3 coat = m_coat_brdf.eval(wo, wi);
    return m_params.coat * m_params.coat_color * coat +
           (1.0f - m_params.coat * m_params.coat_color) *
               (m_params.metalness * metal +
                (1.0f - m_params.metalness) *
                    (diffuse +
                     m_params.specular * m_params.specular_color * specular));
  }

  __device__ float3 sample(const float3& wo, const float3& u, const float2& v,
                           float3& f, float& pdf) const
  {
    // coat
    const float coat_color_luminance = rgb_to_luminance(m_params.coat_color);
    if (u.x < m_params.coat * coat_color_luminance * 0.5f) {
      const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
      f *= m_params.coat * m_params.coat_color;
      pdf *= m_params.coat * coat_color_luminance * 0.5f;
      return wi;
    } else {
      // TODO: use clearcoat directional albedo
      float3 f_mult = (1.0f - m_params.coat * m_params.coat_color * 0.0f);
      float pdf_mult = (1.0f - m_params.coat * coat_color_luminance * 0.5f);

      // metal
      if (u.y < m_params.metalness) {
        const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
        f *= f_mult * m_params.metalness;
        pdf *= pdf_mult * m_params.metalness;
        return wi;
      }
      // specular + diffuse
      else {
        f_mult *= (1.0f - m_params.metalness);
        pdf_mult *= (1.0f - m_params.metalness);

        const float specular_color_luminance =
            rgb_to_luminance(m_params.specular_color);
        // specular
        // TODO: use specular directional albedo
        if (u.z < m_params.specular * specular_color_luminance * 0.5f) {
          const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
          f *= f_mult * m_params.specular * m_params.specular_color;
          pdf *= pdf_mult * m_params.specular * specular_color_luminance * 0.5f;
          return wi;
        }
        // diffuse
        else {
          const float3 wi = m_diffuse_brdf.sample(wo, v, f, pdf);
          f *= f_mult;
          pdf *= pdf_mult *
                 (1.0f - m_params.specular * specular_color_luminance * 0.5f);
          return wi;
        }
      }
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float specular_color_luminance =
        rgb_to_luminance(m_params.specular_color);
    return (1.0f - m_params.specular * specular_color_luminance * 0.5f) *
               m_diffuse_brdf.eval_pdf(wo, wi) +
           m_params.specular * specular_color_luminance * 0.5f *
               m_specular_brdf.eval_pdf(wo, wi);
  }

 private:
  ShadingParams m_params;

  MicrofacetReflectionDielectric m_coat_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
  MicrofacetReflectionConductor m_metal_brdf;
  Lambert m_diffuse_brdf;
};