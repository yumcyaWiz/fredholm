#pragma once

#include <emmintrin.h>

#include "bxdf.cu"
#include "fredholm/shared.h"
#include "math.cu"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params)
      : m_base_color(shading_params.base_color),
        m_specular(shading_params.specular),
        m_specular_color(shading_params.specular_color),
        m_specular_roughness(shading_params.specular_roughness),
        m_metalness(shading_params.metalness),
        m_coat(shading_params.coat),
        m_coat_color(shading_params.coat_color),
        m_coat_roughness(shading_params.coat_roughness)
  {
    m_coat_brdf = MicrofacetReflectionDielectric(1.5f, m_coat_roughness, 0.0f);

    m_specular_brdf =
        MicrofacetReflectionDielectric(1.5f, m_specular_roughness, 0.0f);

    float3 n, k;
    const float3 reflectivity =
        clamp(shading_params.base_color, make_float3(0), make_float3(0.99));
    const float3 edge_tint =
        clamp(shading_params.specular_color, make_float3(0), make_float3(0.99));
    artist_friendly_metallic_fresnel(reflectivity, edge_tint, n, k);
    m_metal_brdf =
        MicrofacetReflectionConductor(n, k, m_specular_roughness, 0.0f);

    m_diffuse_brdf = Lambert(m_base_color);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_diffuse_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    const float3 metal = m_metal_brdf.eval(wo, wi);
    const float3 coat = m_coat_brdf.eval(wo, wi);
    return m_coat * m_coat_color * coat +
           (1.0f - m_coat * m_coat_color) *
               (m_metalness * metal +
                (1.0f - m_metalness) *
                    (diffuse + m_specular * m_specular_color * specular));
  }

  __device__ float3 sample(const float3& wo, const float3& u, const float2& v,
                           float3& f, float& pdf) const
  {
    // coat
    const float coat_color_luminance = rgb_to_luminance(m_coat_color);
    if (u.x < m_coat * coat_color_luminance * 0.5f) {
      const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
      f *= m_coat * m_coat_color;
      pdf *= m_coat * coat_color_luminance * 0.5f;
      return wi;
    } else {
      // TODO: use clearcoat directional albedo
      float3 f_mult = (1.0f - m_coat * m_coat_color * 0.0f);
      float pdf_mult = (1.0f - m_coat * coat_color_luminance * 0.5f);

      // metal
      if (u.y < m_metalness) {
        const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
        f *= f_mult * m_metalness;
        pdf *= pdf_mult * m_metalness;
        return wi;
      }
      // specular + diffuse
      else {
        f_mult *= (1.0f - m_metalness);
        pdf_mult *= (1.0f - m_metalness);

        const float specular_color_luminance =
            rgb_to_luminance(m_specular_color);
        // specular
        // TODO: use specular directional albedo
        if (u.z < m_specular * specular_color_luminance * 0.5f) {
          const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
          f *= f_mult * m_specular * m_specular_color;
          pdf *= pdf_mult * m_specular * specular_color_luminance * 0.5f;
          return wi;
        }
        // diffuse
        else {
          const float3 wi = m_diffuse_brdf.sample(wo, v, f, pdf);
          f *= f_mult;
          pdf *=
              pdf_mult * (1.0f - m_specular * specular_color_luminance * 0.5f);
          return wi;
        }
      }
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float specular_color_luminance = rgb_to_luminance(m_specular_color);
    return (1.0f - m_specular * specular_color_luminance * 0.5f) *
               m_diffuse_brdf.eval_pdf(wo, wi) +
           m_specular * specular_color_luminance * 0.5f *
               m_specular_brdf.eval_pdf(wo, wi);
  }

 private:
  // TODO: hold Shading Params instead?
  float3 m_base_color;

  float m_specular;
  float3 m_specular_color;
  float m_specular_roughness;

  float m_metalness;

  float m_coat;
  float3 m_coat_color;
  float m_coat_roughness;

  MicrofacetReflectionDielectric m_coat_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
  MicrofacetReflectionConductor m_metal_brdf;
  Lambert m_diffuse_brdf;
};