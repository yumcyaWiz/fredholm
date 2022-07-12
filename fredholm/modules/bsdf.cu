#pragma once

#include "bxdf.cu"
#include "fredholm/shared.h"
#include "lut.cu"
#include "math.cu"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params, bool is_entering)

      : m_params(shading_params)
  {
    m_ni = is_entering ? 1.0f : 1.5f;
    m_nt = is_entering ? 1.5f : 1.0f;
    const float eta = m_nt / m_ni;
    m_coat_brdf =
        MicrofacetReflectionDielectric(eta, m_params.coat_roughness, 0.0f);

    m_specular_brdf =
        MicrofacetReflectionDielectric(eta, m_params.specular_roughness, 0.0f);

    float3 n, k;
    const float3 reflectivity =
        clamp(shading_params.base_color, make_float3(0), make_float3(0.99));
    const float3 edge_tint =
        clamp(shading_params.specular_color, make_float3(0), make_float3(0.99));
    artist_friendly_metallic_fresnel(reflectivity, edge_tint, n, k);
    m_metal_brdf =
        MicrofacetReflectionConductor(n, k, m_params.specular_roughness, 0.0f);

    m_transmission_btdf =
        MicrofacetTransmission(m_ni, m_nt, m_params.specular_roughness, 0.0f);

    m_diffuse_brdf = OrenNayer(m_params.base_color, 0.0f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 diffuse = m_diffuse_brdf.eval(wo, wi);
    const float3 specular = m_specular_brdf.eval(wo, wi);
    const float3 metal = m_metal_brdf.eval(wo, wi);
    const float3 coat = m_coat_brdf.eval(wo, wi);
    const float3 transmission = m_transmission_btdf.eval(wo, wi);

    const float3 coat_weight = m_params.coat * m_params.coat_color;
    const float3 metal_weight = make_float3(m_params.metalness);
    const float3 specular_weight = m_params.specular * m_params.specular_color;

    return coat_weight * coat +
           (1.0f - coat_weight) *
               (metal_weight * metal +
                (1.0f - metal_weight) * (diffuse + specular_weight * specular));
  }

  __device__ float3 sample(const float3& wo, const float4& u, const float2& v,
                           float3& f, float& pdf) const
  {
    // coat
    const float clearcoat_F0 = compute_F0(m_ni, m_nt);
    const float coat_color_luminance = rgb_to_luminance(m_params.coat_color);
    const float coat_directional_albedo =
        compute_directional_albedo(wo, m_params.coat_roughness, clearcoat_F0);
    if (u.x < m_params.coat * coat_color_luminance * coat_directional_albedo) {
      const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
      f *= m_params.coat * m_params.coat_color;
      pdf *= m_params.coat * coat_color_luminance * coat_directional_albedo;
      return wi;
    }
    // metal or transmission or specular or diffuse
    else {
      float3 f_mult = (1.0f - m_params.coat * m_params.coat_color *
                                  coat_directional_albedo);
      float pdf_mult = (1.0f - m_params.coat * coat_color_luminance *
                                   coat_directional_albedo);

      // metal
      if (u.y < m_params.metalness) {
        const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
        f *= f_mult * m_params.metalness;
        pdf *= pdf_mult * m_params.metalness;
        return wi;
      }
      // specular or transmission or diffuse
      else {
        f_mult *= (1.0f - m_params.metalness);
        pdf_mult *= (1.0f - m_params.metalness);

        const float specular_F0 = compute_F0(m_ni, m_nt);
        const float specular_color_luminance =
            rgb_to_luminance(m_params.specular_color);
        float specular_directional_albedo =
            m_nt > m_ni ? compute_directional_albedo(
                              wo, m_params.specular_roughness, specular_F0)
                        : compute_directional_albedo2(
                              wo, m_params.specular_roughness, m_nt / m_ni);
        // specular
        if (u.z < m_params.specular * specular_color_luminance *
                      specular_directional_albedo) {
          const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
          f *= f_mult * m_params.specular * m_params.specular_color;
          pdf *= pdf_mult * m_params.specular * specular_color_luminance *
                 specular_directional_albedo;
          return wi;
        }
        // transmission or diffuse
        else {
          f_mult *= (1.0f - m_params.specular * m_params.specular_color *
                                specular_directional_albedo);
          pdf_mult *= (1.0f - m_params.specular * specular_color_luminance *
                                  specular_directional_albedo);

          // transmission
          if (u.w < m_params.transmission) {
            const float3 wi = m_transmission_btdf.sample(wo, v, f, pdf);
            f *= f_mult * m_params.transmission * m_params.transmission_color;
            pdf *= pdf_mult * m_params.transmission;
            return wi;
          }
          // diffuse
          else {
            const float3 wi = m_diffuse_brdf.sample(wo, v, f, pdf);
            f *= f_mult * (1.0f - m_params.transmission);
            pdf *= pdf_mult * (1.0f - m_params.transmission);
            return wi;
          }
        }
      }
    }
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float coat_color_luminance = rgb_to_luminance(m_params.coat_color);
    const float coat_weight = m_params.coat * coat_color_luminance * 0.5f;

    const float metal_weight = m_params.metalness;

    const float transmission_color_luminance =
        rgb_to_luminance(m_params.transmission_color);
    const float transmission_weight =
        m_params.transmission * transmission_color_luminance * 0.5f;

    const float specular_color_luminance =
        rgb_to_luminance(m_params.specular_color);
    const float specular_weight =
        m_params.specular * specular_color_luminance * 0.5f;

    return coat_weight * m_coat_brdf.eval_pdf(wo, wi) +
           (1.0f - coat_weight) *
               (metal_weight * m_metal_brdf.eval_pdf(wo, wi) *
                (1.0f - metal_weight) *
                (transmission_weight * m_transmission_btdf.eval_pdf(wo, wi) +
                 (1.0f - transmission_weight) *
                     (specular_weight * m_specular_brdf.eval_pdf(wo, wi) +
                      (1.0f - specular_weight) *
                          m_diffuse_brdf.eval_pdf(wo, wi))));
  }

 private:
  ShadingParams m_params;
  float m_ni;
  float m_nt;

  MicrofacetReflectionDielectric m_coat_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
  MicrofacetReflectionConductor m_metal_brdf;
  MicrofacetTransmission m_transmission_btdf;
  OrenNayer m_diffuse_brdf;

  static __device__ float compute_F0(float ior_i, float ior_t)
  {
    const float t = (ior_t - ior_i) / (ior_t + ior_i);
    return t * t;
  }
};