#pragma once

#include <emmintrin.h>

#include "bxdf.cu"
#include "fredholm/shared.h"
#include "math.cu"

class BSDF
{
 public:
  __device__ BSDF(const ShadingParams& shading_params, bool is_entering,
                  const cudaTextureObject_t& lut)
      : m_params(shading_params), m_lut(lut)
  {
    const float n_i = is_entering ? 1.0f : 1.5f;
    const float n_t = is_entering ? 1.5f : 1.0f;
    const float eta = n_t / n_i;
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
        MicrofacetTransmission(n_i, n_t, m_params.specular_roughness, 0.0f);

    m_diffuse_brdf = Lambert(m_params.base_color);
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
    const float clearcoat_F0 = compute_F0(1.0f, 1.5f);
    const float clearcoat_directional_albedo =
        fetch_directional_albedo(wo, m_params.coat_roughness, clearcoat_F0);
    const float coat_color_luminance = rgb_to_luminance(m_params.coat_color);
    if (u.x <
        m_params.coat * coat_color_luminance * clearcoat_directional_albedo) {
      const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
      f *= m_params.coat * m_params.coat_color;
      pdf *=
          m_params.coat * coat_color_luminance * clearcoat_directional_albedo;
      return wi;
    }
    // metal or transmission or specular or diffuse
    else {
      float3 f_mult = (1.0f - m_params.coat * m_params.coat_color *
                                  clearcoat_directional_albedo);
      float pdf_mult = (1.0f - m_params.coat * coat_color_luminance *
                                   clearcoat_directional_albedo);

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

        // TODO: set ior
        const float specular_F0 = compute_F0(1.0f, 1.5f);
        const float specular_directional_albedo = fetch_directional_albedo(
            wo, m_params.specular_roughness, specular_F0);
        const float specular_color_luminance =
            rgb_to_luminance(m_params.specular_color);
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

  MicrofacetReflectionDielectric m_coat_brdf;
  MicrofacetReflectionDielectric m_specular_brdf;
  MicrofacetReflectionConductor m_metal_brdf;
  MicrofacetTransmission m_transmission_btdf;
  Lambert m_diffuse_brdf;

  cudaTextureObject_t m_lut;

  __device__ float fetch_directional_albedo(const float3& wo, float roughness,
                                            float F0) const
  {
    const float cos = abs_cos_theta(wo);
    // TODO: 1.0f - roughness? since image is y-flipped
    const float4 RGBA = tex2D<float4>(m_lut, cos, roughness);
    return F0 * RGBA.x + fmax(1.0f - F0, 0.0f) * RGBA.y;
  }

  static __device__ float compute_F0(float ior_i, float ior_t)
  {
    const float t = (ior_t - ior_i) / (ior_t + ior_i);
    return t * t;
  }
};