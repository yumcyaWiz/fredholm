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
    m_eta = m_nt / m_ni;

    // init BxDF
    m_coat_brdf =
        MicrofacetReflectionDielectric(m_eta, m_params.coat_roughness, 0.0f);

    m_specular_brdf = MicrofacetReflectionDielectric(
        m_eta, m_params.specular_roughness, 0.0f);

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
    float3 diffuse = m_diffuse_brdf.eval(wo, wi);
    diffuse = (isinf(diffuse) || isnan(diffuse)) ? make_float3(0.0f) : diffuse;

    float3 specular = m_specular_brdf.eval(wo, wi);
    specular =
        (isinf(specular) || isnan(specular)) ? make_float3(0.0f) : specular;

    float3 metal = m_metal_brdf.eval(wo, wi);
    metal = (isinf(metal) || isnan(metal)) ? make_float3(0.0f) : metal;

    float3 coat = m_coat_brdf.eval(wo, wi);
    coat = (isinf(coat) || isnan(coat)) ? make_float3(0.0f) : coat;

    float3 transmission = m_transmission_btdf.eval(wo, wi);
    transmission = (isinf(transmission) || isnan(transmission))
                       ? make_float3(0.0f)
                       : transmission;

    const float clearcoat_F0 = compute_F0(m_ni, m_nt);
    const float coat_directional_albedo =
        compute_directional_albedo(wo, m_params.coat_roughness, clearcoat_F0);

    const float specular_F0 = compute_F0(m_ni, m_nt);
    float specular_directional_albedo =
        m_eta >= 1.0f ? compute_directional_albedo(
                            wo, m_params.specular_roughness, specular_F0)
                      : compute_directional_albedo2(
                            wo, m_params.specular_roughness, m_nt / m_ni);

    float3 ret = make_float3(0.0f);
    float3 f_mult = make_float3(1.0f);

    // coat
    ret += m_params.coat * coat;
    f_mult *= lerp(make_float3(1.0f),
                   m_params.coat_color * (1.0f - coat_directional_albedo),
                   m_params.coat);

    // metal
    ret += f_mult * m_params.metalness * metal;
    f_mult *= (1.0f - m_params.metalness);

    // specular
    ret += f_mult * m_params.specular * m_params.specular_color * specular;
    f_mult *= (1.0f - m_params.specular * m_params.specular_color *
                          specular_directional_albedo);

    // transmission
    ret += f_mult * m_params.transmission * m_params.transmission_color *
           transmission;
    f_mult *= (1.0f - m_params.transmission);

    // diffuse
    ret += f_mult * diffuse;

    return ret;
  }

  __device__ float3 sample(const float3& wo, float u, const float2& v,
                           float3& f, float& pdf) const
  {
    const float clearcoat_F0 = compute_F0(m_ni, m_nt);
    const float coat_directional_albedo =
        compute_directional_albedo(wo, m_params.coat_roughness, clearcoat_F0);
    const float3 coat_absorption_color = lerp(
        make_float3(1.0f),
        m_params.coat_color * (1.0f - coat_directional_albedo), m_params.coat);

    const float specular_F0 = compute_F0(m_ni, m_nt);
    const float specular_color_luminance =
        rgb_to_luminance(m_params.specular_color);
    float specular_directional_albedo =
        m_eta >= 1.0f ? compute_directional_albedo(
                            wo, m_params.specular_roughness, specular_F0)
                      : compute_directional_albedo2(
                            wo, m_params.specular_roughness, m_nt / m_ni);

    // compute weights of each BxDF
    // coat, metal, specular, transmission, diffuse
    float weights[5];
    weights[0] = m_params.coat * coat_directional_albedo;
    weights[1] =
        (1.0f - m_params.coat * coat_directional_albedo) * m_params.metalness;
    weights[2] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) * m_params.specular *
                 specular_directional_albedo;
    weights[3] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) *
                 (1.0f - m_params.specular * specular_directional_albedo) *
                 m_params.transmission;
    weights[4] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) *
                 (1.0f - m_params.specular * specular_directional_albedo) *
                 (1.0f - m_params.transmission);

    // sample BxDF
    DiscreteDistribution1D dist(weights, 5);
    float bxdf_pdf;
    const int bxdf_idx = dist.sample(u, bxdf_pdf);

    switch (bxdf_idx) {
      // coat
      case 0: {
        const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
        f *= m_params.coat;
        pdf *= bxdf_pdf;
        return wi;
      } break;
      // metal
      case 1: {
        const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
        f *= coat_absorption_color * m_params.metalness;
        pdf *= bxdf_pdf;
        return wi;
      } break;
      // specular
      case 2: {
        const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
        f *= coat_absorption_color * (1.0f - m_params.metalness) *
             m_params.specular * m_params.specular_color;
        pdf *= bxdf_pdf;
        return wi;
      } break;
      // transmission
      case 3: {
        const float3 wi = m_transmission_btdf.sample(wo, v, f, pdf);
        f *= coat_absorption_color * (1.0f - m_params.metalness) *
             (1.0f - m_params.specular * m_params.specular_color *
                         specular_directional_albedo) *
             m_params.transmission * m_params.transmission_color;
        pdf *= bxdf_pdf;
        return wi;
      } break;
      // diffuse
      case 4: {
        const float3 wi = m_diffuse_brdf.sample(wo, v, f, pdf);
        f *= coat_absorption_color * (1.0f - m_params.metalness) *
             (1.0f - m_params.specular * m_params.specular_color *
                         specular_directional_albedo) *
             (1.0f - m_params.transmission);
        pdf *= bxdf_pdf;
        return wi;
      } break;
    }

    return make_float3(0.0f);
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    // evaluate each BxDF pdf
    float coat = m_coat_brdf.eval_pdf(wo, wi);
    coat = (isinf(coat) || isnan(coat)) ? 0.0f : coat;

    float metal = m_metal_brdf.eval_pdf(wo, wi);
    metal = (isinf(metal) || isnan(metal)) ? 0.0f : metal;

    float specular = m_specular_brdf.eval_pdf(wo, wi);
    specular = (isinf(specular) || isnan(specular)) ? 0.0f : specular;

    float transmission = m_transmission_btdf.eval_pdf(wo, wi);
    transmission =
        (isinf(transmission) || isnan(transmission)) ? 0.0f : transmission;

    float diffuse = m_diffuse_brdf.eval_pdf(wo, wi);
    diffuse = (isinf(diffuse) || isnan(diffuse)) ? 0.0f : diffuse;

    const float clearcoat_F0 = compute_F0(m_ni, m_nt);
    const float coat_directional_albedo =
        compute_directional_albedo(wo, m_params.coat_roughness, clearcoat_F0);
    const float3 coat_absorption_color = lerp(
        make_float3(1.0f),
        m_params.coat_color * (1.0f - coat_directional_albedo), m_params.coat);

    const float specular_F0 = compute_F0(m_ni, m_nt);
    const float specular_color_luminance =
        rgb_to_luminance(m_params.specular_color);
    float specular_directional_albedo =
        m_eta >= 1.0f ? compute_directional_albedo(
                            wo, m_params.specular_roughness, specular_F0)
                      : compute_directional_albedo2(
                            wo, m_params.specular_roughness, m_nt / m_ni);

    // compute weights of each BxDF
    // coat, metal, specular, transmission, diffuse
    float weights[5];
    weights[0] = m_params.coat * coat_directional_albedo;
    weights[1] =
        (1.0f - m_params.coat * coat_directional_albedo) * m_params.metalness;
    weights[2] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) * m_params.specular *
                 specular_directional_albedo;
    weights[3] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) *
                 (1.0f - m_params.specular * specular_directional_albedo) *
                 m_params.transmission;
    weights[4] = (1.0f - m_params.coat * coat_directional_albedo) *
                 (1.0f - m_params.metalness) *
                 (1.0f - m_params.specular * specular_directional_albedo) *
                 (1.0f - m_params.transmission);

    DiscreteDistribution1D dist(weights, 5);

    return dist.eval_pmf(0) * diffuse + dist.eval_pmf(1) * metal +
           dist.eval_pmf(2) * specular + dist.eval_pmf(3) * transmission +
           dist.eval_pmf(4) * diffuse;
  }

 private:
  ShadingParams m_params;
  float m_ni;
  float m_nt;
  float m_eta;

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