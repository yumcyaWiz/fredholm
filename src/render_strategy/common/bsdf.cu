#pragma once

#include "bxdf.cu"
#include "lut.cu"
#include "math.cu"
#include "shared.h"
#include "util.cu"

class BSDF
{
   public:
    __device__ BSDF(const float3& wo, const ShadingParams& shading_params,
                    bool is_entering)

        : m_params(shading_params), m_is_entering(is_entering)
    {
        // init IOR
        m_ni = m_is_entering ? 1.0f : 1.5f;
        m_nt = m_is_entering ? 1.5f : 1.0f;
        m_eta = m_nt / m_ni;

        // compute luminance
        m_coat_color_luminance = rgb_to_luminance(m_params.coat_color);
        m_specular_color_luminance = rgb_to_luminance(m_params.specular_color);
        m_sheen_color_luminance = rgb_to_luminance(m_params.sheen_color);

        // compute coat absorption color
        m_coat_absorption_color =
            lerp(make_float3(1.0f),
                 m_params.coat_color * (1.0f - m_coat_directional_albedo),
                 m_params.coat);

        // compute directional albedo
        const float clearcoat_F0 = compute_F0(m_ni, m_nt);
        if (m_params.coat * m_coat_color_luminance > 0.0f)
        {
            m_coat_directional_albedo =
                m_is_entering ? compute_directional_albedo_reflection(
                                    wo, m_params.coat_roughness, clearcoat_F0)
                              : 0.0f;
        }

        if (m_params.specular * m_specular_color_luminance > 0.0f)
        {
            const float specular_F0 = compute_F0(m_ni, m_nt);
            m_specular_directional_albedo =
                m_eta >= 1.0f
                    ? compute_directional_albedo_reflection(
                          wo, m_params.specular_roughness, specular_F0)
                    : 0.0f;
        }

        if (m_params.sheen * m_sheen_color_luminance)
        {
            m_sheen_directional_albedo = m_is_entering
                                             ? compute_directional_albedo_sheen(
                                                   wo, m_params.sheen_roughness)
                                             : 0.0f;
        }

        // disable coat, metal, specular, sheen, diffuse reflection when
        // evaluating from inside
        m_params.coat = m_is_entering ? m_params.coat : 0.0f;
        m_params.metalness = m_is_entering ? m_params.metalness : 0.0f;
        m_params.specular = m_is_entering ? m_params.specular : 0.0f;
        m_params.sheen = m_is_entering ? m_params.sheen : 0.0f;
        m_params.diffuse = m_is_entering ? m_params.diffuse : 0.0f;

        // compute weights of each BxDF
        // coat, metal, specular, transmission, sheen, diffuse transmission,
        // diffuse reflection
        float weights[7];
        weights[0] = m_params.coat * m_coat_directional_albedo;
        weights[1] = (1.0f - m_params.coat * m_coat_directional_albedo) *
                     m_params.metalness;
        weights[2] = (1.0f - m_params.coat * m_coat_directional_albedo) *
                     (1.0f - m_params.metalness) * m_params.specular *
                     m_specular_directional_albedo;
        weights[3] =
            (1.0f - m_params.coat * m_coat_directional_albedo) *
            (1.0f - m_params.metalness) *
            (1.0f - m_params.specular * m_specular_directional_albedo) *
            m_params.transmission;
        weights[4] =
            (1.0f - m_params.coat * m_coat_directional_albedo) *
            (1.0f - m_params.metalness) *
            (1.0f - m_params.specular * m_specular_directional_albedo) *
            m_params.sheen * m_sheen_directional_albedo;
        weights[5] =
            (1.0f - m_params.coat * m_coat_directional_albedo) *
            (1.0f - m_params.metalness) *
            (1.0f - m_params.specular * m_specular_directional_albedo) *
            (1.0f - m_params.transmission) *
            (1.0f - m_params.sheen * m_sheen_directional_albedo) *
            m_params.subsurface * m_params.thin_walled;
        weights[6] =
            (1.0f - m_params.coat * m_coat_directional_albedo) *
            (1.0f - m_params.metalness) *
            (1.0f - m_params.specular * m_specular_directional_albedo) *
            (1.0f - m_params.transmission) *
            (1.0f - m_params.sheen * m_sheen_directional_albedo) *
            (1.0f - m_params.subsurface) * m_params.diffuse;

        // init distribution for sampling BxDF
        m_dist.init(weights, 7);

        // init each BxDF
        // TODO: add coat anisotropy
        m_coat_brdf = MicrofacetReflectionDielectric(
            m_eta, m_params.coat_roughness, 0.0f);

        // TODO: add specular anisotropy
        m_specular_brdf = MicrofacetReflectionDielectric(
            m_eta, m_params.specular_roughness, 0.0f);

        // TODO: add specular anisotropy
        float3 n, k;
        const float3 reflectivity =
            clamp(m_params.base_color, make_float3(0), make_float3(0.99));
        const float3 edge_tint =
            clamp(m_params.specular_color, make_float3(0), make_float3(0.99));
        artist_friendly_metallic_fresnel(reflectivity, edge_tint, n, k);
        m_metal_brdf = MicrofacetReflectionConductor(
            n, k, m_params.specular_roughness, 0.0f);

        // TODO: add specular anisotropy
        m_transmission_btdf = MicrofacetTransmission(
            m_ni, m_nt, m_params.specular_roughness, 0.0f);

        m_sheen_brdf = MicrofacetSheen(m_params.sheen_roughness);

        m_diffuse_btdf = DiffuseTransmission(m_params.base_color,
                                             m_params.diffuse_roughness);

        m_diffuse_brdf =
            OrenNayar(m_params.base_color, m_params.diffuse_roughness);
    }

    __device__ float3 eval(const float3& wo, const float3& wi) const
    {
        float3 coat = make_float3(0.0f);
        if (m_params.coat * m_coat_color_luminance > 0.0f)
        {
            coat = m_coat_brdf.eval(wo, wi);
            coat = (isinf(coat) || isnan(coat)) ? make_float3(0.0f) : coat;
        }

        float3 metal = make_float3(0.0f);
        if (m_params.metalness > 0.0f)
        {
            metal = m_metal_brdf.eval(wo, wi);
            metal = (isinf(metal) || isnan(metal)) ? make_float3(0.0f) : metal;
        }

        float3 specular = make_float3(0.0f);
        if (m_params.specular * m_specular_color_luminance > 0.0f)
        {
            specular = m_specular_brdf.eval(wo, wi);
            specular = (isinf(specular) || isnan(specular)) ? make_float3(0.0f)
                                                            : specular;
        }

        float3 transmission = make_float3(0.0f);
        if (m_params.transmission > 0.0f)
        {
            transmission = m_transmission_btdf.eval(wo, wi);
            transmission = (isinf(transmission) || isnan(transmission))
                               ? make_float3(0.0f)
                               : transmission;
        }

        float3 sheen = make_float3(0.0f);
        if (m_params.sheen * m_sheen_color_luminance > 0.0f)
        {
            sheen = m_sheen_brdf.eval(wo, wi);
            sheen = (isinf(sheen) || isnan(sheen)) ? make_float3(0.0f) : sheen;
        }

        float3 diffuse_t = make_float3(0.0f);
        if (m_params.subsurface * m_params.thin_walled > 0.0f)
        {
            diffuse_t = m_diffuse_btdf.eval(wo, wi);
            diffuse_t = (isinf(diffuse_t) || isnan(diffuse_t))
                            ? make_float3(0.0f)
                            : diffuse_t;
        }

        float3 diffuse_r = make_float3(0.0f);
        if (m_params.diffuse > 0.0f)
        {
            diffuse_r = m_diffuse_brdf.eval(wo, wi);
            diffuse_r = (isinf(diffuse_r) || isnan(diffuse_r))
                            ? make_float3(0.0f)
                            : diffuse_r;
        }

        float3 ret = make_float3(0.0f);
        float3 f_mult = make_float3(1.0f);

        // coat
        ret += m_params.coat * coat;
        f_mult *= m_coat_absorption_color;

        // metal
        ret += f_mult * m_params.metalness * metal;
        f_mult *= (1.0f - m_params.metalness);

        // specular
        ret += f_mult * m_params.specular * m_params.specular_color * specular;
        f_mult *= (1.0f - m_params.specular * m_params.specular_color *
                              m_specular_directional_albedo);

        // transmission
        ret += f_mult * m_params.transmission * m_params.transmission_color *
               transmission;
        f_mult *= (1.0f - m_params.transmission);

        // sheen
        ret += f_mult * m_params.sheen * m_params.sheen_color * sheen;
        f_mult *= (1.0f - m_params.sheen * m_sheen_directional_albedo);

        // diffuse transmission
        ret += f_mult * m_params.subsurface * m_params.subsurface_color *
               m_params.thin_walled * diffuse_t;
        f_mult *= (1.0f - m_params.subsurface);

        // diffuse
        ret += f_mult * m_params.diffuse * diffuse_r;

        return ret;
    }

    __device__ float3 sample(const float3& wo, float u, const float2& v,
                             float3& f, float& pdf) const
    {
        // sample BxDF
        float bxdf_pdf;
        const int bxdf_idx = m_dist.sample(u, bxdf_pdf);

        switch (bxdf_idx)
        {
            // coat
            case 0:
            {
                const float3 wi = m_coat_brdf.sample(wo, v, f, pdf);
                f *= m_params.coat;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // metal
            case 1:
            {
                const float3 wi = m_metal_brdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * m_params.metalness;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // specular
            case 2:
            {
                const float3 wi = m_specular_brdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * (1.0f - m_params.metalness) *
                     m_params.specular * m_params.specular_color;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // transmission
            case 3:
            {
                const float3 wi = m_transmission_btdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * (1.0f - m_params.metalness) *
                     (1.0f - m_params.specular * m_params.specular_color *
                                 m_specular_directional_albedo) *
                     m_params.transmission * m_params.transmission_color;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // sheen
            case 4:
            {
                const float3 wi = m_sheen_brdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * (1.0f - m_params.metalness) *
                     (1.0f - m_params.specular * m_params.specular_color *
                                 m_specular_directional_albedo) *
                     (1.0f - m_params.transmission) * m_params.sheen *
                     m_params.sheen_color;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // diffuse transmission
            case 5:
            {
                const float3 wi = m_diffuse_btdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * (1.0f - m_params.metalness) *
                     (1.0f - m_params.specular * m_params.specular_color *
                                 m_specular_directional_albedo) *
                     (1.0f - m_params.transmission) *
                     (1.0f - m_params.sheen * m_sheen_directional_albedo) *
                     m_params.subsurface * m_params.subsurface_color *
                     m_params.thin_walled;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
            // diffuse reflection
            case 6:
            {
                const float3 wi = m_diffuse_brdf.sample(wo, v, f, pdf);
                f *= m_coat_absorption_color * (1.0f - m_params.metalness) *
                     (1.0f - m_params.specular * m_params.specular_color *
                                 m_specular_directional_albedo) *
                     (1.0f - m_params.transmission) *
                     (1.0f - m_params.sheen * m_sheen_directional_albedo) *
                     (1.0f - m_params.subsurface) * m_params.diffuse;
                pdf *= bxdf_pdf;
                return wi;
            }
            break;
        }

        return make_float3(0.0f);
    }

    __device__ float eval_pdf(const float3& wo, const float3& wi) const
    {
        // evaluate each BxDF pdf
        float coat = 0.0f;
        if (m_params.coat * m_coat_color_luminance > 0.0f)
        {
            coat = m_coat_brdf.eval_pdf(wo, wi);
            coat = (isinf(coat) || isnan(coat)) ? 0.0f : coat;
        }

        float metal = 0.0f;
        if (m_params.metalness > 0.0f)
        {
            metal = m_metal_brdf.eval_pdf(wo, wi);
            metal = (isinf(metal) || isnan(metal)) ? 0.0f : metal;
        }

        float specular = 0.0f;
        if (m_params.specular * m_specular_color_luminance > 0.0f)
        {
            specular = m_specular_brdf.eval_pdf(wo, wi);
            specular = (isinf(specular) || isnan(specular)) ? 0.0f : specular;
        }

        float transmission = 0.0f;
        if (m_params.transmission > 0.0f)
        {
            transmission = m_transmission_btdf.eval_pdf(wo, wi);
            transmission = (isinf(transmission) || isnan(transmission))
                               ? 0.0f
                               : transmission;
        }

        float sheen = 0.0f;
        if (m_params.sheen * m_sheen_color_luminance > 0.0f)
        {
            sheen = m_sheen_brdf.eval_pdf(wo, wi);
            sheen = (isinf(sheen) || isnan(sheen)) ? 0.0f : sheen;
        }

        float diffuse_t = 0.0f;
        if (m_params.subsurface * m_params.thin_walled > 0.0f)
        {
            diffuse_t = m_diffuse_btdf.eval_pdf(wo, wi);
            diffuse_t =
                (isinf(diffuse_t) || isnan(diffuse_t)) ? 0.0f : diffuse_t;
        }

        float diffuse_r = 0.0f;
        if (m_params.diffuse > 0.0f)
        {
            diffuse_r = m_diffuse_brdf.eval_pdf(wo, wi);
            diffuse_r =
                (isinf(diffuse_r) || isnan(diffuse_r)) ? 0.0f : diffuse_r;
        }

        return m_dist.eval_pmf(0) * coat + m_dist.eval_pmf(1) * metal +
               m_dist.eval_pmf(2) * specular +
               m_dist.eval_pmf(3) * transmission + m_dist.eval_pmf(4) * sheen +
               m_dist.eval_pmf(5) * diffuse_t + m_dist.eval_pmf(6) * diffuse_r;
    }

   private:
    ShadingParams m_params;
    bool m_is_entering;
    float m_ni;
    float m_nt;
    float m_eta;

    MicrofacetReflectionDielectric m_coat_brdf;
    MicrofacetReflectionDielectric m_specular_brdf;
    MicrofacetReflectionConductor m_metal_brdf;
    MicrofacetTransmission m_transmission_btdf;
    MicrofacetSheen m_sheen_brdf;
    DiffuseTransmission m_diffuse_btdf;
    OrenNayar m_diffuse_brdf;

    float3 m_coat_absorption_color = make_float3(1.0f);
    float m_coat_color_luminance = 0.0f;
    float m_coat_directional_albedo = 0.0f;

    float m_specular_color_luminance = 0.0f;
    float m_specular_directional_albedo = 0.0f;

    float m_sheen_color_luminance = 0.0f;
    float m_sheen_directional_albedo = 0.0f;

    DiscreteDistribution1D m_dist;

    static __device__ float compute_F0(float ior_i, float ior_t)
    {
        const float t = (ior_t - ior_i) / (ior_t + ior_i);
        return t * t;
    }
};