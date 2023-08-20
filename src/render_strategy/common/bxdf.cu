#pragma once

#include "cuda_util.h"
#include "helper_math.h"
#include "math.cu"
#include "sampling.cu"

CUDA_INLINE CUDA_DEVICE float cos_theta(const float3& w) { return w.y; }

CUDA_INLINE CUDA_DEVICE float cos2_theta(const float3& w) { return w.y * w.y; }

CUDA_INLINE CUDA_DEVICE float abs_cos_theta(const float3& w)
{
    return fabs(w.y);
}

CUDA_INLINE CUDA_DEVICE float sin_theta(const float3& w)
{
    return sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

CUDA_INLINE CUDA_DEVICE float sin2_theta(const float3& w)
{
    return fmax(1.0f - w.y * w.y, 0.0f);
}

CUDA_INLINE CUDA_DEVICE float abs_sin_theta(const float3& w)
{
    return fabs(sin_theta(w));
}

CUDA_INLINE CUDA_DEVICE float tan_theta(const float3& w)
{
    return sin_theta(w) / cos_theta(w);
}

CUDA_INLINE CUDA_DEVICE float tan2_theta(const float3& w)
{
    return 1.0f / (w.y * w.y) - 1.0f;
}

CUDA_INLINE CUDA_DEVICE float abs_tan_theta(const float3& w)
{
    return fabs(tan_theta(w));
}

CUDA_INLINE CUDA_DEVICE float sin_phi(const float3& w)
{
    return w.z / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

CUDA_INLINE CUDA_DEVICE float sin2_phi(const float3& w)
{
    return w.z * w.z / fmax(1.0f - w.y * w.y, 0.0f);
}

CUDA_INLINE CUDA_DEVICE float abs_sin_phi(const float3& w)
{
    return fabs(sin_phi(w));
}

CUDA_INLINE CUDA_DEVICE float cos_phi(const float3& w)
{
    return w.x / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

CUDA_INLINE CUDA_DEVICE float cos2_phi(const float3& w)
{
    return w.x * w.x / fmax(1.0f - w.y * w.y, 0.0f);
}

CUDA_INLINE CUDA_DEVICE float abs_cos_phi(const float3& w)
{
    return fabs(cos_phi(w));
}

CUDA_INLINE CUDA_DEVICE float3 reflect(const float3& w, const float3& n)
{
    return normalize(-w + 2.0f * dot(w, n) * n);
}

CUDA_INLINE CUDA_DEVICE bool refract(const float3& w, const float3& n,
                                     float ior_i, float ior_t, float3& wt)
{
    const float3 th = -ior_i / ior_t * (w - dot(w, n) * n);
    if (dot(th, th) > 1.0f) return false;
    const float3 tp = -sqrtf(fmax(1.0f - dot(th, th), 0.0f)) * n;
    wt = th + tp;
    return true;
}

CUDA_INLINE CUDA_DEVICE float2 roughness_to_alpha(float roughness,
                                                  float anisotropy)
{
    // Revisiting Physically Based Shading at Imageworks p.24
    float2 alpha;
    alpha.x = roughness * roughness * (1.0f + anisotropy);
    alpha.y = roughness * roughness * (1.0f - anisotropy);
    return alpha;
}

// https://jcgt.org/published/0003/04/03/
CUDA_INLINE CUDA_DEVICE void artist_friendly_metallic_fresnel(
    const float3& reflectivity, const float3& edge_tint, float3& n, float3& k)
{
    const float3 r_sqrt = sqrtf(reflectivity);
    n = edge_tint * (1.0f - reflectivity) / (1.0f + reflectivity) +
        (1.0f - edge_tint) * (1.0f + r_sqrt) / (1.0f - r_sqrt);
    const float3 t1 = n + 1.0f;
    const float3 t2 = n - 1.0f;
    k = sqrtf((reflectivity * (t1 * t1) - t2 * t2) / (1.0f - reflectivity));
}

// Lambert BRDF
class Lambert
{
   public:
    CUDA_INLINE CUDA_DEVICE Lambert() {}
    CUDA_INLINE CUDA_DEVICE Lambert(const float3& albedo) : m_albedo(albedo) {}

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        return m_albedo / M_PIf;
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
    {
        const float3 wi = sample_cosine_weighted_hemisphere(u);

        f = eval(wo, wi);
        pdf = abs_cos_theta(wi) / M_PIf;

        return wi;
    }

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        return abs_cos_theta(wi) / M_PIf;
    }

   private:
    float3 m_albedo;
};

// Oren-Nayar Diffuse BRDF
class OrenNayar
{
   public:
    CUDA_INLINE CUDA_DEVICE OrenNayar() {}
    CUDA_INLINE CUDA_DEVICE OrenNayar(const float3& albedo, float roughness)
        : m_albedo(albedo), m_roughness(roughness)
    {
        const float sigma2 = roughness * roughness;
        m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
        m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float s_theta_o = sin_theta(wo);
        const float s_theta_i = sin_theta(wi);

        float c_max = 0.0f;
        if (s_theta_i > 1e-4f && s_theta_o > 1e-4f)
        {
            const float s_phi_o = sin_phi(wo), c_phi_o = cos_phi(wo);
            const float s_phi_i = sin_phi(wi), c_phi_i = cos_phi(wi);
            const float c = c_phi_i * c_phi_o + s_phi_i * s_phi_o;
            c_max = fmax(c, 0.0f);
        }

        const bool b = abs_cos_theta(wi) > abs_cos_theta(wo);
        const float s_alpha = b ? s_theta_o : s_theta_i;
        const float t_beta =
            b ? s_theta_i / abs_cos_theta(wi) : s_theta_o / abs_cos_theta(wo);

        return m_albedo * (m_A + m_B * c_max * s_alpha * t_beta) / M_PIf;
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
    {
        const float3 wi = sample_cosine_weighted_hemisphere(u);

        f = eval(wo, wi);
        pdf = abs_cos_theta(wi) / M_PIf;

        return wi;
    }

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        return abs_cos_theta(wi) / M_PIf;
    }

   private:
    float3 m_albedo;
    float m_roughness;
    float m_A;
    float m_B;
};

// The diffuse transmission is modeled via a Oren-Nayar microfacet BRDF flipped
// about the shading normal to make it a BTDF
class DiffuseTransmission
{
   public:
    CUDA_INLINE CUDA_DEVICE DiffuseTransmission() {}
    CUDA_INLINE CUDA_DEVICE DiffuseTransmission(const float3& albedo,
                                                float roughness)
        : m_albedo(albedo), m_roughness(roughness)
    {
        const float sigma2 = roughness * roughness;
        m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
        m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float s_theta_o = sin_theta(wo);
        const float s_theta_i = sin_theta(wi);

        float c_max = 0.0f;
        if (s_theta_i > 1e-4f && s_theta_o > 1e-4f)
        {
            const float s_phi_o = sin_phi(wo), c_phi_o = cos_phi(wo);
            const float s_phi_i = sin_phi(wi), c_phi_i = cos_phi(wi);
            const float c = c_phi_i * c_phi_o + s_phi_i * s_phi_o;
            c_max = fmax(c, 0.0f);
        }

        const bool b = abs_cos_theta(wi) > abs_cos_theta(wo);
        const float s_alpha = b ? s_theta_o : s_theta_i;
        const float t_beta =
            b ? s_theta_i / abs_cos_theta(wi) : s_theta_o / abs_cos_theta(wo);

        return m_albedo * (m_A + m_B * c_max * s_alpha * t_beta) / M_PIf;
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
    {
        float3 wi = sample_cosine_weighted_hemisphere(u);
        wi = -wi;  // flip direction

        f = eval(wo, wi);
        pdf = abs_cos_theta(wi) / M_PIf;

        return wi;
    }

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        return abs_cos_theta(wi) / M_PIf;
    }

   private:
    float3 m_albedo;
    float m_roughness;
    float m_A;
    float m_B;
};

// Schlick approximation fresnel
CUDA_INLINE CUDA_DEVICE float fresnel_schlick(float cos, float F0)
{
    const float t = fmax(1.0f - cos, 0.0f);
    return F0 + fmax(1.0f - F0, 0.0f) * t * t * t * t * t;
}

// Dielectric fresnel
CUDA_INLINE CUDA_DEVICE float fresnel_dielectric(float cos, float ior)
{
    const float temp = ior * ior + cos * cos - 1.0f;
    if (temp < 0.0f) { return 1.0f; }

    const float g = sqrtf(temp);
    const float t0 = (g - cos) / (g + cos);
    const float t1 = ((g + cos) * cos - 1.0f) / ((g - cos) * cos + 1.0f);
    return 0.5f * t0 * t0 * (1.0f + t1 * t1);
}

// Conductor fresnel
CUDA_INLINE CUDA_DEVICE float3 fresnel_conductor(float cos, const float3& ior,
                                                 const float3& k)
{
    const float c2 = cos * cos;
    const float3 two_eta_cos = 2.0f * ior * cos;

    const float3 t0 = ior * ior + k * k;
    const float3 t1 = t0 * c2;
    const float3 Rs = (t0 - two_eta_cos + c2) / (t0 + two_eta_cos + c2);
    const float3 Rp = (t1 - two_eta_cos + 1.0f) / (t1 + two_eta_cos + 1.0f);

    return 0.5f * (Rp + Rs);
}

CUDA_INLINE CUDA_DEVICE void fresnel_dielectric_poralized(
    float cos, float ior1, float ior2, float& R_p, float& R_s, float& phi_p,
    float& phi_s)
{
    const float sin = 1.0 - cos * cos;
    const float eta = ior1 / ior2;

    // Total internal reflection
    if (eta * eta * sin > 1.0f)
    {
        R_p = 1.0f;
        R_s = 1.0f;
        phi_p = 2.0f * atanf(-eta * eta * sqrtf(sin - 1.0 / (eta * eta)) / cos);
        phi_s = 2.0f * atanf(-sqrtf(sin - 1.0f / (eta * eta)) / cos);
    }

    const float cos2 = sqrtf(1.0f - eta * eta * sin);
    const float r_p = (ior2 * cos - ior1 * cos2) / (ior2 * cos + ior1 * cos2);
    const float r_s = (ior1 * cos - ior2 * cos2) / (ior1 * cos + ior2 * cos2);
    R_p = r_p * r_p;
    R_s = r_s * r_s;
    phi_p = (r_p < 0.0f) ? M_PIf : 0.0f;
    phi_s = (r_s < 0.0f) ? M_PIf : 0.0f;
}

CUDA_INLINE CUDA_DEVICE void fresnel_conductor_poralized(
    float cos, float ior1, const float3& ior2, const float3& k2, float3& R_p,
    float3& R_s, float3& phi_p, float3& phi_s)
{
    if (k2.x == 0.0f && k2.y == 0.0f && k2.z == 0.0f)
    {
        fresnel_dielectric_poralized(cos, ior1, ior2.x, R_p.x, R_s.x, phi_p.x,
                                     phi_s.x);
        R_p = make_float3(R_p.x);
        R_s = make_float3(R_s.x);
        phi_p = make_float3(phi_p.x);
        phi_s = make_float3(phi_s.x);
    }

    const float3 A =
        ior2 * ior2 * (1.0f - k2 * k2) - ior1 * ior1 * (1.0f - cos * cos);
    const float3 B = sqrtf(A * A + square(2.0f * ior2 * ior2 * k2));
    const float3 U = sqrtf(0.5f * (A + B));
    const float3 V = sqrtf(0.5f * (B - A));

    R_s = (square(ior1 * cos - U) + V * V) / (square(ior1 * cos + U) + V * V);
    phi_s = atan2f(2.0f * ior1 * V * cos, U * U + V * V - (ior1 * cos)) + M_PIf;
    R_p = (square(ior2 * ior2 + (1.0f - k2 * k2) * cos - ior1 * U) +
           square(2.0f * ior2 * ior2 * k2 * cos - ior1 * V)) /
          (square(ior2 * ior2 * (1.0f - k2 * k2) * cos + ior1 * U) +
           square(2.0f * ior2 * ior2 * k2 * cos + ior1 * V));
    phi_p = atan2f(2.0f * ior1 * ior2 * ior2 * cos *
                       (2.0f * k2 * U - (1.0f - k2 * k2) * V),
                   square(ior2 * ior2 * (1.0f + k2 * k2) * cos) -
                       ior1 * ior1 * (U * U + V * V));
}

// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
CUDA_INLINE CUDA_DEVICE float3 eval_sensitivity(float opd, const float3& shift)
{
    const float phase = 2.0f * M_PIf * opd;
    const float3 val = make_float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    const float3 pos = make_float3(1.6810e6f, 1.7953e6f, 2.2084e6f);
    const float3 var = make_float3(4.3278e9f, 9.3046e9f, 6.6121e9f);
    float3 xyz = val * sqrtf(2.0f * M_PIf * var) * cosf(pos * phase + shift) *
                 expf(-var * phase * phase);
    xyz.x += 9.7470e-14f * sqrtf(2.0f * M_PIf * 4.5282e9f) *
             cosf(2.2399e6f * phase + shift.x) *
             expf(-4.5282e9f * phase * phase);
    xyz /= 1.0685e-7;

    return xyz_to_rgb(xyz);
}

// Thin film interference
// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
CUDA_INLINE CUDA_DEVICE float3 fresnel_airy(float cos, float ior1, float ior2,
                                            float thickness, const float3& ior3,
                                            const float3& k3)
{
    float R12p, R12s, phi12p, phi12s;
    fresnel_dielectric_poralized(cos, ior1, ior2, R12p, R12s, phi12p, phi12s);
    const float T12p = 1.0f - R12p;
    const float T12s = 1.0f - R12s;

    const float s1 = 1.0f - cos * cos;
    const float eta = ior1 / ior2;
    const float c2 = sqrtf(1.0f - eta * eta * s1);

    const float phi21p = M_PIf - phi12p;
    const float phi21s = M_PIf - phi12s;

    float3 R23p, R23s, phi23p, phi23s;
    fresnel_conductor_poralized(cos, ior2, ior3, k3, R23p, R23s, phi23p,
                                phi23s);

    const float opd = 2.0f * ior2 * (thickness * 1e-9f) * c2;
    const float3 phi2p = phi21p + phi23p;
    const float3 phi2s = phi21s + phi23s;

    const float T121p = T12p * T12p;
    const float3 Rsp = T121p * R23p / (1.0f - R23p * R12p);
    const float T121s = T12s * T12s;
    const float3 Rss = T121s * R23s / (1.0f - R23s * R12s);

    // m = 0
    float3 I = make_float3(0.0f);
    const float3 C0 = (R12p + Rsp + R12s + Rss);
    I += C0;

    // m > 0
    float3 Cmp = Rsp - sqrtf(T121p);
    float3 Cms = Rss - sqrtf(T121s);
    for (int m = 1; m <= 3; ++m)
    {
        Cmp *= sqrtf(R23p * R12p);
        Cms *= sqrtf(R23s * R12s);
        const float3 Sp = 2.0f * eval_sensitivity(m * opd, m * phi2p);
        const float3 Ss = 2.0f * eval_sensitivity(m * opd, m * phi2s);
        I += (Cmp * Sp + Cms * Ss);
    }

    // average
    I *= 0.5f;

    return clamp(I, make_float3(0.0f), make_float3(1.0f));
}

// Microfacet(GGX) with dielectric fresnel
// TODO: use template parameter for fresnel term?
class MicrofacetReflectionDielectric
{
   public:
    CUDA_INLINE CUDA_DEVICE MicrofacetReflectionDielectric() {}
    CUDA_INLINE CUDA_DEVICE MicrofacetReflectionDielectric(
        float ior, float roughness, float anisotropy,
        float thin_film_thickness = 0.0f, float thin_film_ior = 1.5f)
        : m_ior(ior),
          m_thin_film_thickness(thin_film_thickness),
          m_thin_film_ior(thin_film_ior)
    {
        m_alpha = roughness_to_alpha(roughness, anisotropy);
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float3 wh = normalize(wo + wi);

        float3 f;
        if (m_thin_film_thickness > 0.0f)
        {
            f = fresnel_airy(fabs(dot(wo, wh)), 1.0f, m_thin_film_ior,
                             m_thin_film_thickness, make_float3(m_ior),
                             make_float3(0.0f));
        }
        else { f = make_float3(fresnel_dielectric(fabs(dot(wo, wh)), m_ior)); }

        const float d = D(wh);
        const float g = G2(wo, wi);
        return 0.25f * (f * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
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

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        const float3 wh = normalize(wo + wi);
        return 0.25f * D_visible(wo, wh) / fabs(dot(wo, wh));
    }

   private:
    CUDA_INLINE CUDA_DEVICE float D(const float3& wh) const
    {
        const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                        wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
        return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
    }

    CUDA_INLINE CUDA_DEVICE float D_visible(const float3& w,
                                            const float3& wh) const
    {
        return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
    }

    CUDA_INLINE CUDA_DEVICE float lambda(const float3& w) const
    {
        const float t = (m_alpha.x * m_alpha.x * w.x * w.x +
                         m_alpha.y * m_alpha.y * w.z * w.z) /
                        (w.y * w.y);
        return 0.5f * (-1.0f + sqrtf(1.0f + t));
    }

    CUDA_INLINE CUDA_DEVICE float G1(const float3& w) const
    {
        return 1.0f / (1.0f + lambda(w));
    }

    CUDA_INLINE CUDA_DEVICE float G2(const float3& wo, const float3& wi) const
    {
        return 1.0f / (1.0f + lambda(wo) + lambda(wi));
    }

    float m_ior;
    float2 m_alpha;
    float m_thin_film_thickness;
    float m_thin_film_ior;
};

// Microfacet(GGX) with conductor fresnel
// TODO: use template parameter for fresnel term?
class MicrofacetReflectionConductor
{
   public:
    CUDA_INLINE CUDA_DEVICE MicrofacetReflectionConductor() {}
    CUDA_INLINE CUDA_DEVICE MicrofacetReflectionConductor(
        const float3& ior, const float3& k, float roughness, float anisotropy,
        float thin_film_thickness = 0.0f, float thin_film_ior = 1.5f)
        : m_ior(ior),
          m_k(k),
          m_thin_film_thickness(thin_film_thickness),
          m_thin_film_ior(thin_film_ior)
    {
        m_alpha = roughness_to_alpha(roughness, anisotropy);
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float3 wh = normalize(wo + wi);
        float3 f;
        if (m_thin_film_thickness > 0.0f)
        {
            f = fresnel_airy(fabs(dot(wo, wh)), 1.0f, m_thin_film_ior,
                             m_thin_film_thickness, m_ior, m_k);
        }
        else { f = fresnel_conductor(fabs(dot(wo, wh)), m_ior, m_k); }
        const float d = D(wh);
        const float g = G2(wo, wi);
        return 0.25f * (f * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
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

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        const float3 wh = normalize(wo + wi);
        return 0.25f * D_visible(wo, wh) / fabs(dot(wo, wh));
    }

   private:
    CUDA_INLINE CUDA_DEVICE float D(const float3& wh) const
    {
        const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                        wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
        return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
    }

    CUDA_INLINE CUDA_DEVICE float D_visible(const float3& w,
                                            const float3& wh) const
    {
        return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
    }

    CUDA_INLINE CUDA_DEVICE float lambda(const float3& w) const
    {
        const float t = (m_alpha.x * m_alpha.x * w.x * w.x +
                         m_alpha.y * m_alpha.y * w.z * w.z) /
                        (w.y * w.y);
        return 0.5f * (-1.0f + sqrtf(1.0f + t));
    }

    CUDA_INLINE CUDA_DEVICE float G1(const float3& w) const
    {
        return 1.0f / (1.0f + lambda(w));
    }

    CUDA_INLINE CUDA_DEVICE float G2(const float3& wo, const float3& wi) const
    {
        return 1.0f / (1.0f + lambda(wo) + lambda(wi));
    }

    float3 m_ior;
    float3 m_k;
    float2 m_alpha;
    float m_thin_film_thickness;
    float m_thin_film_ior;
};

// Walter, Bruce, et al. "Microfacet Models for Refraction through Rough
// Surfaces." Rendering techniques 2007 (2007): 18th.
class MicrofacetTransmission
{
   public:
    CUDA_INLINE CUDA_DEVICE MicrofacetTransmission() {}
    CUDA_INLINE CUDA_DEVICE MicrofacetTransmission(
        float ior_i, float ior_t, float roughness, float anisotropy,
        float thin_film_thickness = 0.0f, float thin_film_ior = 1.5f)
        : m_ior_i(ior_i),
          m_ior_t(ior_t),
          m_thin_film_thickness(thin_film_thickness),
          m_thin_film_ior(thin_film_ior)
    {
        m_alpha = roughness_to_alpha(roughness, anisotropy);
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float3 wh = compute_half_vector(wo, wi);
        float3 f;
        if (m_thin_film_thickness > 0.0f)
        {
            f = fresnel_airy(fabs(dot(wo, wh)), m_ior_i, m_thin_film_ior,
                             m_thin_film_thickness, make_float3(m_ior_t),
                             make_float3(0.0f));
        }
        else
        {
            f = make_float3(
                fresnel_dielectric(fabs(dot(wo, wh)), m_ior_t / m_ior_i));
        }
        const float d = D(wh);
        const float g = G2(wo, wi);
        const float wo_dot_wh = dot(wo, wh);
        const float wi_dot_wh = dot(wi, wh);
        const float t = m_ior_i * wo_dot_wh + m_ior_t * wi_dot_wh;
        return fabs(wo_dot_wh) * fabs(wi_dot_wh) * m_ior_t * m_ior_t *
               fmaxf(1.0f - f, make_float3(0.0f)) * g * d /
               (abs_cos_theta(wo) * abs_cos_theta(wi) * t * t);
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
    {
        // sample half-vector
        const float3 wh = sample_vndf(wo, m_alpha, u);

        // compute incident direction
        float3 wi;
        if (!refract(wo, wh, m_ior_i, m_ior_t, wi))
        {
            // total internal reflection
            wi = reflect(wo, wh);

            float3 fr;
            if (m_thin_film_thickness > 0.0f)
            {
                fr = fresnel_airy(fabs(dot(wo, wh)), m_ior_i, m_thin_film_ior,
                                  m_thin_film_thickness, make_float3(m_ior_t),
                                  make_float3(0.0f));
            }
            else
            {
                fr = make_float3(
                    fresnel_dielectric(fabs(dot(wo, wh)), m_ior_t / m_ior_i));
            }

            const float d = D(wh);
            const float g = G2(wo, wi);
            f = 0.25f * (fr * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
            pdf = 0.25f * D_visible(wo, wh) / fabs(dot(wi, wh));
            return wi;
        }

        // evaluate BxDF and pdf
        f = eval(wo, wi);
        pdf = eval_pdf(wo, wi);

        return wi;
    }

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        const float3 wh = compute_half_vector(wo, wi);
        const float wi_dot_wh = dot(wi, wh);
        const float t = m_ior_i * dot(wo, wh) + m_ior_t * wi_dot_wh;
        return D_visible(wo, wh) * m_ior_t * m_ior_t * fabs(wi_dot_wh) /
               (t * t);
    }

   private:
    CUDA_INLINE CUDA_DEVICE float3 compute_half_vector(const float3& wo,
                                                       const float3& wi) const
    {
        float3 wh = normalize(-(m_ior_i * wo + m_ior_t * wi));
        if (wh.y < 0.0f) wh = -wh;
        return wh;
    }

    CUDA_INLINE CUDA_DEVICE float D(const float3& wh) const
    {
        const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                        wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
        return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
    }

    CUDA_INLINE CUDA_DEVICE float D_visible(const float3& w,
                                            const float3& wh) const
    {
        return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
    }

    CUDA_INLINE CUDA_DEVICE float lambda(const float3& w) const
    {
        const float t = (m_alpha.x * m_alpha.x * w.x * w.x +
                         m_alpha.y * m_alpha.y * w.z * w.z) /
                        (w.y * w.y);
        return 0.5f * (-1.0f + sqrtf(1.0f + t));
    }

    CUDA_INLINE CUDA_DEVICE float G1(const float3& w) const
    {
        return 1.0f / (1.0f + lambda(w));
    }

    CUDA_INLINE CUDA_DEVICE float G2(const float3& wo, const float3& wi) const
    {
        return 1.0f / (1.0f + lambda(wo) + lambda(wi));
    }

    float m_ior_i;
    float m_ior_t;
    float2 m_alpha;
    float m_thin_film_thickness;
    float m_thin_film_ior;
};

// Production Friendly Microfacet Sheen BRDF
class MicrofacetSheen
{
   public:
    CUDA_INLINE CUDA_DEVICE MicrofacetSheen() {}
    CUDA_INLINE CUDA_DEVICE MicrofacetSheen(float roughness)
        : m_roughness(roughness)
    {
    }

    CUDA_INLINE CUDA_DEVICE float3 eval(const float3& wo,
                                        const float3& wi) const
    {
        const float3 wh = normalize(wo + wi);
        const float f = 1.0f;
        const float d = D(wh);
        const float g = G2(wo, wi);
        return make_float3(0.25f * (f * d * g) /
                           (abs_cos_theta(wo) * abs_cos_theta(wi)));
    }

    CUDA_INLINE CUDA_DEVICE float3 sample(const float3& wo, const float2& u,
                                          float3& f, float& pdf) const
    {
        // sample half-vector
        const float3 wh = sample_cosine_weighted_hemisphere(u);

        // compute incident direction
        const float3 wi = reflect(wo, wh);

        // evaluate BxDF and pdf
        f = eval(wo, wi);
        pdf = eval_pdf(wo, wi);

        return wi;
    }

    CUDA_INLINE CUDA_DEVICE float eval_pdf(const float3& wo,
                                           const float3& wi) const
    {
        return abs_cos_theta(wi) / M_PIf;
    }

   private:
    CUDA_INLINE CUDA_DEVICE float D(const float3& wh) const
    {
        const float s = abs_sin_theta(wh);
        return (2.0f + 1.0f / m_roughness) * powf(s, 1.0f / m_roughness) /
               (2.0f * M_PIf);
    }

    CUDA_INLINE CUDA_DEVICE float lambda(const float3& w) const
    {
        const float cos = abs_cos_theta(w);
        return (cos < 0.5f) ? expf(L(cos))
                            : expf(2.0f * L(0.5f) - L(1.0f - cos));
    }

    CUDA_INLINE CUDA_DEVICE float G1(const float3& w) const
    {
        return 1.0f / (1.0f + lambda(w));
    }

    CUDA_INLINE CUDA_DEVICE float G2(const float3& wo, const float3& wi) const
    {
        return 1.0f / (1.0f + lambda(wo) + lambda(wi));
    }

    CUDA_INLINE CUDA_DEVICE float L(float x) const
    {
        const auto interpolate = [](float roughness, float p0, float p1)
        {
            const float t = (1.0f - roughness);
            const float t2 = t * t;
            return t2 * p0 + (1.0f - t2) * p1;
        };

        const float a = interpolate(m_roughness, 25.3245, 21.5473);
        const float b = interpolate(m_roughness, 3.32435, 3.82987);
        const float c = interpolate(m_roughness, 0.16801, 0.19823);
        const float d = interpolate(m_roughness, -1.27393, -1.97760);
        const float e = interpolate(m_roughness, -4.85967, -4.32054);

        return a / (1.0f + b * powf(x, c)) + d * x + e;
    }

    float m_roughness;
};