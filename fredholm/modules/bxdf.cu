#pragma once

#include <cmath>

#include "math.cu"
#include "sampling.cu"
#include "sutil/vec_math.h"

__forceinline__ __device__ float cos_theta(const float3& w) { return w.y; }

__forceinline__ __device__ float cos2_theta(const float3& w)
{
  return w.y * w.y;
}

__forceinline__ __device__ float abs_cos_theta(const float3& w)
{
  return fabs(w.y);
}

__forceinline__ __device__ float sin_theta(const float3& w)
{
  return sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float sin2_theta(const float3& w)
{
  return fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_sin_theta(const float3& w)
{
  return fabs(sin_theta(w));
}

__forceinline__ __device__ float tan_theta(const float3& w)
{
  return sin_theta(w) / cos_theta(w);
}

__forceinline__ __device__ float tan2_theta(const float3& w)
{
  return 1.0f / (w.y * w.y) - 1.0f;
}

__forceinline__ __device__ float abs_tan_theta(const float3& w)
{
  return fabs(tan_theta(w));
}

__forceinline__ __device__ float sin_phi(const float3& w)
{
  return w.z / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float sin2_phi(const float3& w)
{
  return w.z * w.z / fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_sin_phi(const float3& w)
{
  return fabs(sin_phi(w));
}

__forceinline__ __device__ float cos_phi(const float3& w)
{
  return w.x / sqrtf(fmax(1.0f - w.y * w.y, 0.0f));
}

__forceinline__ __device__ float cos2_phi(const float3& w)
{
  return w.x * w.x / fmax(1.0f - w.y * w.y, 0.0f);
}

__forceinline__ __device__ float abs_cos_phi(const float3& w)
{
  return fabs(cos_phi(w));
}

__forceinline__ __device__ float3 reflect(const float3& w, const float3& n)
{
  return normalize(-w + 2.0f * dot(w, n) * n);
}

__forceinline__ __device__ bool refract(const float3& w, const float3& n,
                                        float ior_i, float ior_t, float3& wt)
{
  const float3 th = -ior_i / ior_t * (w - dot(w, n) * n);
  if (dot(th, th) > 1.0f) return false;
  const float3 tp = -sqrtf(fmax(1.0f - dot(th, th), 0.0f)) * n;
  wt = th + tp;
  return true;
}

__forceinline__ __device__ float2 roughness_to_alpha(float roughness,
                                                     float anisotropy)
{
  // Revisiting Physically Based Shading at Imageworks p.24
  float2 alpha;
  alpha.x = roughness * roughness * (1.0f + anisotropy);
  alpha.y = roughness * roughness * (1.0f - anisotropy);
  return alpha;
}

// https://jcgt.org/published/0003/04/03/
__forceinline__ __device__ float3 artist_friendly_metallic_fresnel(
    const float3& reflectivity, const float3& edge_tint, float3& n, float3& k)
{
  const float3 r_sqrt = sqrt(reflectivity);
  n = edge_tint * (1.0f - reflectivity) / (1.0f + reflectivity) +
      (1.0f - edge_tint) * (1.0f + r_sqrt) / (1.0f - r_sqrt);
  const float3 t1 = n + 1.0f;
  const float3 t2 = n - 1.0f;
  k = sqrt((reflectivity * (t1 * t1) - t2 * t2) / (1.0f - reflectivity));
}

class Lambert
{
 public:
  __device__ Lambert() {}
  __device__ Lambert(const float3& albedo) : m_albedo(albedo) {}

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    return m_albedo / M_PIf;
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    const float3 wi = sample_cosine_weighted_hemisphere(u);

    f = eval(wo, wi);
    pdf = abs_cos_theta(wi) / M_PIf;

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return abs_cos_theta(wi) / M_PIf;
  }

 private:
  float3 m_albedo;
};

class OrenNayer
{
 public:
  __device__ OrenNayer() {}
  __device__ OrenNayer(const float3& albedo, float roughness)
      : m_albedo(albedo), m_roughness(roughness)
  {
    const float sigma2 = roughness * roughness;
    m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
    m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float s_theta_o = sin_theta(wo);
    const float s_theta_i = sin_theta(wi);

    float c_max = 0.0f;
    if (s_theta_i > 1e-4f && s_theta_o > 1e-4f) {
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

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    const float3 wi = sample_cosine_weighted_hemisphere(u);

    f = eval(wo, wi);
    pdf = abs_cos_theta(wi) / M_PIf;

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
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
  __device__ DiffuseTransmission() {}
  __device__ DiffuseTransmission(const float3& albedo, float roughness)
      : m_albedo(albedo), m_roughness(roughness)
  {
    const float sigma2 = roughness * roughness;
    m_A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
    m_B = 0.45f * sigma2 / (sigma2 + 0.09f);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float s_theta_o = sin_theta(wo);
    const float s_theta_i = sin_theta(wi);

    float c_max = 0.0f;
    if (s_theta_i > 1e-4f && s_theta_o > 1e-4f) {
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

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    float3 wi = sample_cosine_weighted_hemisphere(u);
    // flip direction
    wi = -wi;

    f = eval(wo, wi);
    pdf = abs_cos_theta(wi) / M_PIf;

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return abs_cos_theta(wi) / M_PIf;
  }

 private:
  float3 m_albedo;
  float m_roughness;
  float m_A;
  float m_B;
};

struct FresnelSchlick {
  __device__ FresnelSchlick() {}
  __device__ FresnelSchlick(float n)
  {
    const float t = (1.0f - n) / (1.0f + n);
    m_F0 = t * t;
  }

  __device__ float eval(float cos) const
  {
    const float t = fmax(1.0f - cos, 0.0f);
    return m_F0 + fmax(1.0f - m_F0, 0.0f) * t * t * t * t * t;
  }

  float m_F0;
};

struct FresnelDielectric {
  __device__ FresnelDielectric() {}
  __device__ FresnelDielectric(float n) : m_n(n) {}

  __device__ float eval(float cos) const
  {
    const float temp = m_n * m_n + cos * cos - 1.0f;
    if (temp < 0.0f) { return 1.0f; }

    const float g = sqrtf(temp);
    const float t0 = (g - cos) / (g + cos);
    const float t1 = ((g + cos) * cos - 1.0f) / ((g - cos) * cos + 1.0f);
    return 0.5f * t0 * t0 * (1.0f + t1 * t1);
  }

  float m_n;
};

struct FresnelConductor {
  __device__ FresnelConductor() {}
  __device__ FresnelConductor(const float3& n, const float3& k) : m_n(n), m_k(k)
  {
  }

  __device__ float3 eval(float cos) const
  {
    const float c2 = cos * cos;
    const float3 two_eta_cos = 2.0f * m_n * cos;

    const float3 t0 = m_n * m_n + m_k * m_k;
    const float3 t1 = t0 * c2;
    const float3 Rs = (t0 - two_eta_cos + c2) / (t0 + two_eta_cos + c2);
    const float3 Rp = (t1 - two_eta_cos + 1.0f) / (t1 + two_eta_cos + 1.0f);

    return 0.5f * (Rp + Rs);
  }

  float3 m_n;
  float3 m_k;
};

class MicrofacetReflectionDielectric
{
 public:
  __device__ MicrofacetReflectionDielectric() {}
  __device__ MicrofacetReflectionDielectric(float ior, float roughness,
                                            float anisotropy)
      : m_fresnel(ior)
  {
    m_alpha = roughness_to_alpha(roughness, anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float f = m_fresnel.eval(fabs(dot(wo, wh)));
    const float d = D(wh);
    const float g = G2(wo, wi);
    return make_float3(0.25f * (f * d * g) /
                       (abs_cos_theta(wo) * abs_cos_theta(wi)));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
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

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    return 0.25f * D_visible(wo, wh) / fabs(dot(wo, wh));
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                    wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a2 = (cos2_phi(w) * m_alpha.x * m_alpha.x +
                      sin2_phi(w) * m_alpha.y * m_alpha.y);
    const float t = 1.0f / (a2 * tan2_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / t));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  FresnelDielectric m_fresnel;
  float2 m_alpha;
};

class MicrofacetReflectionConductor
{
 public:
  __device__ MicrofacetReflectionConductor() {}
  __device__ MicrofacetReflectionConductor(const float3& n, const float3& k,
                                           float roughness, float anisotropy)
      : m_fresnel(n, k)
  {
    m_alpha = roughness_to_alpha(roughness, anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float3 f = m_fresnel.eval(fabs(dot(wo, wh)));
    const float d = D(wh);
    const float g = G2(wo, wi);
    return 0.25f * (f * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
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

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    return 0.25f * D_visible(wo, wh) / fabs(dot(wo, wh));
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                    wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a2 = (cos2_phi(w) * m_alpha.x * m_alpha.x +
                      sin2_phi(w) * m_alpha.y * m_alpha.y);
    const float t = 1.0f / (a2 * tan2_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / t));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  FresnelConductor m_fresnel;
  float2 m_alpha;
};

// Walter, Bruce, et al. "Microfacet Models for Refraction through Rough
// Surfaces." Rendering techniques 2007 (2007): 18th.
class MicrofacetTransmission
{
 public:
  __device__ MicrofacetTransmission() {}
  __device__ MicrofacetTransmission(float ior_i, float ior_t, float roughness,
                                    float anisotropy)
      : m_ior_i(ior_i), m_ior_t(ior_t), m_fresnel(ior_t / ior_i)
  {
    m_alpha = roughness_to_alpha(roughness, anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = compute_half_vector(wo, wi);
    const float f = m_fresnel.eval(fabs(dot(wo, wh)));
    const float d = D(wh);
    const float g = G2(wo, wi);
    const float wo_dot_wh = dot(wo, wh);
    const float wi_dot_wh = dot(wi, wh);
    const float t = m_ior_i * wo_dot_wh + m_ior_t * wi_dot_wh;
    return make_float3(fabs(wo_dot_wh) * fabs(wi_dot_wh) * m_ior_t * m_ior_t *
                       fmax(1.0f - f, 0.0f) * g * d /
                       (abs_cos_theta(wo) * abs_cos_theta(wi) * t * t));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    // sample half-vector
    const float3 wh = sample_vndf(wo, m_alpha, u);

    // compute incident direction
    float3 wi;
    if (!refract(wo, wh, m_ior_i, m_ior_t, wi)) {
      // total internal reflection
      f = make_float3(0.0f);
      pdf = 1;
      return wi;
    }

    // evaluate BxDF and pdf
    f = eval(wo, wi);
    pdf = eval_pdf(wo, wi);

    return wi;
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    const float3 wh = compute_half_vector(wo, wi);
    const float wi_dot_wh = dot(wi, wh);
    const float t = m_ior_i * dot(wo, wh) + m_ior_t * wi_dot_wh;
    return D_visible(wo, wh) * m_ior_t * m_ior_t * fabs(wi_dot_wh) / (t * t);
  }

 private:
  __device__ float3 compute_half_vector(const float3& wo,
                                        const float3& wi) const
  {
    float3 wh = normalize(-(m_ior_i * wo + m_ior_t * wi));
    if (wh.y < 0.0f) wh = -wh;
    return wh;
  }

  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (m_alpha.x * m_alpha.x) +
                    wh.z * wh.z / (m_alpha.y * m_alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * m_alpha.x * m_alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a2 = (cos2_phi(w) * m_alpha.x * m_alpha.x +
                      sin2_phi(w) * m_alpha.y * m_alpha.y);
    const float t = 1.0f / (a2 * tan2_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / t));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  float m_ior_i;
  float m_ior_t;
  FresnelDielectric m_fresnel;
  float2 m_alpha;
};

// Production Friendly Microfacet Sheen BRDF
class MicrofacetSheen
{
 public:
  __device__ MicrofacetSheen() {}
  __device__ MicrofacetSheen(float roughness, float eta)
      : m_fresnel(eta), m_roughness(roughness)
  {
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float f = m_fresnel.eval(fabs(dot(wo, wh)));
    const float d = D(wh);
    const float g = G2(wo, wi);
    return make_float3(0.25f * (f * d * g) /
                       (abs_cos_theta(wo) * abs_cos_theta(wi)));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
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

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return abs_cos_theta(wi) / M_PIf;
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float s = abs_sin_theta(wh);
    return (2.0f + 1.0f / m_roughness) * powf(s, 1.0f / m_roughness) /
           (2.0f * M_PIf);
  }

  __device__ float lambda(const float3& w) const
  {
    const float cos = abs_cos_theta(w);
    if (cos < 0.5f) {
      return expf(L(cos));
    } else {
      return expf(2.0f * L(0.5f) - L(1.0f - cos));
    }
  }
  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  __device__ float L(float x) const
  {
    const auto interpolate = [](float roughness, float p0, float p1) {
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
  FresnelDielectric m_fresnel;
};