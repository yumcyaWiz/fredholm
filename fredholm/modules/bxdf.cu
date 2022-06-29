#include "sutil/vec_math.h"

__forceinline__ __device__ float cos_theta(const float3& w) { return w.y; }

__forceinline__ __device__ float cos2_theta(const float3& w)
{
  return w.y * w.y;
}

__forceinline__ __device__ float abs_cos_theta(const float3& w)
{
  return fabs(w.y * w.y);
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
  return -w + 2.0f * dot(w, n) * n;
}

class MicrofacetReflection
{
 public:
  __device__ MicrofacetReflection(float roughness, float anisotropy)
  {
    // Revisiting Physically Based Shading at Imageworks p.24
    alpha.x = roughness * roughness * (1.0f + anisotropy);
    alpha.y = roughness * roughness * (1.0f - anisotropy);
  }

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    const float3 wh = normalize(wo + wi);
    const float3 f = make_float3(1.0f);
    const float d = D(wh);
    const float g = G2(wo, wi);
    return 0.25f * (f * d * g) / (abs_cos_theta(wo) * abs_cos_theta(wi));
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    // sample half-vector
    const float3 wh = sample_vndf(wo, u);

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
    return 0.25f * D_visible(wo, wh) / abs_cos_theta(wo);
  }

 private:
  __device__ float D(const float3& wh) const
  {
    const float t = wh.x * wh.x / (alpha.x * alpha.x) +
                    wh.z * wh.z / (alpha.y * alpha.y) + wh.y * wh.y;
    return 1.0f / (M_PI * alpha.x * alpha.y * t * t);
  }

  __device__ float D_visible(const float3& w, const float3& wh) const
  {
    return G1(w) * fabs(dot(w, wh)) * D(wh) / abs_cos_theta(w);
  }

  __device__ float lambda(const float3& w) const
  {
    const float a0 = sqrtf(cos2_phi(w) * alpha.x * alpha.x +
                           sin2_phi(w) * alpha.y * alpha.y);
    const float a = 1.0f / (a0 * tan_theta(w));
    return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / (a * a)));
  }

  __device__ float G1(const float3& w) const
  {
    return 1.0f / (1.0f + lambda(w));
  }

  __device__ float G2(const float3& wo, const float3& wi) const
  {
    return 1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  // https://jcgt.org/published/0007/04/01/
  __device__ float3 sample_vndf(const float3& wo, const float2& u) const
  {
    const float3 Vh =
        normalize(make_float3(alpha.x * wo.x, wo.y, alpha.y * wo.z));

    const float lensq = Vh.x * Vh.x + Vh.z * Vh.z;
    const float3 T1 = lensq > 0 ? make_float3(Vh.z, 0, -Vh.x) / sqrtf(lensq)
                                : make_float3(0, 0, 1);
    const float3 T2 = cross(Vh, T1);

    const float r = sqrtf(u.x);
    const float phi = 2.0f * M_PI * u.y;
    const float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + Vh.y);
    t2 = (1.0f - s) * sqrtf(fmax(1.0f - t1 * t1, 0.0f)) + s * t2;
    const float3 Nh =
        t1 * T1 + t2 * T2 + sqrtf(fmax(1.0f - t1 * t1 - t2 * t2, 0.0f)) * Vh;
    const float3 Ne = normalize(
        make_float3(alpha.x * Nh.x, fmax(0.0f, Nh.y), alpha.y * Nh.z));

    return Ne;
  }

  float2 alpha;
};