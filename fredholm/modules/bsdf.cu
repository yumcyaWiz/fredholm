#pragma once

#include "bxdf.cu"

class BSDF
{
 public:
  __device__ BSDF(const float3& base_color) : m_lambert(base_color) {}

  __device__ float3 eval(const float3& wo, const float3& wi) const
  {
    return m_lambert.eval(wo, wi);
  }

  __device__ float3 sample(const float3& wo, const float2& u, float3& f,
                           float& pdf) const
  {
    return m_lambert.sample(wo, u, f, pdf);
  }

  __device__ float eval_pdf(const float3& wo, const float3& wi) const
  {
    return m_lambert.eval_pdf(wo, wi);
  }

 private:
  Lambert m_lambert;
};