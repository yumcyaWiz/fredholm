#pragma once

#include "sutil/vec_math.h"

namespace fredholm
{

struct Camera {
  float3 m_origin;
  float3 m_forward;
  float3 m_right;
  float3 m_up;

  Camera(const float3& origin, const float3& forward)
      : m_origin(origin), m_forward(forward)
  {
    m_right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
    m_up = normalize(cross(m_right, forward));
  }
};

}  // namespace fredholm