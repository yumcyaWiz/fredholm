#pragma once

#include <cmath>

#include "sutil/vec_math.h"

namespace fredholm
{

enum class CameraMovement {
  FORWARD,
  BACKWARD,
  RIGHT,
  LEFT,
  UP,
  DOWN,
};

struct Camera {
  float3 m_origin;
  float3 m_forward;
  float3 m_right;
  float3 m_up;

  float m_f;

  float m_movement_speed;
  float m_look_around_speed;

  float m_phi;
  float m_theta;

  Camera(const float3& origin, const float3& forward, float fov = 0.5f * M_PI)
      : m_origin(origin), m_forward(forward)
  {
    m_right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
    m_up = normalize(cross(m_right, forward));

    m_f = 1.0f / std::tan(0.5f * fov);
  }
};

}  // namespace fredholm