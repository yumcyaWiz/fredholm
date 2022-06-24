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

  Camera()
      : m_origin(make_float3(0, 0, 0)),
        m_movement_speed(10.0f),
        m_look_around_speed(0.1f),
        m_phi(270.0f),
        m_theta(90.0f)
  {
    set_forward(make_float3(0, 0, -1));
    set_fov(0.5f * M_PI);
  }

  Camera(const float3& origin, const float3& forward, float fov = 0.5f * M_PI)
      : m_origin(origin),
        m_movement_speed(1.0f),
        m_look_around_speed(0.1f),
        m_phi(270.0f),
        m_theta(90.0f)
  {
    set_forward(forward);
    set_fov(fov);
  }

  void set_forward(const float3& forward)
  {
    m_forward = forward;
    m_right = normalize(cross(m_forward, make_float3(0.0f, 1.0f, 0.0f)));
    m_up = normalize(cross(m_right, m_forward));
  }

  void set_fov(float fov) { m_f = 1.0f / std::tan(0.5f * fov); }

  void move(const CameraMovement& direction, float dt)
  {
    const float velocity = m_movement_speed * dt;
    switch (direction) {
      case CameraMovement::FORWARD:
        m_origin += velocity * m_forward;
        break;
      case CameraMovement::BACKWARD:
        m_origin -= velocity * m_forward;
        break;
      case CameraMovement::RIGHT:
        m_origin += velocity * m_right;
        break;
      case CameraMovement::LEFT:
        m_origin -= velocity * m_right;
        break;
      case CameraMovement::UP:
        m_origin += velocity * m_up;
        break;
      case CameraMovement::DOWN:
        m_origin -= velocity * m_up;
        break;
    }
  }

  void lookAround(float d_phi, float d_theta)
  {
    // update phi, theta
    m_phi += m_look_around_speed * d_phi;
    if (m_phi < 0.0f) m_phi = 360.0f;
    if (m_phi > 360.0f) m_phi = 0.0f;

    m_theta += m_look_around_speed * d_theta;
    if (m_theta < 0.0f) m_theta = 180.0f;
    if (m_theta > 180.0f) m_theta = 0.0f;

    // set camera directions
    const float phi_rad = m_phi / 180.0f * M_PI;
    const float theta_rad = m_theta / 180.0f * M_PI;
    m_forward = make_float3(std::cos(phi_rad) * std::sin(theta_rad),
                            std::cos(theta_rad),
                            std::sin(phi_rad) * std::sin(theta_rad));
    m_right = normalize(cross(m_forward, make_float3(0.0f, 1.0f, 0.0f)));
    m_up = normalize(cross(m_right, m_forward));
  }
};

}  // namespace fredholm