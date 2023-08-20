#pragma once

#include <cmath>
#include <iostream>

#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"

namespace fredholm
{

enum class CameraMovement
{
    FORWARD,
    BACKWARD,
    RIGHT,
    LEFT,
    UP,
    DOWN,
};

class Camera
{
   private:
    glm::mat4 m_transform = glm::identity<glm::mat4>();

    float m_fov = 0.5f * M_PI;
    float m_F = 100.0f;       // F number
    float m_focus = FLT_MAX;  // focus distance

    float m_movement_speed = 1.0f;
    float m_look_around_speed = 0.1f;

    glm::vec3 m_origin = glm::vec3(0.0f);
    glm::vec3 m_forward = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 m_right = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
    float m_phi = 270.0f;
    float m_theta = 90.0f;

   public:
    Camera() {}

    Camera(const glm::vec3& origin, float fov = 0.5f * M_PI, float F = 8.0f,
           float focus = 10000.0f, float movement_speed = 1.0f,
           float look_around_speed = 0.1f)
        : m_origin(origin),
          m_fov(fov),
          m_F(F),
          m_focus(focus),
          m_movement_speed(movement_speed),
          m_look_around_speed(look_around_speed),
          m_phi(270.0f),
          m_theta(90.0f)
    {
        m_forward = glm::vec3(0, 0, -1);
        m_right = glm::normalize(glm::cross(m_forward, glm::vec3(0, 1, 0)));
        m_up = glm::normalize(glm::cross(m_right, m_forward));
    }

    glm::vec3 get_origin() const { return m_origin; }

    glm::vec3 get_forward() const { return m_forward; }

    glm::mat4 get_transform() const
    {
        return glm::inverse(
            glm::lookAt(m_origin, m_origin + 0.01f * m_forward, m_up));
    }

    float get_fov() const { return m_fov; }

    float get_F() const { return m_F; }

    float get_focus() const { return m_focus; }

    void set_fov(float fov) { m_fov = fov; }

    void set_F(float F) { m_F = F; }

    void set_focus(float focus) { m_focus = focus; }

    void move(const CameraMovement& direction, float dt)
    {
        const float velocity = m_movement_speed * dt;

        switch (direction)
        {
            case CameraMovement::FORWARD:
            {
                m_origin += velocity * m_forward;
            }
            break;
            case CameraMovement::BACKWARD:
            {
                m_origin -= velocity * m_forward;
            }
            break;
            case CameraMovement::RIGHT:
            {
                m_origin += velocity * m_right;
            }
            break;
            case CameraMovement::LEFT:
            {
                m_origin -= velocity * m_right;
            }
            break;
            case CameraMovement::UP:
            {
                m_origin += velocity * m_up;
            }
            break;
            case CameraMovement::DOWN:
            {
                m_origin -= velocity * m_up;
            }
            break;
        }
    }

    void look_around(float d_phi, float d_theta)
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
        m_forward = glm::vec3(std::cos(phi_rad) * std::sin(theta_rad),
                              std::cos(theta_rad),
                              std::sin(phi_rad) * std::sin(theta_rad));
        m_right =
            glm::normalize(glm::cross(m_forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        m_up = glm::normalize(glm::cross(m_right, m_forward));
    }
};

}  // namespace fredholm