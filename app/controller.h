#pragma once
#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

#include "camera.h"
#include "cuda_gl_util.h"
#include "renderer.h"
#include "scene.h"

inline float deg2rad(float deg) { return deg / 180.0f * M_PI; }

class Controller
{
 public:
  int m_imgui_resolution[2] = {512, 512};

  float m_imgui_origin[3] = {0, 1, 5};
  float m_imgui_forward[3] = {0, 0, -1};
  float m_imgui_fov = 90.0f;

  Controller()
  {
    m_camera = std::make_unique<fredholm::Camera>();
    m_scene = std::make_unique<fredholm::Scene>();
    m_renderer = std::make_unique<fredholm::Renderer>();
  }

  void init_camera()
  {
    const float3 origin =
        make_float3(m_imgui_origin[0], m_imgui_origin[1], m_imgui_origin[2]);
    const float3 forward =
        make_float3(m_imgui_forward[0], m_imgui_forward[1], m_imgui_forward[2]);
    m_camera = std::make_unique<fredholm::Camera>(origin, forward,
                                                  deg2rad(m_imgui_fov));
  }

  void move_camera(const fredholm::CameraMovement& direction, float dt)
  {
    m_camera->move(direction, dt);
    m_imgui_origin[0] = m_camera->m_origin.x;
    m_imgui_origin[1] = m_camera->m_origin.y;
    m_imgui_origin[2] = m_camera->m_origin.z;

    init_render_states();
  }

  void rotate_camera(float dphi, float dtheta)
  {
    m_camera->lookAround(dphi, dtheta);
    m_imgui_forward[0] = m_camera->m_forward.x;
    m_imgui_forward[1] = m_camera->m_forward.y;
    m_imgui_forward[2] = m_camera->m_forward.z;

    init_render_states();
  }

  void init_renderer()
  {
    m_renderer->create_context();
    m_renderer->create_module(std::filesystem::path(MODULES_SOURCE_DIR) /
                              "pt.ptx");
    m_renderer->create_program_group();
    m_renderer->create_pipeline();
    m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);
  }

  void init_framebuffer()
  {
    m_framebuffer = std::make_unique<app::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
  }

  // TODO: clear scene before loading
  void load_scene(const std::filesystem::path& filepath)
  {
    m_scene->load_obj(filepath);
    m_renderer->load_scene(*m_scene);
    m_renderer->build_accel();
    m_renderer->create_sbt();
  }

  void update_resolution()
  {
    init_framebuffer();
    m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);
  }

  void init_render_states() { m_renderer->init_render_states(); }

  void render(uint32_t n_samples, uint32_t max_depth)
  {
    m_renderer->render(*m_camera, m_framebuffer->get_device_ptr(), n_samples,
                       max_depth);
    // TODO: Is is safe to remove this?
    m_renderer->wait_for_completion();
  }

  const gcss::Buffer<float4>& get_gl_framebuffer() const
  {
    return m_framebuffer->get_gl_buffer();
  }

 private:
  std::unique_ptr<fredholm::Camera> m_camera = nullptr;
  std::unique_ptr<fredholm::Scene> m_scene = nullptr;

  std::unique_ptr<app::CUDAGLBuffer<float4>> m_framebuffer = nullptr;
  std::unique_ptr<fredholm::Renderer> m_renderer = nullptr;
};