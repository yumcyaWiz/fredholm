#pragma once
#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

#include "camera.h"
#include "cuda_gl_util.h"
#include "renderer.h"
#include "scene.h"

struct CameraSettings {
  float3 origin;
  float3 forward;
  float fov;
};

class Controller
{
 public:
  int m_imgui_resolution[2] = {512, 512};

  Controller()
  {
    m_camera = std::make_unique<fredholm::Camera>();
    m_scene = std::make_unique<fredholm::Scene>();
    m_renderer = std::make_unique<fredholm::Renderer>();
  }

  uint32_t get_width() const { return m_imgui_resolution[0]; }
  uint32_t get_height() const { return m_imgui_resolution[1]; }

  void init_camera(const CameraSettings& params)
  {
    m_camera = std::make_unique<fredholm::Camera>(params.origin, params.forward,
                                                  params.fov);
  }

  void move_camera(const fredholm::CameraMovement& direction, float dt)
  {
    m_camera->move(direction, dt);
    init_render_states();
  }

  void rotate_camera(float dphi, float dtheta)
  {
    m_camera->lookAround(dphi, dtheta);
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