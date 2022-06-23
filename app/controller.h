#pragma once
#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

#include "camera.h"
#include "cuda_gl_util.h"
#include "denoiser.h"
#include "renderer.h"
#include "scene.h"
#include "shared.h"

inline float deg2rad(float deg) { return deg / 180.0f * M_PI; }

enum class AOVType : int { BEAUTY, DENOISED, POSITION, NORMAL, DEPTH, ALBEDO };

class Controller
{
 public:
  int m_imgui_resolution[2] = {512, 512};
  int m_imgui_n_samples = 0;
  int m_imgui_max_samples = 100;
  int m_imgui_max_depth = 100;
  AOVType m_imgui_aov_type = AOVType::BEAUTY;

  float m_imgui_origin[3] = {0, 1, 5};
  float m_imgui_forward[3] = {0, 0, -1};
  float m_imgui_fov = 90.0f;
  float m_imgui_movement_speed = 1.0f;
  float m_imgui_rotation_speed = 0.1f;

  std::unique_ptr<fredholm::Camera> m_camera = nullptr;
  std::unique_ptr<fredholm::Scene> m_scene = nullptr;

  std::unique_ptr<app::CUDAGLBuffer<float4>> m_layer_beauty = nullptr;
  std::unique_ptr<app::CUDAGLBuffer<float4>> m_layer_position = nullptr;
  std::unique_ptr<app::CUDAGLBuffer<float4>> m_layer_normal = nullptr;
  std::unique_ptr<app::CUDAGLBuffer<float>> m_layer_depth = nullptr;
  std::unique_ptr<app::CUDAGLBuffer<float4>> m_layer_albedo = nullptr;
  std::unique_ptr<app::CUDAGLBuffer<float4>> m_layer_denoised = nullptr;

  std::unique_ptr<fredholm::Renderer> m_renderer = nullptr;
  std::unique_ptr<fredholm::Denoiser> m_denoiser = nullptr;

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

  void update_camera()
  {
    m_camera->set_fov(deg2rad(m_imgui_fov));
    m_camera->m_movement_speed = m_imgui_movement_speed;
    m_camera->m_look_around_speed = m_imgui_rotation_speed;
  }

  void move_camera(const fredholm::CameraMovement& direction, float dt)
  {
    m_camera->move(direction, dt);
    m_imgui_origin[0] = m_camera->m_origin.x;
    m_imgui_origin[1] = m_camera->m_origin.y;
    m_imgui_origin[2] = m_camera->m_origin.z;
  }

  void rotate_camera(float dphi, float dtheta)
  {
    m_camera->lookAround(dphi, dtheta);
    m_imgui_forward[0] = m_camera->m_forward.x;
    m_imgui_forward[1] = m_camera->m_forward.y;
    m_imgui_forward[2] = m_camera->m_forward.z;
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

  void init_denoiser()
  {
    m_layer_denoised = std::make_unique<app::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
    m_denoiser = std::make_unique<fredholm::Denoiser>(
        m_renderer->get_context(), m_imgui_resolution[0], m_imgui_resolution[1],
        m_layer_beauty->get_device_ptr(), m_layer_normal->get_device_ptr(),
        m_layer_albedo->get_device_ptr(), m_layer_denoised->get_device_ptr());
  }

  void init_render_layers()
  {
    m_layer_beauty = std::make_unique<app::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
    m_layer_position = std::make_unique<app::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
    m_layer_normal = std::make_unique<app::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
    m_layer_depth = std::make_unique<app::CUDAGLBuffer<float>>(
        m_imgui_resolution[0], m_imgui_resolution[1]);
    m_layer_albedo = std::make_unique<app::CUDAGLBuffer<float4>>(
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
    init_render_layers();
    m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);

    init_denoiser();
  }

  void init_render_states()
  {
    m_imgui_n_samples = 0;
    m_renderer->init_render_states();
  }

  void render()
  {
    if (m_imgui_n_samples < m_imgui_max_samples) {
      fredholm::RenderLayer render_layers;
      render_layers.beauty = m_layer_beauty->get_device_ptr();
      render_layers.position = m_layer_position->get_device_ptr();
      render_layers.normal = m_layer_normal->get_device_ptr();
      render_layers.depth = m_layer_depth->get_device_ptr();
      render_layers.albedo = m_layer_albedo->get_device_ptr();

      m_renderer->render(*m_camera, render_layers, 1, m_imgui_max_depth);
      // TODO: Is is safe to remove this?
      m_renderer->wait_for_completion();

      m_imgui_n_samples++;
    }
  }

  void denoise()
  {
    m_denoiser->denoise();
    m_denoiser->wait_for_completion();
  }
};