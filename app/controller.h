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

struct Controller {
  std::unique_ptr<fredholm::Camera> camera = nullptr;
  std::unique_ptr<fredholm::Scene> scene = nullptr;

  std::unique_ptr<app::CUDAGLBuffer<float4>> framebuffer = nullptr;
  std::unique_ptr<fredholm::Renderer> renderer = nullptr;

  Controller()
  {
    camera = std::make_unique<fredholm::Camera>();
    scene = std::make_unique<fredholm::Scene>();
    renderer = std::make_unique<fredholm::Renderer>();
  }

  void init_camera(const CameraSettings& params)
  {
    camera = std::make_unique<fredholm::Camera>(params.origin, params.forward,
                                                params.fov);
  }

  void move_camera(const fredholm::CameraMovement& direction, float dt)
  {
    camera->move(direction, dt);
    init_render_states();
  }

  void rotate_camera(float dphi, float dtheta)
  {
    camera->lookAround(dphi, dtheta);
    init_render_states();
  }

  void init_renderer()
  {
    renderer->create_context();
    renderer->create_module(std::filesystem::path(MODULES_SOURCE_DIR) /
                            "pt.ptx");
    renderer->create_program_group();
    renderer->create_pipeline();
  }

  // TODO: clear scene before loading
  void load_scene(const std::filesystem::path& filepath)
  {
    scene->load_obj(filepath);
    renderer->load_scene(*scene);
    renderer->build_accel();
    renderer->create_sbt();
  }

  void set_resolution(uint32_t width, uint32_t height)
  {
    framebuffer = std::make_unique<app::CUDAGLBuffer<float4>>(width, height);
    renderer->set_resolution(width, height);
  }

  void init_render_states() { renderer->init_render_states(); }

  void render(uint32_t n_samples, uint32_t max_depth)
  {
    renderer->render(*camera, framebuffer->get_device_ptr(), n_samples,
                     max_depth);
    // TODO: Is is safe to remove this?
    renderer->wait_for_completion();
  }

  const gcss::Buffer<float4>& get_gl_framebuffer() const
  {
    return framebuffer->get_gl_buffer();
  }
};