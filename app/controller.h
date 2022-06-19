#pragma once
#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

#include "camera.h"
#include "cuda_gl_util.h"
#include "renderer.h"
#include "scene.h"

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

  void init_render_state(const RenderSettings& settings)
  {
    renderer->init_render_states();
  }
};