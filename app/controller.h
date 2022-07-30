#pragma once

#include <filesystem>
#include <memory>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//
#include "cwl/buffer.h"
//
#include "optwl/optwl.h"
//
#include "fredholm/camera.h"
#include "fredholm/denoiser.h"
#include "fredholm/renderer.h"
#include "fredholm/scene.h"
#include "fredholm/shared.h"

inline float deg2rad(float deg) { return deg / 180.0f * M_PI; }

enum class AOVType : int {
  BEAUTY,
  DENOISED,
  POSITION,
  NORMAL,
  TEXCOORD,
  DEPTH,
  ALBEDO
};

enum class SkyType : int { CONSTANT, IBL };

static std::vector<std::filesystem::path> scene_filepaths = {
    "../resources/cornellbox/CornellBox.obj",
    "../resources/sponza/sponza.obj",
    "../resources/salle_de_bain/salle_de_bain.obj",
    "../resources/sibenik/sibenik.obj",
    "../resources/san_miguel/san-miguel.obj",
    "../resources/rungholt/rungholt.obj",
    "../resources/vokselia/vokselia_spawn.obj",
    "../resources/bmw/bmw.obj",
    "../resources/specular_test/spheres_test_scene.obj",
    "../resources/metal_test/spheres_test_scene.obj",
    "../resources/coat_test/spheres_test_scene.obj",
    "../resources/transmission_test/spheres_test_scene.obj",
    "../resources/transmission_test_sphere/sphere.obj",
    "../resources/specular_transmission_test/spheres_test_scene.obj",
    "../resources/diffuse_transmission_test/spheres_test_scene.obj",
    "../resources/texture_test/plane.obj",
    "../resources/normalmap_test/normalmap_test.obj",
    "../resources/specular_white_furnace_test/spheres.obj",
    "../resources/coat_white_furnace_test/spheres.obj"};

static std::vector<std::filesystem::path> ibl_filepaths = {
    "../resources/ibl/PaperMill_Ruins_E/PaperMill_E_3k.hdr"};

class Controller
{
 public:
  int m_imgui_scene_id = 0;
  int m_imgui_resolution[2] = {1920, 1080};
  int m_imgui_n_samples = 0;
  int m_imgui_max_samples = 100;
  int m_imgui_max_depth = 10;
  AOVType m_imgui_aov_type = AOVType::BEAUTY;
  char m_imgui_filename[256] = "output.png";

  float m_imgui_origin[3] = {0, 1, 5};
  float m_imgui_forward[3] = {0, 0, -1};
  float m_imgui_fov = 90.0f;
  float m_imgui_movement_speed = 1.0f;
  float m_imgui_rotation_speed = 0.1f;

  float m_imgui_directional_light_le[3] = {0, 0, 0};
  float m_imgui_directional_light_dir[3] = {0, 1, 0};
  float m_imgui_directional_light_angle = 0.0f;

  SkyType m_imgui_sky_type = SkyType::CONSTANT;
  float m_imgui_bg_color[3] = {0, 0, 0};
  int m_imgui_ibl_id = 0;

  std::unique_ptr<fredholm::Camera> m_camera = nullptr;
  std::unique_ptr<fredholm::Scene> m_scene = nullptr;

  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_beauty = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_position = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_normal = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float>> m_layer_depth = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_texcoord = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_albedo = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_denoised = nullptr;

  std::unique_ptr<optwl::Context> m_context = nullptr;
  std::unique_ptr<fredholm::Renderer> m_renderer = nullptr;
  std::unique_ptr<fredholm::Denoiser> m_denoiser = nullptr;

  Controller()
  {
    m_camera = std::make_unique<fredholm::Camera>();
    m_scene = std::make_unique<fredholm::Scene>();

    // init CUDA
    CUDA_CHECK(cudaFree(0));

    m_context = std::make_unique<optwl::Context>();
    m_renderer = std::make_unique<fredholm::Renderer>(m_context->m_context);
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
    m_camera->set_origin(
        make_float3(m_imgui_origin[0], m_imgui_origin[1], m_imgui_origin[2]));
    m_camera->set_forward(make_float3(m_imgui_forward[0], m_imgui_forward[1],
                                      m_imgui_forward[2]));
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
    m_renderer->create_module(std::filesystem::path(MODULES_SOURCE_DIR) /
                              "pt.ptx");
    m_renderer->create_program_group();
    m_renderer->create_pipeline();
    m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);
  }

  void init_denoiser()
  {
    m_layer_denoised = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_denoiser = std::make_unique<fredholm::Denoiser>(
        m_context->m_context, m_imgui_resolution[0], m_imgui_resolution[1],
        m_layer_beauty->get_device_ptr(), m_layer_normal->get_device_ptr(),
        m_layer_albedo->get_device_ptr(), m_layer_denoised->get_device_ptr());
  }

  void init_render_layers()
  {
    m_layer_beauty = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_layer_position = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_layer_normal = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_layer_depth = std::make_unique<cwl::CUDAGLBuffer<float>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_layer_texcoord = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
    m_layer_albedo = std::make_unique<cwl::CUDAGLBuffer<float4>>(
        m_imgui_resolution[0] * m_imgui_resolution[1]);
  }

  void clear_render_layers()
  {
    m_layer_beauty->clear();
    m_layer_position->clear();
    m_layer_normal->clear();
    m_layer_depth->clear();
    m_layer_texcoord->clear();
    m_layer_albedo->clear();
  }

  void load_scene()
  {
    m_scene->load_obj(scene_filepaths[m_imgui_scene_id]);

    m_renderer->load_scene(*m_scene);
    m_renderer->build_accel();
    m_renderer->create_sbt();
  }

  void update_directional_light()
  {
    m_renderer->set_directional_light(
        make_float3(m_imgui_directional_light_le[0],
                    m_imgui_directional_light_le[1],
                    m_imgui_directional_light_le[2]),
        make_float3(m_imgui_directional_light_dir[0],
                    m_imgui_directional_light_dir[1],
                    m_imgui_directional_light_dir[2]),
        m_imgui_directional_light_angle);
  }

  void update_sky_type()
  {
    switch (m_imgui_sky_type) {
      case SkyType::CONSTANT: {
        m_renderer->clear_ibl();
      } break;
      case SkyType::IBL: {
        load_ibl();
      } break;
    }
  }

  void load_ibl() { m_renderer->load_ibl(ibl_filepaths[m_imgui_ibl_id]); }

  void update_resolution()
  {
    init_render_layers();
    m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);

    init_denoiser();
  }

  void clear_render()
  {
    m_imgui_n_samples = 0;
    clear_render_layers();
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
      render_layers.texcoord = m_layer_texcoord->get_device_ptr();
      render_layers.albedo = m_layer_albedo->get_device_ptr();

      m_renderer->render(*m_camera,
                         make_float3(m_imgui_bg_color[0], m_imgui_bg_color[1],
                                     m_imgui_bg_color[2]),
                         render_layers, 1, m_imgui_max_depth);
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

  void save_image() const
  {
    // copy image from device to host
    std::vector<float4> image_f4;
    switch (m_imgui_aov_type) {
      case AOVType::BEAUTY: {
        m_layer_beauty->copy_from_device_to_host(image_f4);
      } break;
      case AOVType::DENOISED: {
        m_layer_denoised->copy_from_device_to_host(image_f4);
      } break;
      case AOVType::POSITION: {
        m_layer_position->copy_from_device_to_host(image_f4);
      } break;
      case AOVType::NORMAL: {
        m_layer_normal->copy_from_device_to_host(image_f4);
      } break;
      // TODO: handle depth case
      case AOVType::DEPTH: {
      } break;
      case AOVType::TEXCOORD: {
        m_layer_texcoord->copy_from_device_to_host(image_f4);
      } break;
      case AOVType::ALBEDO: {
        m_layer_albedo->copy_from_device_to_host(image_f4);
      } break;
    }

    // convert float4 to uchar4
    std::vector<uchar4> image_c4(image_f4.size());
    for (int j = 0; j < m_imgui_resolution[1]; ++j) {
      for (int i = 0; i < m_imgui_resolution[0]; ++i) {
        const int idx = i + m_imgui_resolution[0] * j;
        float4 v = image_f4[idx];

        // gamma correction
        if (m_imgui_aov_type == AOVType::BEAUTY ||
            m_imgui_aov_type == AOVType::DENOISED ||
            m_imgui_aov_type == AOVType::ALBEDO) {
          v.x = std::pow(v.x, 1.0f / 2.2f);
          v.y = std::pow(v.y, 1.0f / 2.2f);
          v.z = std::pow(v.z, 1.0f / 2.2f);
        }

        image_c4[idx].x =
            static_cast<unsigned char>(std::clamp(255.0f * v.x, 0.0f, 255.0f));
        image_c4[idx].y =
            static_cast<unsigned char>(std::clamp(255.0f * v.y, 0.0f, 255.0f));
        image_c4[idx].z =
            static_cast<unsigned char>(std::clamp(255.0f * v.z, 0.0f, 255.0f));
        image_c4[idx].w =
            static_cast<unsigned char>(std::clamp(255.0f * v.w, 0.0f, 255.0f));
      }
    }

    // save image
    stbi_write_png(m_imgui_filename, m_imgui_resolution[0],
                   m_imgui_resolution[1], 4, image_c4.data(),
                   sizeof(uchar4) * m_imgui_resolution[0]);

    spdlog::info("[GUI] image saved as {}", m_imgui_filename);
  }
};