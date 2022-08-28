#pragma once

#include <filesystem>
#include <memory>
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

enum class SkyType : int { CONSTANT, IBL, ARHOSEK };

static std::vector<std::filesystem::path> scene_filepaths = {
    "../resources/cornellbox/CornellBox.obj",
    "../resources/rtcamp8/rtcamp8.gltf",
    "../resources/sponza/sponza.obj",
    "../resources/pbr_sponza/Sponza.gltf",
    "../resources/modern_sponza/NewSponza_Main_Blender_glTF.gltf",
    "../resources/modern_sponza_small/NewSponza_Main_Blender_glTF.gltf",
    "../resources/salle_de_bain/salle_de_bain.obj",
    "../resources/sibenik/sibenik.obj",
    "../resources/san_miguel/san-miguel.obj",
    "../resources/rungholt/rungholt.obj",
    "../resources/vokselia/vokselia_spawn.obj",
    "../resources/bmw/bmw.obj",
    "../resources/bmw_gltf/bmw.gltf",
    "../resources/specular_test/spheres_test_scene.obj",
    "../resources/metal_test/spheres_test_scene.obj",
    "../resources/coat_test/spheres_test_scene.obj",
    "../resources/transmission_test/spheres_test_scene.obj",
    "../resources/transmission_test_sphere/sphere.obj",
    "../resources/specular_transmission_test/spheres_test_scene.obj",
    "../resources/sheen_test/spheres_test_scene.obj",
    "../resources/diffuse_test/spheres_test_scene.obj",
    "../resources/diffuse_transmission_test/spheres_test_scene.obj",
    "../resources/texture_test/plane.obj",
    "../resources/normalmap_test/normalmap_test.obj",
    "../resources/specular_white_furnace_test/spheres.obj",
    "../resources/coat_white_furnace_test/spheres.obj",
    "../resources/metal_rough_spheres/MetalRoughSpheres.gltf",
    "../resources/clear_coat_test/ClearCoatTest.gltf",
    "../resources/emission_texture_test/emission_texture_test.gltf",
    "../resources/instance_test/instance_test.gltf",
    "../resources/animation_test/animation_test.gltf",
    "../resources/camera_animation_test/camera_animation_test.gltf",
    "../resources/box/Box.gltf",
    "../resources/cube/Cube.gltf",
    "../resources/animated_cube/AnimatedCube.gltf",
    "../resources/gltf_test/gltf_test.gltf",
    "../resources/gltf_test2/gltf_test2.gltf"};

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
  float m_imgui_time = 0.0f;
  bool m_imgui_play_animation = false;
  float m_imgui_timestep = 0.01f;
  char m_imgui_filename[256] = "output.png";

  float m_imgui_origin[3] = {0, 1, 5};
  float m_imgui_fov = 90.0f;
  float m_imgui_F = 100.0f;
  float m_imgui_focus = 10000.0f;
  float m_imgui_movement_speed = 1.0f;
  float m_imgui_rotation_speed = 0.1f;

  float m_imgui_directional_light_le[3] = {0, 0, 0};
  float m_imgui_directional_light_dir[3] = {0, 1, 0};
  float m_imgui_directional_light_angle = 0.0f;

  SkyType m_imgui_sky_type = SkyType::CONSTANT;
  float m_imgui_bg_color[3] = {0, 0, 0};
  float m_imgui_sky_intensity = 1.0f;
  int m_imgui_ibl_id = 0;
  float m_imgui_arhosek_turbidity = 3.0f;
  float m_imgui_arhosek_albedo = 0.3f;

  float m_imgui_bloom_threshold = 1.0f;
  float m_imgui_bloom_sigma = 1.0f;
  float m_imgui_iso = 400.0f;

  std::unique_ptr<fredholm::Camera> m_camera = nullptr;

  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_beauty = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_position = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_normal = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float>> m_layer_depth = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_texcoord = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_albedo = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_denoised = nullptr;

  // post processed layer
  std::unique_ptr<cwl::CUDABuffer<float4>> m_beauty_high_luminance = nullptr;
  std::unique_ptr<cwl::CUDABuffer<float4>> m_denoised_high_luminance = nullptr;
  std::unique_ptr<cwl::CUDABuffer<float4>> m_beauty_temp = nullptr;
  std::unique_ptr<cwl::CUDABuffer<float4>> m_denoised_temp = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_beauty_pp = nullptr;
  std::unique_ptr<cwl::CUDAGLBuffer<float4>> m_layer_denoised_pp = nullptr;

  std::unique_ptr<optwl::Context> m_context = nullptr;
  std::unique_ptr<fredholm::Renderer> m_renderer = nullptr;
  std::unique_ptr<fredholm::Denoiser> m_denoiser = nullptr;

  Controller();

  void init_camera();
  void update_camera();
  void move_camera(const fredholm::CameraMovement& direction, float dt);
  void rotate_camera(float dphi, float dtheta);

  void init_renderer();

  void init_denoiser();

  void init_render_layers();
  void clear_render_layers();

  void load_scene();

  void update_directional_light();

  void update_sky_type();
  void set_sky_intensity();

  void load_ibl();

  void load_arhosek();

  void set_time();
  void advance_time();

  void update_resolution();

  void clear_render();

  void render();

  void denoise();

  void post_process();

  void save_image() const;
};