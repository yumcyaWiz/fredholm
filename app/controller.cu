#include "controller.h"
#include "cwl/buffer.h"
#include "kernels/post-process.h"
#include "stb_image_write.h"

Controller::Controller()
{
  m_camera = std::make_unique<fredholm::Camera>();

  // init CUDA
  CUDA_CHECK(cudaFree(0));

  m_context = std::make_unique<optwl::Context>();
  m_renderer = std::make_unique<fredholm::Renderer>(m_context->m_context);
}

void Controller::init_camera()
{
  const float3 origin =
      make_float3(m_imgui_origin[0], m_imgui_origin[1], m_imgui_origin[2]);
  m_camera = std::make_unique<fredholm::Camera>(origin, deg2rad(m_imgui_fov));
  m_camera->m_fov = deg2rad(m_imgui_fov);
  m_camera->m_F = m_imgui_F;
  m_camera->m_focus = m_imgui_focus;
  m_camera->m_movement_speed = m_imgui_movement_speed;
  m_camera->m_look_around_speed = m_imgui_rotation_speed;
}

void Controller::update_camera()
{
  m_camera->set_origin(
      make_float3(m_imgui_origin[0], m_imgui_origin[1], m_imgui_origin[2]));
  m_camera->m_fov = deg2rad(m_imgui_fov);
  m_camera->m_F = m_imgui_F;
  m_camera->m_focus = m_imgui_focus;
  m_camera->m_movement_speed = m_imgui_movement_speed;
  m_camera->m_look_around_speed = m_imgui_rotation_speed;
}

void Controller::move_camera(const fredholm::CameraMovement &direction,
                             float dt)
{
  m_camera->move(direction, dt);
  const float3 origin = m_camera->get_origin();
  m_imgui_origin[0] = origin.x;
  m_imgui_origin[1] = origin.y;
  m_imgui_origin[2] = origin.z;
}

void Controller::rotate_camera(float dphi, float dtheta)
{
  m_camera->lookAround(dphi, dtheta);
  // m_imgui_forward[0] = m_camera->m_forward.x;
  // m_imgui_forward[1] = m_camera->m_forward.y;
  // m_imgui_forward[2] = m_camera->m_forward.z;
}

void Controller::init_renderer()

{
  m_renderer->create_module(std::filesystem::path(MODULES_SOURCE_DIR) /
                            "pt.ptx");
  m_renderer->create_program_group();
  m_renderer->create_pipeline();
  m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);
}

void Controller::init_denoiser()
{
  m_layer_denoised = std::make_unique<cwl::CUDAGLBuffer<float4>>(
      m_imgui_resolution[0] * m_imgui_resolution[1]);
  m_denoiser = std::make_unique<fredholm::Denoiser>(
      m_context->m_context, m_imgui_resolution[0], m_imgui_resolution[1],
      m_layer_beauty->get_device_ptr(), m_layer_normal->get_device_ptr(),
      m_layer_albedo->get_device_ptr(), m_layer_denoised->get_device_ptr());
}

void Controller::init_render_layers()
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

  m_layer_beauty_pp = std::make_unique<cwl::CUDAGLBuffer<float4>>(
      m_imgui_resolution[0] * m_imgui_resolution[1]);
  m_layer_denoised_pp = std::make_unique<cwl::CUDAGLBuffer<float4>>(
      m_imgui_resolution[0] * m_imgui_resolution[1]);
}

void Controller::clear_render_layers()
{
  m_layer_beauty->clear();
  m_layer_position->clear();
  m_layer_normal->clear();
  m_layer_depth->clear();
  m_layer_texcoord->clear();
  m_layer_albedo->clear();

  m_layer_beauty_pp->clear();
  m_layer_denoised_pp->clear();
}

void Controller::load_scene()
{
  m_renderer->load_scene(scene_filepaths[m_imgui_scene_id]);
  m_renderer->build_gas();
  m_renderer->build_ias();
  m_renderer->create_sbt();
}

void Controller::update_directional_light()
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

void Controller::update_sky_type()
{
  switch (m_imgui_sky_type) {
    case SkyType::CONSTANT: {
      m_renderer->clear_ibl();
      m_renderer->clear_arhosek_sky();
    } break;
    case SkyType::IBL: {
      m_renderer->clear_arhosek_sky();
      load_ibl();
    } break;
    case SkyType::ARHOSEK: {
      m_renderer->clear_ibl();
      load_arhosek();
    } break;
  }
}

void Controller::set_sky_intensity()
{
  m_renderer->set_sky_intensity(m_imgui_sky_intensity);
}

void Controller::load_ibl()
{
  m_renderer->load_ibl(ibl_filepaths[m_imgui_ibl_id]);
}

void Controller::load_arhosek()
{
  m_renderer->load_arhosek_sky(m_imgui_arhosek_turbidity,
                               m_imgui_arhosek_albedo);
}

void Controller::set_time() { m_renderer->set_time(m_imgui_time); }

void Controller::advance_time()
{
  m_imgui_time += m_imgui_timestep;
  m_renderer->set_time(m_imgui_time);
}

void Controller::update_resolution()
{
  init_render_layers();
  m_renderer->set_resolution(m_imgui_resolution[0], m_imgui_resolution[1]);

  init_denoiser();
}

void Controller::clear_render()
{
  m_imgui_n_samples = 0;
  clear_render_layers();
  m_renderer->init_render_states();
}

void Controller::render()
{
  if (m_imgui_play_animation) {
    advance_time();
    clear_render();
  }

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

void Controller::denoise()
{
  m_denoiser->denoise();
  m_denoiser->wait_for_completion();
}

void Controller::post_process()
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(m_imgui_resolution[0] / threads_per_block.x,
                    m_imgui_resolution[1] / threads_per_block.y);
  post_process_kernel<<<blocks, threads_per_block>>>(
      m_layer_beauty->get_device_ptr(), m_layer_denoised->get_device_ptr(),
      m_imgui_resolution[0], m_imgui_resolution[1], m_imgui_iso,
      m_layer_beauty_pp->get_device_ptr(),
      m_layer_denoised_pp->get_device_ptr());
}

void Controller::save_image() const
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
  stbi_write_png(m_imgui_filename, m_imgui_resolution[0], m_imgui_resolution[1],
                 4, image_c4.data(), sizeof(uchar4) * m_imgui_resolution[0]);

  spdlog::info("[GUI] image saved as {}", m_imgui_filename);
}