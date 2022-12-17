#include <chrono>
#include <mutex>
#include <queue>
#include <ratio>
#include <stdexcept>
#include <thread>

#include "cwl/buffer.h"
#include "cwl/util.h"
#include "fredholm/denoiser.h"
#include "fredholm/renderer.h"
#include "kernels/post-process.h"
#include "optwl/optwl.h"
#include "spdlog/spdlog.h"
#include "stb_image_write.h"

inline float deg2rad(float deg) { return deg / 180.0f * M_PI; }

class Timer
{
 public:
  Timer() {}

  void start() { m_start = std::chrono::steady_clock::now(); }

  void end() { m_end = std::chrono::steady_clock::now(); }

  template <typename T>
  int elapsed() const
  {
    return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() -
                                         m_start)
        .count();
  }

  template <typename T>
  int duration() const
  {
    return std::chrono::duration_cast<T>(m_end - m_start).count();
  }

 private:
  std::chrono::steady_clock::time_point m_start;
  std::chrono::steady_clock::time_point m_end;
};

int main()
{
  const int width = 1920;
  const int height = 1080;
  const bool upscale = false;
  const int n_spp = 16;
  const std::string scene_filepath =
      "../resources/camera_animation_test/camera_animation_test.gltf";

  const int max_depth = 5;
  const float ISO = 80.0f;
  const float chromatic_aberration = 1.0f;
  const float bloom_threshold = 2.0f;
  const float bloom_sigma = 5.0f;
  const float max_time = 9.5f;
  const float fps = 24.0f;
  const float time_step = 1.0f / fps;
  const int kill_time = 590;

  const int width_denoised = upscale ? 2 * width : width;
  const int height_denoised = upscale ? 2 * height : height;

  // if global timer elapsed greater than kill time, program will exit
  // immediately
  Timer global_timer;
  global_timer.start();

  // init CUDA
  CUDA_CHECK(cudaFree(0));
  optwl::Context context;

  // init renderer
  fredholm::Renderer renderer(context.m_context);
  renderer.create_module("./fredholm/CMakeFiles/modules.dir/modules/pt.ptx");
  renderer.create_program_group();
  renderer.create_pipeline();
  renderer.set_resolution(width, height);

  // init render layers
  cwl::CUDABuffer<float4> layer_beauty =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_position =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_normal =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float> layer_depth = cwl::CUDABuffer<float>(width * height);
  cwl::CUDABuffer<float4> layer_texcoord =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_albedo =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_denoised =
      cwl::CUDABuffer<float4>(width_denoised * height_denoised);

  cwl::CUDABuffer<float4> layer_beauty_pp =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_denoised_pp =
      cwl::CUDABuffer<float4>(width_denoised * height_denoised);

  cwl::CUDABuffer<float4> beauty_high_luminance =
      cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> denoised_high_luminance =
      cwl::CUDABuffer<float4>(width_denoised * height_denoised);
  cwl::CUDABuffer<float4> beauty_temp = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> denoised_temp =
      cwl::CUDABuffer<float4>(width_denoised * height_denoised);

  // init denoiser
  fredholm::Denoiser denoiser = fredholm::Denoiser(
      context.m_context, width, height, layer_beauty.get_device_ptr(),
      layer_normal.get_device_ptr(), layer_albedo.get_device_ptr(),
      layer_denoised.get_device_ptr(), upscale);

  // load scene
  renderer.load_scene("../resources/rtcamp8/rtcamp8.obj");
  renderer.load_scene("../resources/rtcamp8/rtcamp8_camera.gltf", false);
  renderer.build_gas();
  renderer.build_ias();
  renderer.create_sbt();

  // init camera
  fredholm::Camera camera;
  camera.m_fov = deg2rad(60.0f);
  camera.m_F = 100.0f;
  camera.m_focus = 8.0f;

  // init render layer
  fredholm::RenderLayer render_layer;
  render_layer.beauty = layer_beauty.get_device_ptr();
  render_layer.position = layer_position.get_device_ptr();
  render_layer.normal = layer_normal.get_device_ptr();
  render_layer.depth = layer_depth.get_device_ptr();
  render_layer.texcoord = layer_texcoord.get_device_ptr();
  render_layer.albedo = layer_albedo.get_device_ptr();

  // set directional light
  renderer.set_directional_light(make_float3(20, 20, 20),
                                 make_float3(-0.1f, 1, 0.1f), 1.0f);

  // set arhosek sky
  renderer.load_arhosek_sky(3.0f, 0.3f);

  Timer render_timer;
  Timer denoiser_timer;
  Timer pp_timer;
  Timer transfer_timer;
  Timer convert_timer;
  Timer save_timer;

  std::queue<std::pair<int, std::vector<float4>>> queue;
  std::mutex queue_mutex;
  bool render_finished = false;

  std::thread render_thread([&] {
    int frame_idx = 0;
    float time = 0.0f;

    while (true) {
      spdlog::info("[Render] rendering frame: {}", frame_idx);

      if (time > max_time ||
          global_timer.elapsed<std::chrono::seconds>() > kill_time) {
        render_finished = true;
        break;
      }

      // clear render layers
      layer_beauty.clear();
      layer_position.clear();
      layer_normal.clear();
      layer_depth.clear();
      layer_texcoord.clear();
      layer_albedo.clear();

      // clear render states
      renderer.init_render_states();

      // render
      render_timer.start();
      renderer.set_time(time);
      renderer.render(camera, make_float3(0, 0, 0), render_layer, n_spp,
                      max_depth);
      CUDA_SYNC_CHECK();
      render_timer.end();

      spdlog::info("[Render] rendering time: {}",
                   render_timer.duration<std::chrono::milliseconds>());

      // denoise
      denoiser_timer.start();
      denoiser.denoise();
      CUDA_SYNC_CHECK();
      denoiser_timer.end();

      spdlog::info("[Render] denoising time: {}",
                   denoiser_timer.duration<std::chrono::milliseconds>());

      // post process
      PostProcessParams params;
      params.use_bloom = true;
      params.bloom_threshold = bloom_threshold;
      params.bloom_sigma = bloom_sigma;
      params.chromatic_aberration = chromatic_aberration;
      params.ISO = ISO;

      pp_timer.start();
      post_process_kernel_launch(layer_beauty.get_device_ptr(),
                                 beauty_high_luminance.get_device_ptr(),
                                 beauty_temp.get_device_ptr(), width, height,
                                 params, layer_beauty_pp.get_device_ptr());
      post_process_kernel_launch(layer_denoised.get_device_ptr(),
                                 denoised_high_luminance.get_device_ptr(),
                                 denoised_temp.get_device_ptr(), width_denoised,
                                 height_denoised, params,
                                 layer_denoised_pp.get_device_ptr());
      CUDA_SYNC_CHECK();
      pp_timer.end();

      spdlog::info("[Render] post process time: {}",
                   pp_timer.duration<std::chrono::milliseconds>());

      // copy image from device to host
      std::vector<float4> image_f4;
      transfer_timer.start();
      layer_denoised_pp.copy_from_device_to_host(image_f4);
      transfer_timer.end();

      spdlog::info("[Render] transfer time: {}",
                   transfer_timer.duration<std::chrono::milliseconds>());

      // add image to queue
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push({frame_idx, image_f4});
      }

      // go to next frame
      frame_idx++;
      time += time_step;
    }
  });

  std::thread save_thread([&] {
    while (true) {
      if (global_timer.elapsed<std::chrono::seconds>() > kill_time) { break; }
      if (render_finished && queue.empty()) { break; }

      if (queue.empty()) continue;

      // get image from queue
      int frame_idx;
      std::vector<float4> image_f4;
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_idx = queue.front().first;
        image_f4 = queue.front().second;
        queue.pop();
      }

      // convert float image to uchar image
      convert_timer.start();
      std::vector<uchar4> image_c4(width_denoised * height_denoised);
      for (int j = 0; j < height_denoised; ++j) {
        for (int i = 0; i < width_denoised; ++i) {
          const int idx = i + width_denoised * j;
          const float4& v = image_f4[idx];
          image_c4[idx].x = static_cast<unsigned char>(
              std::clamp(255.0f * v.x, 0.0f, 255.0f));
          image_c4[idx].y = static_cast<unsigned char>(
              std::clamp(255.0f * v.y, 0.0f, 255.0f));
          image_c4[idx].z = static_cast<unsigned char>(
              std::clamp(255.0f * v.z, 0.0f, 255.0f));
          image_c4[idx].w = 255;
        }
      }
      convert_timer.end();

      spdlog::info("[Image Write] convert time: {}",
                   convert_timer.duration<std::chrono::milliseconds>());

      save_timer.start();
      const std::string filename =
          "output/" + std::to_string(frame_idx) + ".png";
      stbi_write_png(filename.c_str(), width_denoised, height_denoised, 4,
                     image_c4.data(), sizeof(uchar4) * width_denoised);
      save_timer.end();

      spdlog::info("[Image Write] {} saved", filename);
      spdlog::info("[Image Write] image save time: {}",
                   save_timer.duration<std::chrono::milliseconds>());
    }
  });

  render_thread.join();
  save_thread.join();

  return 0;
}