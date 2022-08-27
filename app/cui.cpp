#include <stdexcept>

#include "argparse/argparse.hpp"

#include "cwl/util.h"
#include "fredholm/renderer.h"
#include "fredholm/denoiser.h"
#include "kernels/post-process.h"
#include "optwl/optwl.h"
#include "cwl/buffer.h"
#include "stb_image_write.h"

inline float deg2rad(float deg) { return deg / 180.0f * M_PI; }

int main(int argc, char *argv[])
{
  argparse::ArgumentParser program("fredholm");

  program.add_argument("width").help("width").scan<'i', int>();
  program.add_argument("height").help("height").scan<'i', int>();
  program.add_argument("spp").help("number of samples").scan<'i', int>();
  program.add_argument("scene").help("glTF scene file");

  try {
    program.parse_args(argc, argv);
  }
  catch(const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(EXIT_FAILURE);
  }

  const int width = program.get<int>("width");
  const int height = program.get<int>("height");
  const int n_spp = program.get<int>("spp");
  const std::string scene_filepath = program.get("scene");

  const int max_depth = 5;
  const float ISO = 100.0f;

  // init CUDA
  CUDA_CHECK(cudaFree(0));
  optwl::Context context;

  // init renderer
  fredholm::Renderer renderer(context.m_context);
  renderer.create_module(std::filesystem::path(MODULES_SOURCE_DIR) / "pt.ptx");
  renderer.create_program_group();
  renderer.create_pipeline();
  renderer.set_resolution(width, height);

  // init render layers
  cwl::CUDABuffer<float4> layer_beauty = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_position = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_normal = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float> layer_depth = cwl::CUDABuffer<float>(width * height);
  cwl::CUDABuffer<float4> layer_texcoord = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_albedo = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_denoised = cwl::CUDABuffer<float4>(width * height);

  cwl::CUDABuffer<float4> layer_beauty_pp = cwl::CUDABuffer<float4>(width * height);
  cwl::CUDABuffer<float4> layer_denoised_pp = cwl::CUDABuffer<float4>(width * height);

  // init denoiser
  fredholm::Denoiser denoiser = fredholm::Denoiser(context.m_context, width, height, layer_beauty.get_device_ptr(), layer_normal.get_device_ptr(), layer_albedo.get_device_ptr(), layer_denoised.get_device_ptr());

  // load scene
  renderer.load_scene(scene_filepath);
  renderer.build_gas();
  renderer.build_ias();
  renderer.create_sbt();

  // init camera
  fredholm::Camera camera;
  camera.m_fov = deg2rad(90.0f);
  camera.m_F = 100.0f;
  camera.m_focus = 10000.0f;

  // init render layer
  fredholm::RenderLayer render_layer;
  render_layer.beauty = layer_beauty.get_device_ptr();
  render_layer.position = layer_position.get_device_ptr();
  render_layer.normal = layer_normal.get_device_ptr();
  render_layer.depth = layer_depth.get_device_ptr();
  render_layer.texcoord = layer_texcoord.get_device_ptr();
  render_layer.albedo = layer_albedo.get_device_ptr();

  // set arhosek sky
  renderer.load_arhosek_sky(3.0f, 0.2f);

  // render
  renderer.set_time(0.0f);
  renderer.render(camera, make_float3(0, 0, 0), render_layer, n_spp, max_depth);
  CUDA_SYNC_CHECK();

  // denoise
  denoiser.denoise();
  CUDA_SYNC_CHECK();
  
  // post process
  post_process_launch(layer_beauty.get_device_ptr(), layer_denoised.get_device_ptr(), width, height, ISO, layer_beauty_pp.get_device_ptr(), layer_denoised_pp.get_device_ptr());
  CUDA_SYNC_CHECK();

  // save image
  std::vector<float4> image_f4;
  layer_denoised_pp.copy_from_device_to_host(image_f4);

  std::vector<uchar4> image_c4(width * height);
  for(int j = 0; j < height; ++j) {
    for(int i = 0; i < width; ++i) {
      const int idx = i + width * j;
      const float4& v = image_f4[idx];
      image_c4[idx].x = static_cast<unsigned char>(std::clamp(255.0f * v.x, 0.0f, 255.0f));
      image_c4[idx].y = static_cast<unsigned char>(std::clamp(255.0f * v.y, 0.0f, 255.0f));
      image_c4[idx].z = static_cast<unsigned char>(std::clamp(255.0f * v.z, 0.0f, 255.0f));
      image_c4[idx].w = static_cast<unsigned char>(std::clamp(255.0f * v.w, 0.0f, 255.0f));
    }
  }

  stbi_write_png("0.png", width, height, 4, image_c4.data(), sizeof(uchar4) * width);

  return 0;
}