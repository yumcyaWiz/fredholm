#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include <memory>

#include "cwl/buffer.h"
#include "cwl/util.h"
#include "optwl/optwl.h"

namespace fredholm
{

class Denoiser
{
 public:
  Denoiser(OptixDeviceContext context, uint32_t width, uint32_t height,
           const float4* d_beauty, const float4* d_normal,
           const float4* d_albedo, const float4* d_denoised)
      : m_context(context),
        m_width(width),
        m_height(height),
        m_d_beauty(d_beauty),
        m_d_normal(d_normal),
        m_d_albedo(d_albedo),
        m_d_denoised(d_denoised)
  {
    init_denoiser();
    init_layers();
  }

  ~Denoiser() noexcept(false) { OPTIX_CHECK(optixDenoiserDestroy(m_denoiser)); }

  void init_denoiser()
  {
    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;

    // create denoiser
    OptixDenoiserModelKind model_kind = OPTIX_DENOISER_MODEL_KIND_HDR;
    OPTIX_CHECK(
        optixDenoiserCreate(m_context, model_kind, &options, &m_denoiser));

    // compute required memory size
    OptixDenoiserSizes denoiser_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_width,
                                                    m_height, &denoiser_sizes));
    m_scratch_size =
        static_cast<uint32_t>(denoiser_sizes.withoutOverlapScratchSizeInBytes);
    m_state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

    // allocate state, scratch buffer
    m_scratch = std::make_unique<cwl::CUDABuffer<uint8_t>>(m_scratch_size);
    m_state = std::make_unique<cwl::CUDABuffer<uint8_t>>(m_state_size);

    // setup denoiser
    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser, nullptr, m_width, m_height,
        reinterpret_cast<CUdeviceptr>(m_state->get_device_ptr()), m_state_size,
        reinterpret_cast<CUdeviceptr>(m_scratch->get_device_ptr()),
        m_scratch_size));

    // set denoiser params
    m_params.denoiseAlpha = 0;
    m_params.blendFactor = 0.0f;
    // TODO: set these
    m_params.hdrIntensity = reinterpret_cast<CUdeviceptr>(nullptr);
    m_params.hdrAverageColor = reinterpret_cast<CUdeviceptr>(nullptr);
  }

  void init_layers()
  {
    m_guide_layer.normal = create_optix_image_2d(m_width, m_height, m_d_normal);
    m_guide_layer.albedo = create_optix_image_2d(m_width, m_height, m_d_albedo);

    m_layer.input = create_optix_image_2d(m_width, m_height, m_d_beauty);
    m_layer.output = create_optix_image_2d(m_width, m_height, m_d_denoised);
  }

  void denoise()
  {
    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser, nullptr, &m_params,
        reinterpret_cast<CUdeviceptr>(m_state->get_device_ptr()), m_state_size,
        &m_guide_layer, &m_layer, 1, 0, 0,
        reinterpret_cast<CUdeviceptr>(m_scratch->get_device_ptr()),
        m_scratch_size));
  }

  void wait_for_completion() const { CUDA_SYNC_CHECK(); }

  static void log_callback(unsigned int level, const char* tag,
                           const char* message, void* cbdata)
  {
    if (level == 4) {
      spdlog::info("[Denoiser][{}] {}", tag, message);
    } else if (level == 3) {
      spdlog::warn("[Denoiser][{}] {}", tag, message);
    } else if (level == 2) {
      spdlog::error("[Denoiser][{}] {}", tag, message);
    } else if (level == 1) {
      spdlog::critical("[Denoiser][{}] {}", tag, message);
    }
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  const float4* m_d_beauty = nullptr;
  const float4* m_d_normal = nullptr;
  const float4* m_d_albedo = nullptr;
  const float4* m_d_denoised = nullptr;

  OptixDeviceContext m_context = nullptr;
  OptixDenoiser m_denoiser = nullptr;
  OptixDenoiserParams m_params = {};
  OptixDenoiserGuideLayer m_guide_layer = {};
  OptixDenoiserLayer m_layer = {};

  uint32_t m_state_size = 0;
  uint32_t m_scratch_size = 0;
  std::unique_ptr<cwl::CUDABuffer<uint8_t>> m_state;
  std::unique_ptr<cwl::CUDABuffer<uint8_t>> m_scratch;

  static OptixImage2D create_optix_image_2d(uint32_t width, uint32_t height,
                                            const float4* d_image)
  {
    OptixImage2D oi;
    oi.width = width;
    oi.height = height;
    oi.rowStrideInBytes = width * sizeof(float4);
    oi.pixelStrideInBytes = sizeof(float4);
    oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    oi.data = reinterpret_cast<CUdeviceptr>(d_image);

    return oi;
  }
};

}  // namespace fredholm