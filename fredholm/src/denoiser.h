#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include <memory>

#include "device/buffer.h"
#include "device/util.h"

namespace fredholm
{

class Denoiser
{
 public:
  Denoiser(OptixDeviceContext context, uint32_t width, uint32_t height,
           const float4* d_beauty, const float4* d_denoised)
      : m_context(context),
        m_width(width),
        m_height(height),
        m_d_beauty(d_beauty),
        m_d_denoised(d_denoised)
  {
    init_denoiser();
    init_layers();
  }

  ~Denoiser() noexcept(false)
  {
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
  }

  void init_denoiser()
  {
    OptixDenoiserOptions options = {};
    // TODO: set these
    options.guideAlbedo = 0;
    options.guideNormal = 0;

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
    m_scratch = std::make_unique<CUDABuffer<uint8_t>>(m_scratch_size);
    m_state = std::make_unique<CUDABuffer<uint8_t>>(m_state_size);

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
    m_layer.input = create_optix_image_2d(m_width, m_height, m_d_beauty);
    m_layer.output = create_optix_image_2d(m_width, m_height, m_d_denoised);
  }

  void denoise()
  {
    OPTIX_CHECK(optixDenoiserInvoke(
        m_denoiser, nullptr, &m_params,
        reinterpret_cast<CUdeviceptr>(m_state->get_device_ptr()), m_state_size,
        nullptr, &m_layer, 1, 0, 0,
        reinterpret_cast<CUdeviceptr>(m_scratch->get_device_ptr()),
        m_scratch_size));

    // TODO: remove this?
    CUDA_SYNC_CHECK();
  }

 private:
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  const float4* m_d_beauty = nullptr;
  const float4* m_d_denoised = nullptr;

  OptixDeviceContext m_context = nullptr;
  OptixDenoiser m_denoiser = nullptr;
  OptixDenoiserParams m_params = {};
  OptixDenoiserLayer m_layer = {};

  uint32_t m_state_size = 0;
  uint32_t m_scratch_size = 0;
  std::unique_ptr<CUDABuffer<uint8_t>> m_state;
  std::unique_ptr<CUDABuffer<uint8_t>> m_scratch;

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