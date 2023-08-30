#pragma once
#include <optix.h>

#include "cuda_util.h"
#include "optix_util.h"

namespace fredholm
{

class Denoiser
{
   public:
    Denoiser(OptixDeviceContext context, uint32_t width, uint32_t height)
        : context(context), width(width), height(height)
    {
    }

    void init_denoiser()
    {
        OptixDenoiserOptions options = {};
        options.guideAlbedo = 1;
        options.guideNormal = 1;

        OptixDenoiserModelKind model_kind = OPTIX_DENOISER_MODEL_KIND_HDR;
        optix_check(
            optixDenoiserCreate(context, model_kind, &options, &denoiser));

        OptixDenoiserSizes denoiser_sizes;
        optix_check(optixDenoiserComputeMemoryResources(denoiser, width, height,
                                                        &denoiser_sizes));
        scratch_size = static_cast<uint32_t>(
            denoiser_sizes.withoutOverlapScratchSizeInBytes);
        state_size = static_cast<uint32_t>(denoiser_sizes.stateSizeInBytes);

        cuda_check(cuMemAlloc(&state_buffer, state_size));
        cuda_check(cuMemAlloc(&scratch_buffer, scratch_size));

        optix_check(optixDenoiserSetup(denoiser, nullptr, width, height,
                                       state_buffer, state_size, scratch_buffer,
                                       scratch_size));

        // denoiser_params.denoiseAlpha = 0;
        // denoiser_params.blendFactor = 0;
        // TODO: set these
        denoiser_params.hdrIntensity = 0;
        denoiser_params.hdrAverageColor = 0;
    }

    void denoise(CUdeviceptr beauty, CUdeviceptr normal, CUdeviceptr albedo,
                 CUdeviceptr denoised) const
    {
        OptixDenoiserLayer denoiser_layer = {};
        denoiser_layer.input = create_optix_image_2d(width, height, beauty);
        denoiser_layer.output = create_optix_image_2d(width, height, denoised);

        OptixDenoiserGuideLayer guide_layer = {};
        guide_layer.normal = create_optix_image_2d(width, height, normal);
        guide_layer.albedo = create_optix_image_2d(width, height, albedo);

        optix_check(optixDenoiserInvoke(denoiser, nullptr, &denoiser_params,
                                        state_buffer, state_size, &guide_layer,
                                        &denoiser_layer, 1, 0, 0,
                                        scratch_buffer, scratch_size));
    }

   private:
    static OptixImage2D create_optix_image_2d(uint32_t width, uint32_t height,
                                              CUdeviceptr image)
    {
        OptixImage2D oi;
        oi.width = width;
        oi.height = height;
        oi.rowStrideInBytes = width * sizeof(float4);
        oi.pixelStrideInBytes = sizeof(float4);
        oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        oi.data = image;
        return oi;
    }

    uint32_t width = 0;
    uint32_t height = 0;

    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;
    OptixDenoiserParams denoiser_params = {};

    uint32_t state_size = 0;
    uint32_t scratch_size = 0;
    CUdeviceptr state_buffer = 0;
    CUdeviceptr scratch_buffer = 0;
};

}  // namespace fredholm