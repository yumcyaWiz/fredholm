#pragma once
#include "cuda_util.h"
#include "render_strategy/pt/pt_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class PtStrategy : public RenderStrategy
{
   public:
    PtStrategy(const OptixDeviceContext& context, bool debug = false)
    {
        m_module = optix_create_module(context, "pt.ptx", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 1, 2, debug);
    }

    void render(uint32_t width, uint32_t height, const Camera& camera,
                const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle,
                const OptixShaderBindingTable& sbt,
                const RenderLayers& layers) override
    {
        // TODO: remove malloc from this function, maybe it's good to
        // initialize in the constructor
        cuda_check(
            cuMemAlloc(&sample_count, width * height * sizeof(uint32_t)));
        cuda_check(cuMemsetD32(sample_count, 0, width * height));

        PtStrategyParams params;
        params.width = width;
        params.height = height;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene);
        params.ias_handle = ias_handle;
        params.n_samples = n_samples;
        params.max_depth = max_depth;
        params.seed = seed;
        params.sample_count = reinterpret_cast<uint*>(sample_count);
        params.output = reinterpret_cast<float4*>(layers.beauty);

        cuda_check(cuMemAlloc(&m_params_buffer, sizeof(PtStrategyParams)));
        cuda_check(
            cuMemcpyHtoD(m_params_buffer, &params, sizeof(PtStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, m_params_buffer,
                                sizeof(PtStrategyParams), &sbt, width, height,
                                1));

        cuda_check(cuMemFree(sample_count));
    }

   private:
    uint32_t n_samples = 512;
    uint32_t max_depth = 100;
    uint32_t seed = 1;
    CUdeviceptr sample_count = 0;
};

}  // namespace fredholm