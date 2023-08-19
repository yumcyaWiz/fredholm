#pragma once
#include "cuda_util.h"
#include "render_strategy/render_strategy.h"
#include "simple_shared.h"

namespace fredholm
{

class SimpleStrategy : public RenderStrategy
{
   public:
    SimpleStrategy(const OptixDeviceContext& context, bool debug = false)
    {
        m_module = optix_create_module(context, "simple.ptx", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "rg", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 1, 2, debug);

        cuda_check(cuMemAlloc(&params_buffer, sizeof(SimpleStrategyParams)));
    }

    ~SimpleStrategy()
    {
        if (params_buffer != 0)
        {
            cuda_check(cuMemFree(params_buffer));
            params_buffer = 0;
        }
    }

    void render(uint32_t width, uint32_t height, const Camera& camera,
                const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle,
                const OptixShaderBindingTable& sbt,
                const RenderLayers& layers) override
    {
        SimpleStrategyParams params;
        params.width = width;
        params.height = height;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene);
        params.ias_handle = ias_handle;
        params.output = reinterpret_cast<float4*>(layers.beauty);
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(SimpleStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(SimpleStrategyParams), &sbt, width,
                                height, 1));
    }

   private:
    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm