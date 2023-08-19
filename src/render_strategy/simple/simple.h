#pragma once
#include "cuda_util.h"
#include "render_strategy/render_strategy.h"
#include "simple_shared.h"

namespace fredholm
{

class SimpleStrategy : public RenderStrategy
{
   public:
    SimpleStrategy(const RenderOptions& options,
                   const OptixDeviceContext& context, bool debug = false)
        : RenderStrategy(options)
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

    void render(const Camera& camera, const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle,
                const OptixShaderBindingTable& sbt) override
    {
        SimpleStrategyParams params;
        params.width = options.width;
        params.height = options.height;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene);
        params.ias_handle = ias_handle;
        params.output = reinterpret_cast<float4*>(beauty->get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(SimpleStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(SimpleStrategyParams), &sbt,
                                options.width, options.height, 1));
    }

   private:
    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm