#pragma once

#include "cuda_util.h"
#include "hello_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class HelloStrategy : public RenderStrategy
{
   public:
    HelloStrategy(const OptixDeviceContext& context, bool debug,
                  const RenderOptions& options)
        : RenderStrategy(context, debug, options)
    {
    }

    ~HelloStrategy()
    {
        if (params_buffer != 0)
        {
            cuda_check(cuMemFree(params_buffer));
            params_buffer = 0;
        }
    }

    void render(const Camera& camera, const DirectionalLight& directional_light,
                const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle) override
    {
        HelloStrategyParams params;
        params.width = options.resolution.x;
        params.height = options.resolution.y;
        params.output = reinterpret_cast<float4*>(beauty->get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(HelloStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(HelloStrategyParams), &sbt,
                                options.resolution.x, options.resolution.y, 1));
    }

   private:
    void init_render_strategy() override
    {
        m_module =
            optix_create_module(context, "optixir/hello/hello.cu.o", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "rg", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 1, 2, debug);

        cuda_check(cuMemAlloc(&params_buffer, sizeof(HelloStrategyParams)));
    }

    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm