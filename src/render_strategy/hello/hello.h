#pragma once

#include "cuda_util.h"
#include "hello_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class HelloStrategy : public RenderStrategy
{
   public:
    HelloStrategy(const OptixDeviceContext& context, bool debug = false)
    {
        m_module = optix_create_module(context, "hello.ptx", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "rg", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 1, 2, debug);

        cuda_check(cuMemAlloc(&m_params_buffer, sizeof(HelloStrategyParams)));
    }

    void render(uint32_t width, uint32_t height, const Camera& camera,
                const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle,
                const OptixShaderBindingTable& sbt,
                const CUdeviceptr& beauty) override
    {
        HelloStrategyParams params;
        params.width = width;
        params.height = height;
        params.output = reinterpret_cast<float4*>(beauty);

        cuda_check(cuMemcpyHtoD(m_params_buffer, &params,
                                sizeof(HelloStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, m_params_buffer,
                                sizeof(HelloStrategyParams), &sbt, width,
                                height, 1));
    }
};

}  // namespace fredholm