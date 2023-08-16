#pragma once
#include "camera.h"
#include "optix_util.h"
#include "shared.h"

namespace fredholm
{

// TODO: add parameters specific to each strategy
// TODO: add GUI to change each parameters
class RenderStrategy
{
   public:
    RenderStrategy() = default;

    virtual ~RenderStrategy()
    {
        if (m_params_buffer != 0)
        {
            cuda_check(cuMemFree(m_params_buffer));
            m_params_buffer = 0;
        }

        if (m_pipeline != nullptr)
        {
            optix_check(optixPipelineDestroy(m_pipeline));
            m_pipeline = nullptr;
        }

        for (auto& raygen_program_group :
             m_program_group_sets.raygen_program_groups)
        {
            if (raygen_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(raygen_program_group));
                raygen_program_group = nullptr;
            }
        }
        for (auto& miss_program_group :
             m_program_group_sets.miss_program_groups)
        {
            if (miss_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(miss_program_group));
                miss_program_group = nullptr;
            }
        }
        for (auto& hitgroup_program_group :
             m_program_group_sets.hitgroup_program_groups)
        {
            if (hitgroup_program_group != nullptr)
            {
                optix_check(optixProgramGroupDestroy(hitgroup_program_group));
                hitgroup_program_group = nullptr;
            }
        }

        if (m_module != nullptr)
        {
            optix_check(optixModuleDestroy(m_module));
            m_module = nullptr;
        }
    }

    const ProgramGroupSet& get_program_group_sets() const
    {
        return m_program_group_sets;
    }

    virtual void render(uint32_t width, uint32_t height,
                        const CameraParams& camera, const SceneData& scene,
                        const OptixTraversableHandle& ias_handle,
                        const OptixShaderBindingTable& sbt,
                        const CUdeviceptr& beauty) = 0;

   protected:
    OptixModule m_module = nullptr;
    ProgramGroupSet m_program_group_sets = {};
    OptixPipeline m_pipeline = nullptr;

    CUdeviceptr m_params_buffer = 0;
};

}  // namespace fredholm