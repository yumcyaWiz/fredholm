#pragma once
#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "optix_util.h"
#include "render_strategy/render_strategy.h"
#include "scene.h"
#include "shared.h"
#include "util.h"

namespace fredholm
{

class Renderer
{
   public:
    Renderer() {}

    ~Renderer()
    {
        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));
    }

    void set_render_strategy(RenderStrategy* strategy)
    {
        m_render_strategy = strategy;

        // TODO: maybe this could be placed in render strategy?
        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    void render(uint32_t width, uint32_t height, const Camera& camera,
                const SceneDevice& scene, const CUdeviceptr& beauty)
    {
        if (m_render_strategy)
        {
            m_render_strategy->render(width, height, camera, scene,
                                      scene.get_ias_handle(), sbt, beauty);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

   private:
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm