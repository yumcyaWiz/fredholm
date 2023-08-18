#pragma once
#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "io.h"
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
    Renderer(uint32_t width, uint32_t height) : width(width), height(height)
    {
        cuda_check(cuMemAlloc(&beauty, width * height * sizeof(float4)));
        cuda_check(cuMemsetD32(beauty, 0, width * height));
    }

    ~Renderer()
    {
        if (beauty != 0)
        {
            cuda_check(cuMemFree(beauty));
            beauty = 0;
        }

        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));
    }

    void set_render_strategy(RenderStrategy* strategy)
    {
        m_render_strategy = strategy;

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    // TODO: rendere should manage camera and scene?
    void render(const Camera& camera, const SceneDevice& scene)
    {
        if (m_render_strategy)
        {
            m_render_strategy->render(width, height, camera, scene,
                                      scene.get_ias_handle(), sbt, beauty);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        std::vector<float4> beauty_h(width * height);
        cuda_check(cuMemcpyDtoH(beauty_h.data(), beauty,
                                width * height * sizeof(float4)));
        write_image(filepath, width, height, beauty_h.data());
    }

   private:
    uint32_t width = 0;
    uint32_t height = 0;

    CUdeviceptr beauty = 0;

    // TODO: this could be placed in render strategy?
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm