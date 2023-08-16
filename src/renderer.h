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
    Renderer(CUcontext cu_context)
    {
#ifdef NDEBUG
        constexpr bool debug = false;
#else
        constexpr bool debug = true;
#endif
        context = optix_create_context(cu_context, debug);
    }

    ~Renderer()
    {
        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));

        optix_check(optixDeviceContextDestroy(context));
    }

    // TODO: maybe optix context could be placed outside renderer?
    OptixDeviceContext get_optix_context() const { return context; }

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
            CameraParams camera_params;
            camera_params.transform =
                create_mat3x4_from_glm(camera.m_transform);
            camera_params.fov = camera.m_fov;
            camera_params.F = camera.m_F;
            camera_params.focus = camera.m_focus;

            m_render_strategy->render(width, height, camera_params,
                                      scene.get_scene_data(),
                                      scene.get_ias_handle(), sbt, beauty);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

   private:
    OptixDeviceContext context;

    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm