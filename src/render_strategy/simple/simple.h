#pragma once
#include "cuda_util.h"
#include "imgui.h"
#include "render_strategy/render_strategy.h"
#include "simple_shared.h"

namespace fredholm
{

class SimpleStrategy : public RenderStrategy
{
   public:
    SimpleStrategy(const OptixDeviceContext& context, bool debug)
        : RenderStrategy(context, debug)
    {
    }

    ~SimpleStrategy()
    {
        if (params_buffer != 0)
        {
            cuda_check(cuMemFree(params_buffer));
            params_buffer = 0;
        }
    }

    void run_imgui() override
    {
        RenderStrategy::run_imgui();

        ImGui::Combo(
            "mode", &output_mode,
            "Position\0Normal\0Texcoord\0Barycentric\0Clearcoat\0Specu"
            "lar\0Specular "
            "Roughness\0Metalness\0Transmission\0Diffuse\0Emission\0\0");
    }

    void render(const Camera& camera, const DirectionalLight& directional_light,
                const SceneDevice& scene, const RenderLayers& render_layers,
                const OptixTraversableHandle& ias_handle) override
    {
        const RenderOptions& options = RenderOptions::get_instance();

        SimpleStrategyParams params;
        params.width =
            options.get_option<uint2>(RenderOptionNames::RESOLUTION).x;
        params.height =
            options.get_option<uint2>(RenderOptionNames::RESOLUTION).y;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene, directional_light);
        params.output_mode = output_mode;
        params.ias_handle = ias_handle;
        params.output = reinterpret_cast<float4*>(
            render_layers.get_aov(AOVType::BEAUTY).get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(SimpleStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(SimpleStrategyParams), &sbt,
                                params.width, params.height, 1));
        cuda_check(cuCtxSynchronize());
    }

   private:
    void init_render_strategy() override
    {
        m_module =
            optix_create_module(context, "optixir/simple/simple.cu.o", debug);

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

    int output_mode = 0;

    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm