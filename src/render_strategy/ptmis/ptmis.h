#pragma once
#include <algorithm>

#include "cuda_util.h"
#include "imgui.h"
#include "render_strategy/ptmis/ptmis_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class PTMISStrategy : public RenderStrategy
{
   public:
    PTMISStrategy(const OptixDeviceContext& context, bool debug,
                  const RenderOptions& options)
        : RenderStrategy(context, debug, options)
    {
    }

    ~PTMISStrategy()
    {
        if (params_buffer != 0)
        {
            cuda_check(cuMemFree(params_buffer));
            params_buffer = 0;
        }
    }

    void clear_render() override
    {
        RenderStrategy::clear_render();
        sample_count = 0;
    }

    void run_imgui() override
    {
        RenderStrategy::run_imgui();

        ImGui::ProgressBar(static_cast<float>(sample_count) /
                               static_cast<float>(options.n_samples),
                           ImVec2(0.0f, 0.0f));
        ImGui::SameLine();
        ImGui::Text("%d/%d spp", sample_count, options.n_samples);
        if (ImGui::Button("clear")) { clear_render(); }

        if (ImGui::InputInt("n_spp", reinterpret_cast<int*>(&options.n_spp)))
        {
            options.n_spp = std::min(options.n_spp, options.n_samples);
            clear_render();
        }
        if (ImGui::InputInt("n_samples",
                            reinterpret_cast<int*>(&options.n_samples)))
        {
            options.n_samples = std::max(options.n_samples, 1u);
            clear_render();
        }
        if (ImGui::InputInt("max_depth",
                            reinterpret_cast<int*>(&options.max_depth)))
        {
            options.max_depth = std::max(options.max_depth, 1u);
            clear_render();
        }
    }

    void render(const Camera& camera, const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle) override
    {
        if (sample_count >= options.n_samples) return;

        PTMISStrategyParams params;
        params.width = options.resolution.x;
        params.height = options.resolution.y;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene);
        params.ias_handle = ias_handle;
        params.n_samples = options.n_spp;
        params.max_depth = options.max_depth;
        params.seed = seed;
        params.sample_count = sample_count;
        params.output = reinterpret_cast<float4*>(beauty->get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(PTMISStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(PTMISStrategyParams), &sbt,
                                options.resolution.x, options.resolution.y, 1));
        sample_count += options.n_spp;
    }

   private:
    void init_render_strategy() override
    {
        m_module = optix_create_module(context, "ptmis.ptx", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "radiance", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "radiance", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "shadow", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "shadow", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "light", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "light", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 2, 2, debug);

        cuda_check(cuMemAlloc(&params_buffer, sizeof(PTMISStrategyParams)));
    }

    uint32_t seed = 1;
    uint32_t sample_count = 0;

    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm