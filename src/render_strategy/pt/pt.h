#pragma once
#include "cuda_util.h"
#include "imgui.h"
#include "render_strategy/pt/pt_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class PtStrategy : public RenderStrategy
{
   public:
    PtStrategy(const RenderOptions& options, const OptixDeviceContext& context,
               bool debug = false)
        : RenderStrategy(options)
    {
        m_module = optix_create_module(context, "pt.ptx", debug);

        const std::vector<ProgramGroupEntry> program_group_entries = {
            {OPTIX_PROGRAM_GROUP_KIND_RAYGEN, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_MISS, "", m_module},
            {OPTIX_PROGRAM_GROUP_KIND_HITGROUP, "", m_module},
        };

        m_program_group_sets =
            optix_create_program_group(context, program_group_entries);

        m_pipeline =
            optix_create_pipeline(context, m_program_group_sets, 1, 2, debug);

        cuda_check(cuMemAlloc(&params_buffer, sizeof(PtStrategyParams)));
    }

    ~PtStrategy()
    {
        if (params_buffer != 0)
        {
            cuda_check(cuMemFree(params_buffer));
            params_buffer = 0;
        }
    }

    void clear_render() override
    {
        init_render_layers();
        sample_count = 0;
    }

    void runImGui() override
    {
        ImGui::ProgressBar(static_cast<float>(sample_count) / n_samples);
        ImGui::Text("%d / %d spp", sample_count, n_samples);
        if (ImGui::InputInt("n_samples", reinterpret_cast<int*>(&n_samples)))
        {
            clear_render();
        }
        if (ImGui::SliderInt("max_depth", reinterpret_cast<int*>(&max_depth), 1,
                             100))
        {
            clear_render();
        }
    }

    void render(const Camera& camera, const SceneDevice& scene,
                const OptixTraversableHandle& ias_handle,
                const OptixShaderBindingTable& sbt) override
    {
        if (sample_count >= n_samples) return;

        PtStrategyParams params;
        params.width = options.width;
        params.height = options.height;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene);
        params.ias_handle = ias_handle;
        params.n_samples = 1;
        params.max_depth = max_depth;
        params.seed = seed;
        params.sample_count = sample_count;
        params.output = reinterpret_cast<float4*>(beauty->get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(PtStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(PtStrategyParams), &sbt, options.width,
                                options.height, 1));
        sample_count += 1;
    }

   private:
    uint32_t n_samples = 512;
    uint32_t max_depth = 100;
    uint32_t seed = 1;
    uint32_t sample_count = 0;

    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm