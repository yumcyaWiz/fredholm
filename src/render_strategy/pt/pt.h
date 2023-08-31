#pragma once
#include <algorithm>

#include "cuda_util.h"
#include "imgui.h"
#include "render_strategy/pt/pt_shared.h"
#include "render_strategy/render_strategy.h"

namespace fredholm
{

class PtStrategy : public RenderStrategy
{
   public:
    PtStrategy(const OptixDeviceContext& context, bool debug)
        : RenderStrategy(context, debug)
    {
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
        RenderStrategy::clear_render();
        sample_count = 0;
    }

    void run_imgui() override
    {
        RenderStrategy::run_imgui();

        RenderOptions& options = RenderOptions::get_instance();

        uint32_t n_samples =
            options.get_option<uint32_t>(RenderOptionNames::N_SAMPLES);
        ImGui::ProgressBar(
            static_cast<float>(sample_count) / static_cast<float>(n_samples),
            ImVec2(0.0f, 0.0f));
        ImGui::SameLine();
        ImGui::Text("%d/%d spp", sample_count, n_samples);
        if (ImGui::Button("clear")) { clear_render(); }

        uint32_t n_spp = options.get_option<uint32_t>(RenderOptionNames::N_SPP);
        if (ImGui::InputInt("n_spp", reinterpret_cast<int*>(&n_spp)))
        {
            options.set_option<uint32_t>(RenderOptionNames::N_SPP,
                                         std::min(n_spp, n_samples));
            // clear_render();
        }

        if (ImGui::InputInt("n_samples", reinterpret_cast<int*>(&n_samples)))
        {
            options.set_option<uint32_t>(RenderOptionNames::N_SAMPLES,
                                         std::min(n_samples, 1u));
            // clear_render();
        }

        uint32_t max_depth =
            options.get_option<uint32_t>(RenderOptionNames::MAX_DEPTH);
        if (ImGui::InputInt("max_depth", reinterpret_cast<int*>(&max_depth)))
        {
            options.set_option<uint32_t>(RenderOptionNames::MAX_DEPTH,
                                         std::max(max_depth, 1u));
            // clear_render();
        }
    }

    void render(const Camera& camera, const DirectionalLight& directional_light,
                const SceneDevice& scene, const RenderLayers& render_layers,
                const OptixTraversableHandle& ias_handle) override
    {
        const RenderOptions& options = RenderOptions::get_instance();

        if (sample_count >=
            options.get_option<uint32_t>(RenderOptionNames::N_SAMPLES))
            return;

        PtStrategyParams params;
        params.width =
            options.get_option<uint2>(RenderOptionNames::RESOLUTION).x;
        params.height =
            options.get_option<uint2>(RenderOptionNames::RESOLUTION).y;
        params.camera = get_camera_params(camera);
        params.scene = get_scene_data(scene, directional_light);
        params.ias_handle = ias_handle;
        params.n_samples =
            options.get_option<uint32_t>(RenderOptionNames::N_SPP);
        params.max_depth =
            options.get_option<uint32_t>(RenderOptionNames::MAX_DEPTH);
        params.seed = seed;
        params.beauty = reinterpret_cast<float4*>(
            render_layers.get_aov(AOVType::BEAUTY).get_device_ptr());
        params.position = reinterpret_cast<float4*>(
            render_layers.get_aov(AOVType::POSITION).get_device_ptr());
        params.normal = reinterpret_cast<float4*>(
            render_layers.get_aov(AOVType::NORMAL).get_device_ptr());
        params.albedo = reinterpret_cast<float4*>(
            render_layers.get_aov(AOVType::ALBEDO).get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(PtStrategyParams)));

        optix_check(optixLaunch(m_pipeline, 0, params_buffer,
                                sizeof(PtStrategyParams), &sbt, params.width,
                                params.height, 1));
        cuda_check(cuCtxSynchronize());

        sample_count += params.n_samples;
    }

   private:
    void init_render_strategy() override
    {
        m_module = optix_create_module(context, "optixir/pt/pt.cu.o", debug);

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

    uint32_t seed = 1;
    uint32_t sample_count = 0;

    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm