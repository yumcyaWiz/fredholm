#pragma once
#include <optix.h>

#include <memory>

#include "camera.h"
#include "cuda_util.h"
#include "gl_util.h"
#include "imgui.h"
#include "io.h"
#include "optix_util.h"
#include "render_strategy/hello/hello.h"
#include "render_strategy/pt/pt.h"
#include "render_strategy/simple/simple.h"
#include "scene.h"
#include "shared.h"
#include "util.h"

namespace fredholm
{

class RenderStrategyFactory
{
   public:
    static std::unique_ptr<RenderStrategy> create(
        const std::string& name, const RenderOptions& options,
        const OptixDeviceContext& context, bool debug)
    {
        if (name == "hello")
        {
            return std::make_unique<HelloStrategy>(options, context, debug);
        }
        else if (name == "simple")
        {
            return std::make_unique<SimpleStrategy>(options, context, debug);
        }
        else if (name == "pt")
        {
            return std::make_unique<PtStrategy>(options, context, debug);
        }
        else { throw std::runtime_error("Unknown render strategy: " + name); }
    }
};

class Renderer
{
   public:
    Renderer(const OptixDeviceContext& context, bool debug)
        : context(context), debug(debug)
    {
    }

    ~Renderer()
    {
        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));

        if (m_render_strategy) { m_render_strategy.reset(); }
    }

    const CUDABuffer<float4>& get_aov(const std::string& name) const
    {
        return m_render_strategy->get_aov(name);
    }

    template <typename T>
    T get_option(const std::string& name) const
    {
        return m_render_strategy->get_option<T>(name);
    }

    void set_render_strategy(const std::string& name,
                             const RenderOptions& options)
    {
        m_render_strategy =
            RenderStrategyFactory::create(name, options, context, debug);

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    void runImGui()
    {
        if (m_render_strategy) { m_render_strategy->runImGui(); }
    }

    // TODO: rendere should manage camera and scene?
    void render(const Camera& camera, const SceneDevice& scene)
    {
        m_render_strategy->render(camera, scene, scene.get_ias_handle(), sbt);
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        const uint32_t width = m_render_strategy->get_option<uint32_t>("width");
        const uint32_t height =
            m_render_strategy->get_option<uint32_t>("height");

        std::vector<float4> beauty_h(width * height);
        m_render_strategy->get_aov("beauty").copy_d_to_h(beauty_h.data());
        write_image(filepath, width, height, beauty_h.data());
    }

   private:
    OptixDeviceContext context = nullptr;
    bool debug = false;

    // TODO: this could be placed in render strategy?
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    std::unique_ptr<RenderStrategy> m_render_strategy = nullptr;
};

}  // namespace fredholm