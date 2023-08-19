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

enum class RenderStrategyType
{
    HELLO = 0,
    SIMPLE,
    PT,
    N_RENDER_STRATEGIES,
};

class RenderStrategyFactory
{
   public:
    static std::unique_ptr<RenderStrategy> create(
        const RenderStrategyType& type, const RenderOptions& options,
        const OptixDeviceContext& context, bool debug)
    {
        switch (type)
        {
            case RenderStrategyType::HELLO:
                return std::make_unique<HelloStrategy>(options, context, debug);
            case RenderStrategyType::SIMPLE:
                return std::make_unique<SimpleStrategy>(options, context,
                                                        debug);
            case RenderStrategyType::PT:
                return std::make_unique<PtStrategy>(options, context, debug);
            default:
                throw std::runtime_error("unknown render strategy type");
        }
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

    template <typename T>
    void set_option(const std::string& name, const T& value)
    {
        m_render_strategy->set_option<T>(name, value);
    }

    RenderStrategyType get_render_strategy_type() const
    {
        return m_render_strategy_type;
    }

    void set_render_strategy(const RenderStrategyType& type,
                             const RenderOptions& options)
    {
        m_render_strategy =
            RenderStrategyFactory::create(type, options, context, debug);

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    void runImGui()
    {
        if (m_render_strategy) { m_render_strategy->runImGui(); }
    }

    void clear_render()
    {
        if (m_render_strategy) { m_render_strategy->clear_render(); }
    }

    // TODO: rendere should manage camera and scene?
    void render(const Camera& camera, const SceneDevice& scene)
    {
        m_render_strategy->render(camera, scene, scene.get_ias_handle(), sbt);
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        const uint2 resolution =
            m_render_strategy->get_option<uint2>("resolution");

        std::vector<float4> beauty_h(resolution.x * resolution.y);
        m_render_strategy->get_aov("beauty").copy_d_to_h(beauty_h.data());
        write_image(filepath, resolution.x, resolution.y, beauty_h.data());
    }

   private:
    OptixDeviceContext context = nullptr;
    bool debug = false;

    // TODO: this could be placed in render strategy?
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategyType m_render_strategy_type =
        RenderStrategyType::N_RENDER_STRATEGIES;
    std::unique_ptr<RenderStrategy> m_render_strategy = nullptr;
};

}  // namespace fredholm