#pragma once
#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "gl_util.h"
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
   private:
    struct RenderOptions
    {
        uint32_t width = 512;
        uint32_t height = 512;

        bool use_gl_interop = false;
    };

   public:
    Renderer() { init_render_layers(); }

    ~Renderer()
    {
        if (beauty) { beauty.reset(); }

        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));
    }

    template <typename T>
    T get_option(const std::string& name) const
    {
        if (name == "width") { return options.width; }
        else if (name == "height") { return options.height; }
        else if (name == "use_gl_interop") { return options.use_gl_interop; }
        else { throw std::runtime_error("Unknown option name"); }
    }

    template <typename T>
    void set_option(const std::string& name, const T& value)
    {
        if (name == "width")
        {
            options.width = value;
            init_render_layers();
        }
        else if (name == "height")
        {
            options.height = value;
            init_render_layers();
        }
        else if (name == "use_gl_interop")
        {
            options.use_gl_interop = value;
            init_render_layers();
        }
        else { throw std::runtime_error("Unknown option name"); }
    }

    const CUDABuffer<float4>& get_aov(const std::string& name) const
    {
        if (name == "beauty") { return *beauty; }
        else { throw std::runtime_error("Unknown AOV name"); }
    }

    void init_render_layers()
    {
        beauty = std::make_unique<CUDABuffer<float4>>(
            options.width * options.height, options.use_gl_interop);
    }

    void set_render_strategy(RenderStrategy* strategy)
    {
        m_render_strategy = strategy;

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    void runImGui() const
    {
        if (m_render_strategy) { m_render_strategy->runImGui(); }
    }

    // TODO: rendere should manage camera and scene?
    void render(const Camera& camera, const SceneDevice& scene)
    {
        if (m_render_strategy)
        {
            RenderLayers layers = {};
            layers.beauty = beauty->get_device_ptr();

            m_render_strategy->render(options.width, options.height, camera,
                                      scene, scene.get_ias_handle(), sbt,
                                      layers);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        std::vector<float4> beauty_h(options.width * options.height);
        beauty->copy_d_to_h(beauty_h.data());
        write_image(filepath, options.width, options.height, beauty_h.data());
    }

   private:
    RenderOptions options;

    std::unique_ptr<CUDABuffer<float4>> beauty = nullptr;

    // TODO: this could be placed in render strategy?
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm