#pragma once
#include <optix.h>

#include <memory>

#include "camera.h"
#include "cuda_util.h"
#include "denoiser.h"
#include "gl_util.h"
#include "image_io.h"
#include "imgui.h"
#include "optix_util.h"
#include "post_process/post_process.h"
#include "render_strategy/hello/hello.h"
#include "render_strategy/pt/pt.h"
#include "render_strategy/ptmis/ptmis.h"
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
    PTMIS,
    N_RENDER_STRATEGIES,
};

class RenderStrategyFactory
{
   public:
    static std::unique_ptr<RenderStrategy> create(
        const RenderStrategyType& type, const OptixDeviceContext& context,
        bool debug, const RenderOptions& options)
    {
        switch (type)
        {
            case RenderStrategyType::HELLO:
            {
                auto strategy =
                    std::make_unique<HelloStrategy>(context, debug, options);
                strategy->init();
                return strategy;
                break;
            }
            case RenderStrategyType::SIMPLE:
            {
                auto strategy =
                    std::make_unique<SimpleStrategy>(context, debug, options);
                strategy->init();
                return strategy;
                break;
            }
            case RenderStrategyType::PT:
            {
                auto strategy =
                    std::make_unique<PtStrategy>(context, debug, options);
                strategy->init();
                return strategy;
                break;
            }
            case RenderStrategyType::PTMIS:
            {
                auto strategy =
                    std::make_unique<PTMISStrategy>(context, debug, options);
                strategy->init();
                return strategy;
                break;
            }
            default:
                throw std::runtime_error("unknown render strategy type");
        }
    }
};

// TODO: how to combine with RenderOptions?
struct RendererOptions
{
    bool use_denoiser = false;

    template <typename T>
    T get_option(const std::string& name) const;

   private:
    template <>
    bool get_option(const std::string& name) const
    {
        if (name == "use_denoiser") { return use_denoiser; }
        else { throw std::runtime_error("Unknown option name"); }
    }
};

class Renderer
{
   public:
    Renderer(const OptixDeviceContext& context, bool debug)
        : context(context), debug(debug)
    {
        init_post_process();
    }

    ~Renderer()
    {
        if (m_render_strategy) { m_render_strategy.reset(); }

        if (m_denoiser) { m_denoiser.reset(); }
        if (m_denoised) { m_denoised.reset(); }

        if (m_post_process) { m_post_process.reset(); }
        if (m_final) { m_final.reset(); }
    }

    const CUDABuffer<float4>& get_aov(const AOVType& type) const
    {
        switch (type)
        {
            case AOVType::FINAL:
                if (!m_final)
                {
                    throw std::runtime_error("final buffer not initialized");
                }
                return *m_final;
            case AOVType::DENOISED:
                if (!m_denoised)
                {
                    throw std::runtime_error("denoised buffer not initialized");
                }
                return *m_denoised;
            default:
            {
                if (!m_render_strategy)
                {
                    throw std::runtime_error("render strategy not set");
                }
                return m_render_strategy->get_aov(type);
            }
        }
    }

    // TODO: want to add RendererOptions here
    template <typename T>
    T get_option(const std::string& name) const
    {
        if (m_render_strategy)
        {
            return m_render_strategy->get_option<T>(name);
        }
        else { return RenderOptions().get_option<T>(name); }
    }

    template <typename T>
    void set_option(const std::string& name, const T& value)
    {
        m_render_strategy->set_option<T>(name, value);

        if (name == "resolution")
        {
            init_render_layers();
            init_denoiser();
        }
    }

    RenderStrategyType get_render_strategy_type() const
    {
        return m_render_strategy_type;
    }

    bool get_paused() const { return paused; }
    void set_paused(bool paused) { this->paused = paused; }

    void set_render_strategy(const RenderStrategyType& type,
                             const RenderOptions& options)
    {
        m_render_strategy =
            RenderStrategyFactory::create(type, context, debug, options);
        m_render_strategy_type = type;

        init_render_layers();
        init_denoiser();
    }

    void set_render_strategy(const RenderStrategyType& type)
    {
        set_render_strategy(type, m_render_strategy->get_options());
    }

    void run_imgui()
    {
        ImGui::Separator();

        if (ImGui::Button(paused ? "resume" : "pause")) { paused = !paused; }

        if (m_render_strategy) { m_render_strategy->run_imgui(); }

        if (m_post_process) { m_post_process->run_imgui(); }
    }

    void clear_render()
    {
        if (m_render_strategy) { m_render_strategy->clear_render(); }
    }

    // TODO: renderer should manage camera and scene?
    void render(const Camera& camera, const DirectionalLight& directional_light,
                const SceneDevice& scene)
    {
        if (paused) return;
        m_render_strategy->render(camera, directional_light, scene,
                                  scene.get_ias_handle());
        run_denoiser();
        run_post_process();
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        const uint2 resolution = get_option<uint2>("resolution");

        // copy image from device to host
        std::vector<float4> beauty_h(resolution.x * resolution.y);
        get_aov(AOVType::FINAL).copy_d_to_h(beauty_h.data());

        ImageWriter::write_ldr_image(filepath, resolution.x, resolution.y,
                                   beauty_h.data());
    }

   private:
    void init_render_layers()
    {
        const uint2 resolution = get_option<uint2>("resolution");
        const bool use_gl_interop = get_option<bool>("use_gl_interop");

        m_final = std::make_unique<CUDABuffer<float4>>(
            resolution.x * resolution.y, use_gl_interop);
        m_denoised = std::make_unique<CUDABuffer<float4>>(
            resolution.x * resolution.y, use_gl_interop);
    }

    void init_post_process()
    {
        m_post_process = std::make_unique<PostProcess>();
    }

    void init_denoiser()
    {
        const uint2 resolution = get_option<uint2>("resolution");
        m_denoiser =
            std::make_unique<Denoiser>(context, resolution.x, resolution.y);
    }

    void run_post_process()
    {
        const uint2 resolution = get_option<uint2>("resolution");
        if (m_post_process)
        {
            m_post_process->run(
                resolution.x, resolution.y,
                reinterpret_cast<float4*>(
                    get_aov(AOVType::DENOISED).get_device_ptr()),
                reinterpret_cast<float4*>(m_final->get_device_ptr()));
        }
    }

    void run_denoiser()
    {
        const uint2 resolution = get_option<uint2>("resolution");
        if (m_denoiser)
        {
            m_denoiser->denoise(get_aov(AOVType::BEAUTY).get_device_ptr(),
                                get_aov(AOVType::NORMAL).get_device_ptr(),
                                get_aov(AOVType::ALBEDO).get_device_ptr(),
                                m_denoised->get_device_ptr());
        }
    }

    OptixDeviceContext context = nullptr;
    bool debug = false;

    bool paused = false;
    RenderStrategyType m_render_strategy_type =
        RenderStrategyType::N_RENDER_STRATEGIES;
    std::unique_ptr<RenderStrategy> m_render_strategy = nullptr;

    std::unique_ptr<Denoiser> m_denoiser = nullptr;
    std::unique_ptr<CUDABuffer<float4>> m_denoised = nullptr;

    std::unique_ptr<PostProcess> m_post_process = nullptr;
    std::unique_ptr<CUDABuffer<float4>> m_final = nullptr;
};

}  // namespace fredholm