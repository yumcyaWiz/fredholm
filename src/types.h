#pragma once

#include <any>
#include <cstdint>

#include "cuda_util.h"

namespace fredholm
{

enum class AOVType
{
    FINAL = 0,
    DENOISED,
    BEAUTY,
    POSITION,
    NORMAL,
    ALBEDO,
    N_AOV_TYPES
};

class RenderLayers
{
   public:
    RenderLayers(uint32_t width, uint32_t height, bool use_gl_interop)
    {
        init_render_layers(width, height, use_gl_interop);
    }

    const CUDABuffer<float4>& get_aov(const AOVType& type) const
    {
        switch (type)
        {
            case AOVType::FINAL:
                return *final;
            case AOVType::DENOISED:
                return *denoised;
            case AOVType::BEAUTY:
                return *beauty;
            case AOVType::POSITION:
                return *position;
            case AOVType::NORMAL:
                return *normal;
            case AOVType::ALBEDO:
                return *albedo;
            default:
                throw std::runtime_error("unknown AOV type");
        }
    }

    void clear_render_layers()
    {
        final->clear();
        denoised->clear();
        beauty->clear();
        position->clear();
        normal->clear();
        albedo->clear();
    }

    void render_options_callback()
    {
        // TODO: implement
    }

    void render_strategy_callback()
    {
        // TODO: implement
    }

   private:
    void init_render_layers(uint32_t width, uint32_t height,
                            bool use_gl_interop)
    {
        final = std::make_unique<CUDABuffer<float4>>(width * height,
                                                     use_gl_interop);
        denoised = std::make_unique<CUDABuffer<float4>>(width * height,
                                                        use_gl_interop);
        beauty = std::make_unique<CUDABuffer<float4>>(width * height,
                                                      use_gl_interop);
        position = std::make_unique<CUDABuffer<float4>>(width * height,
                                                        use_gl_interop);
        normal = std::make_unique<CUDABuffer<float4>>(width * height,
                                                      use_gl_interop);
        albedo = std::make_unique<CUDABuffer<float4>>(width * height,
                                                      use_gl_interop);
    }

    std::unique_ptr<CUDABuffer<float4>> final = nullptr;
    std::unique_ptr<CUDABuffer<float4>> denoised = nullptr;
    std::unique_ptr<CUDABuffer<float4>> beauty = nullptr;
    std::unique_ptr<CUDABuffer<float4>> position = nullptr;
    std::unique_ptr<CUDABuffer<float4>> normal = nullptr;
    std::unique_ptr<CUDABuffer<float4>> albedo = nullptr;
};

enum class RenderOptionNames
{
    RESOLUTION,
    USE_GL_INTEROP,
    N_SAMPLES,
    N_SPP,
    MAX_DEPTH,
};

class RenderOptions
{
   private:
    std::unordered_map<RenderOptionNames, std::any> options;
    using ObserverCallback = std::function<void(const RenderOptionNames& name,
                                                const RenderOptions& options)>;
    std::vector<ObserverCallback> observers;

    RenderOptions() { init_options(); }

    void init_options()
    {
        options[RenderOptionNames::RESOLUTION] = make_uint2(1280, 720);
        options[RenderOptionNames::USE_GL_INTEROP] = false;
        options[RenderOptionNames::N_SAMPLES] = static_cast<uint32_t>(1);
        options[RenderOptionNames::N_SPP] = static_cast<uint32_t>(1);
        options[RenderOptionNames::MAX_DEPTH] = static_cast<uint32_t>(10);
    }

   public:
    static RenderOptions& get_instance()
    {
        static RenderOptions instance;
        return instance;
    }

    template <typename T>
    T get_option(const RenderOptionNames& name) const
    {
        if (options.find(name) == options.end())
        {
            throw std::runtime_error("Unknown option name");
        }
        return std::any_cast<T>(options.at(name));
    }

    template <typename T>
    void set_option(const RenderOptionNames& name, const T& value)
    {
        if (options.find(name) == options.end())
        {
            throw std::runtime_error("Unknown option name");
        }
        options[name] = value;

        notify_observers(name);
    }

    void register_observer(const ObserverCallback& callback)
    {
        observers.push_back(callback);
    }

    void notify_observers(const RenderOptionNames& name)
    {
        for (const auto& observer : observers) { observer(name, *this); }
    }
};

}  // namespace fredholm