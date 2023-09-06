#pragma once

#include <any>
#include <cstdint>
#include <memory>
#include <queue>

#include "cuda_util.h"

namespace fredholm
{

enum class RenderCommandType
{
    RENDER = 0,
    CLEAR,
    N_RENDER_COMMAND_TYPES,
};

class RenderCommandQueue
{
   private:
    std::queue<RenderCommandType> commands;

   public:
    bool is_empty() const { return commands.empty(); }

    void enqueue(const RenderCommandType& command) { commands.push(command); }

    RenderCommandType dequeue()
    {
        if (is_empty()) { throw std::runtime_error("queue is empty"); }
        const auto command = commands.front();
        commands.pop();
        return command;
    }

    void clear() { commands = std::queue<RenderCommandType>(); }
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
        options[RenderOptionNames::N_SAMPLES] = static_cast<uint32_t>(512);
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

        RenderOptions::get_instance().register_observer(
            [&](const RenderOptionNames& name, const RenderOptions& options)
            {
                if (name == RenderOptionNames::RESOLUTION ||
                    name == RenderOptionNames::USE_GL_INTEROP)
                {
                    init_render_layers(
                        options.get_option<uint2>(RenderOptionNames::RESOLUTION)
                            .x,
                        options.get_option<uint2>(RenderOptionNames::RESOLUTION)
                            .y,
                        options.get_option<bool>(
                            RenderOptionNames::USE_GL_INTEROP));
                }
                else if (name == RenderOptionNames::N_SPP ||
                         name == RenderOptionNames::N_SAMPLES ||
                         name == RenderOptionNames::MAX_DEPTH)
                {
                    clear_render_layers();
                }
            });
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

}  // namespace fredholm