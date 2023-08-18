#pragma once
#include <optix.h>

#include "camera.h"
#include "cuda_util.h"
#include "optix_util.h"
#include "render_strategy/render_strategy.h"
#include "scene.h"
#include "shared.h"
#include "stb_image_write.h"
#include "util.h"

namespace fredholm
{

inline void write_image(const std::filesystem::path& filepath,
                        const uint32_t width, const uint32_t height,
                        const float4* image)
{
    // convert float4 to uchar4
    std::vector<uchar4> image_c4(width * height);
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            const int idx = i + width * j;
            const float4 v = image[idx];
            image_c4[idx].x = static_cast<unsigned char>(
                std::clamp(255.0f * v.x, 0.0f, 255.0f));
            image_c4[idx].y = static_cast<unsigned char>(
                std::clamp(255.0f * v.y, 0.0f, 255.0f));
            image_c4[idx].z = static_cast<unsigned char>(
                std::clamp(255.0f * v.z, 0.0f, 255.0f));
            image_c4[idx].w = 255;
        }
    }

    // save image
    stbi_write_png(filepath.c_str(), width, height, 4, image_c4.data(),
                   sizeof(uchar4) * width);
}

class Renderer
{
   public:
    Renderer(uint32_t width, uint32_t height) : width(width), height(height)
    {
        cuda_check(cuMemAlloc(&beauty, width * height * sizeof(float4)));
        cuda_check(cuMemsetD32(beauty, 0, width * height));
    }

    ~Renderer()
    {
        if (beauty != 0)
        {
            cuda_check(cuMemFree(beauty));
            beauty = 0;
        }

        cuda_check(cuMemFree(sbt_record_set.raygen_records));
        cuda_check(cuMemFree(sbt_record_set.miss_records));
        cuda_check(cuMemFree(sbt_record_set.hitgroup_records));
    }

    void set_render_strategy(RenderStrategy* strategy)
    {
        m_render_strategy = strategy;

        sbt_record_set = optix_create_sbt_records(
            m_render_strategy->get_program_group_sets());
        sbt = optix_create_sbt(sbt_record_set);
    }

    // TODO: rendere should manage camera and scene?
    void render(const Camera& camera, const SceneDevice& scene)
    {
        if (m_render_strategy)
        {
            m_render_strategy->render(width, height, camera, scene,
                                      scene.get_ias_handle(), sbt, beauty);
        }
    }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }

    void save_image(const std::filesystem::path& filepath) const
    {
        std::vector<float4> beauty_h(width * height);
        cuda_check(cuMemcpyDtoH(beauty_h.data(), beauty,
                                width * height * sizeof(float4)));
        write_image(filepath, width, height, beauty_h.data());
    }

   private:
    uint32_t width = 0;
    uint32_t height = 0;

    CUdeviceptr beauty = 0;

    // TODO: this could be placed in render strategy?
    SbtRecordSet sbt_record_set;
    OptixShaderBindingTable sbt;

    RenderStrategy* m_render_strategy = nullptr;
};

}  // namespace fredholm