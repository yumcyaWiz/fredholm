#pragma once
#include <algorithm>
#include <filesystem>
#include <format>
#include <vector>

#include "helper_math.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace fredholm
{

class ImageLoader
{
   public:
    static std::vector<uchar4> load_ldr_image(
        const std::filesystem::path& filepath, uint32_t& width,
        uint32_t& height)
    {
        int w, h, c;
        stbi_set_flip_vertically_on_load(false);
        unsigned char* img =
            stbi_load(filepath.generic_string().c_str(), &w, &h, &c, STBI_rgb_alpha);
        if (img == nullptr)
        {
            throw std::runtime_error(std::format("failed to load texture: {}",
                                                 filepath.generic_string()));
        }

        width = w;
        height = h;
        std::vector<uchar4> ret(w * h);
        for (int i = 0; i < width * height; ++i)
        {
            ret[i] = make_uchar4(img[4 * i], img[4 * i + 1], img[4 * i + 2],
                                 img[4 * i + 3]);
        }
        stbi_image_free(img);

        return ret;
    }

    static std::vector<float3> load_hdr_image(
        const std::filesystem::path& filepath, uint32_t& width,
        uint32_t& height)
    {
        int w, h, c;
        stbi_set_flip_vertically_on_load(false);
        float* img = stbi_loadf(filepath.generic_string().c_str(), &w, &h, &c, STBI_rgb);
        if (img == nullptr)
        {
            throw std::runtime_error(std::format("failed to load texture: {}",
                                                 filepath.generic_string()));
        }

        width = w;
        height = h;
        std::vector<float3> ret(w * h);
        for (int i = 0; i < width * height; ++i)
        {
            ret[i] = make_float3(img[3 * i], img[3 * i + 1], img[3 * i + 2]);
        }
        stbi_image_free(img);

        return ret;
    }
};

class ImageWriter
{
   public:
    template <typename T>
    static void write_ldr_image(const std::filesystem::path& filepath,
                                const uint32_t width, const uint32_t height,
                                const T* image);

    template <>
    void write_ldr_image(const std::filesystem::path& filepath,
                         const uint32_t width, const uint32_t height,
                         const float3* image)
    {
        // convert float3 to uchar3
        std::vector<uchar3> image_c3(width * height);
        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                const int idx = i + width * j;
                const float3 v = image[idx];
                image_c3[idx].x = static_cast<unsigned char>(
                    std::clamp(255.0f * v.x, 0.0f, 255.0f));
                image_c3[idx].y = static_cast<unsigned char>(
                    std::clamp(255.0f * v.y, 0.0f, 255.0f));
                image_c3[idx].z = static_cast<unsigned char>(
                    std::clamp(255.0f * v.z, 0.0f, 255.0f));
            }
        }

        // save image
        stbi_write_png(filepath.generic_string().c_str(), width, height, 3, image_c3.data(),
                       sizeof(uchar3) * width);
    }

    template <>
    void write_ldr_image(const std::filesystem::path& filepath,
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
        stbi_write_png(filepath.generic_string().c_str(), width, height, 4, image_c4.data(),
                       sizeof(uchar4) * width);
    }
};

}  // namespace fredholm