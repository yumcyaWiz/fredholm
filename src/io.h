#pragma once
#include <algorithm>
#include <filesystem>
#include <vector>

#include "helper_math.h"
#include "stb_image_write.h"

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

}  // namespace fredholm