#include <optix.h>

#include "cuda_util.h"
#include "renderer.h"
#include "scene.h"
#include "stb_image_write.h"

void save_png(const std::filesystem::path& filepath, const uint32_t width,
              const uint32_t height, const float4* image)
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

int main()
{
    fredholm::SceneGraph scene;
    scene.load_obj("CornellBox-Original.obj");
    scene.print_tree();

    const auto compiled_scene = scene.compile();
    printf("%d\n", compiled_scene.geometry_nodes.size());
    printf("%d\n", compiled_scene.geometry_transforms.size());

    // fredholm::cuda_check(cuInit(0));
    // fredholm::CUDADevice device(0);

    // optixInit();

    // fredholm::Renderer renderer(device.get_context());

    // constexpr uint32_t width = 512;
    // constexpr uint32_t height = 512;
    // fredholm::CUDABuffer<float4> beauty_d(width * height);
    // renderer.render(width, height, beauty_d.get_device_ptr());
    // renderer.synchronize();

    // std::vector<float4> beauty(width * height);
    // beauty_d.copy_d_to_h(beauty.data());

    // save_png("output.png", width, height, beauty.data());

    return 0;
}