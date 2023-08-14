#include <optix.h>

#include "cuda_util.h"
#include "renderer.h"

int main()
{
    fredholm::cuda_check(cuInit(0));
    fredholm::CUDADevice device(0);

    optixInit();

    fredholm::Renderer renderer(device.get_context());

    constexpr uint32_t width = 512;
    constexpr uint32_t height = 512;
    fredholm::CUDABuffer<float4> beauty(width * height);
    renderer.render(width, height, beauty.get_device_ptr());

    return 0;
}