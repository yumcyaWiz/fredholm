#include "cuda_util.h"
#include "renderer.h"

int main()
{
    fredholm::cuda_check(cuInit(0));
    fredholm::CUDADevice device(0);

    optixInit();

    fredholm::Renderer renderer(device.get_context());

    return 0;
}