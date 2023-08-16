#include <optix.h>

#include "render_strategy/hello/hello_shared.h"

using namespace fredholm;

extern "C"
{
    __constant__ HelloStrategyParams params;
}

struct RayPayload
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
};

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;

    params.output[image_idx] =
        make_float4(idx.x / (float)dim.x, idx.y / (float)dim.y, 1.0f, 1.0f);
}

extern "C" __global__ void __miss__() {}