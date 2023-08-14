#include <optix.h>

#include "shared.h"

using namespace fredholm;

extern "C"
{
    __constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;

    params.render_layer.beauty[image_idx] =
        make_float4(idx.x / (float)dim.x, idx.y / (float)dim.y, 1.0f, 1.0f);
}

extern "C" __global__ void __miss__() {}