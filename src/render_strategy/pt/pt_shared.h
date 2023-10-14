#pragma once

#include <optix.h>

#include "helper_math.h"
#include "shared.h"

namespace fredholm
{

struct PtStrategyParams
{
    uint width;
    uint height;
    CameraParams camera;
    SceneData scene;

    OptixTraversableHandle ias_handle;

    uint n_samples;
    uint max_depth;
    uint seed;

    float4* beauty;
    float4* position;
    float4* normal;
    float4* albedo;
};

}  // namespace fredholm