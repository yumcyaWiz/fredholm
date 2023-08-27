#pragma once

#include <optix.h>

#include "helper_math.h"
#include "shared.h"

namespace fredholm
{

struct PTMISStrategyParams
{
    uint width;
    uint height;
    CameraParams camera;
    SceneData scene;

    OptixTraversableHandle ias_handle;

    uint n_samples;
    uint max_depth;
    uint seed;

    float4* output;
};

}  // namespace fredholm