#pragma once
#include <optix.h>

#include "helper_math.h"
#include "shared.h"

namespace fredholm
{

struct SimpleStrategyParams
{
    uint width;
    uint height;
    CameraParams camera;
    SceneData scene;

    OptixTraversableHandle ias_handle;

    float4* output;
};

}  // namespace fredholm