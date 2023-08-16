#pragma once
#include <optix.h>

#include "helper_math.h"
#include "shared.h"

namespace fredholm
{

struct HelloStrategyParams
{
    uint width;
    uint height;

    float4* output;
};

}  // namespace fredholm