#pragma once

#include <optix.h>

#include "cuda_util.h"
#include "helper_math.h"

#define FLT_MAX 1e9f

struct Ray
{
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = make_float3(0.0f, 0.0f, 0.0f);
    float tmin = 0.0f;
    float tmax = FLT_MAX;
};