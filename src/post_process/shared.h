#pragma once
#include "cuda_util.h"

struct PostProcessParams
{
    uint width;
    uint height;
    float4* input;
    float4* output;
};