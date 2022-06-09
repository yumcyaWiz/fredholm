#include <optix.h>

#include "shared.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__white()
{
  uint3 launch_index = optixGetLaunchIndex();
  params.image[launch_index.y * params.image_width + launch_index.x] =
      make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}