#include <optix.h>

#include "shared.h"

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__white()
{
  uint3 launch_index = optixGetLaunchIndex();
  params.framebuffer[launch_index.y * params.width + launch_index.x] =
      make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}