#include <optix.h>

struct Params {
  float3* image;
  unsigned int image_width;
  unsigned int image_height;
};

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__white()
{
  uint3 launch_index = optixGetLaunchIndex();
  params.image[launch_index.y * params.image_width + launch_index.x] =
      make_float3(1.0f, 1.0f, 1.0f);
}