#include <optix.h>

#include "shared.h"
#include "sutil/vec_math.h"

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void set_payload(float3 p)
{
  optixSetPayload_0(__float_as_int(p.x));
  optixSetPayload_1(__float_as_int(p.y));
  optixSetPayload_2(__float_as_int(p.z));
}

extern "C" __global__ void __raygen__rg()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();

  const float2 uv = make_float2((2.0f * idx.x - dim.x) / dim.x,
                                (2.0f * idx.y - dim.y) / dim.y);
  const float3 ray_origin = params.cam_origin;
  const float3 ray_direction = normalize(
      uv.x * params.cam_right + uv.y * params.cam_up + params.cam_forward);

  unsigned int p0, p1, p2;
  optixTrace(params.handle, ray_origin, ray_direction, 0.0f, 1e9f, 0.0f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1,
             p2);
  float4 color;
  color.x = __int_as_float(p0);
  color.y = __int_as_float(p1);
  color.z = __int_as_float(p2);

  params.image[idx.x + idx.y * params.image_width] = color;
}

extern "C" __global__ void __miss__ms() {}

extern "C" __global__ void __closesthit__ch()
{
  const float2 barycentrics = optixGetTriangleBarycentrics();

  set_payload(make_float3(barycentrics, 1.0f));
}