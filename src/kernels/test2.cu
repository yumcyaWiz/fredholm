#include <optix.h>

#include "camera.cu"
#include "shared.h"
#include "util.cu"

using namespace fredholm;

#define FLT_MAX 1e9f

extern "C"
{
    __constant__ LaunchParams params;
}

struct Ray
{
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = make_float3(0.0f, 0.0f, 0.0f);
    float tmin = 0.0f;
    float tmax = FLT_MAX;
};

struct RayPayload
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
};

static __forceinline__ __device__ void trace_ray(
    const OptixTraversableHandle& handle, const Ray& ray,
    RayPayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray.origin, ray.direction, ray.tmin, ray.tmax, 0.0f,
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;

    // sample primary ray
    float2 uv = make_float2((2.0f * idx.x - dim.x) / dim.x,
                            (2.0f * idx.y - dim.y) / dim.y);
    uv.x = -uv.x;
    Ray ray;
    float camera_pdf;
    sample_ray_pinhole_camera(params.camera, uv, ray.origin, ray.direction,
                              camera_pdf);

    // trace ray
    RayPayload payload;
    trace_ray(params.ias_handle, ray, &payload);

    params.render_layer.beauty[image_idx] = make_float4(payload.color, 1.0f);
}

extern "C" __global__ void __miss__()
{
    RayPayload* payload_ptr = get_payload_ptr<RayPayload>();
    payload_ptr->color = make_float3(0.0f, 0.0f, 0.0f);
}

extern "C" __global__ void __anyhit__() {}

extern "C" __global__ void __closesthit__()
{
    RayPayload* payload_ptr = get_payload_ptr<RayPayload>();
    payload_ptr->color = make_float3(1.0f, 1.0f, 1.0f);
}