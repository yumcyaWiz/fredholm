#include <optix.h>

#include "optix/camera.cu"
#include "render_strategy/common.cu"
#include "render_strategy/simple/simple_shared.h"
#include "shared.h"

using namespace fredholm;

extern "C"
{
    __constant__ SimpleStrategyParams params;
}

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

    params.output[image_idx] = make_float4(payload.color, 1.0f);
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

    const uint prim_id = optixGetPrimitiveIndex();
    const uint instance_id = optixGetInstanceIndex();
    const float2 barycentric = optixGetTriangleBarycentrics();

    const uint geom_id = params.scene.geometry_ids[instance_id];
    const uint indices_offset = params.scene.indices_offsets[geom_id];
    const uint3 idx = params.scene.indices[indices_offset + prim_id];

    const Matrix3x4 world_to_object =
        params.scene.world_to_objects[instance_id];

    const float3 n0 =
        transform_normal(world_to_object, params.scene.normals[idx.x]);
    const float3 n1 =
        transform_normal(world_to_object, params.scene.normals[idx.y]);
    const float3 n2 =
        transform_normal(world_to_object, params.scene.normals[idx.z]);
    const float3 ns = normalize((1.0f - barycentric.x - barycentric.y) * n0 +
                                barycentric.x * n1 + barycentric.y * n2);

    payload_ptr->color = 0.5f * (ns + 1.0f);
}