#include <optix.h>

#include "render_strategy/common/camera.cu"
#include "render_strategy/common/util.cu"
#include "render_strategy/simple/simple_shared.h"
#include "shared.h"

using namespace fredholm;

extern "C"
{
    CUDA_CONSTANT SimpleStrategyParams params;
}

struct RayPayload
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
};

static CUDA_INLINE CUDA_DEVICE void trace_ray(
    const OptixTraversableHandle& handle, const Ray& ray,
    RayPayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray.origin, ray.direction, ray.tmin, ray.tmax, 0.0f,
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);
}

extern "C" CUDA_KERNEL void __raygen__rg()
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

extern "C" CUDA_KERNEL void __miss__()
{
    RayPayload* payload_ptr = get_payload_ptr<RayPayload>();
    payload_ptr->color = make_float3(0.0f, 0.0f, 0.0f);
}

extern "C" CUDA_KERNEL void __anyhit__() {}

extern "C" CUDA_KERNEL void __closesthit__()
{
    RayPayload* payload_ptr = get_payload_ptr<RayPayload>();

    const uint prim_id = optixGetPrimitiveIndex();
    const uint instance_id = optixGetInstanceIndex();
    const uint geom_id = params.scene.geometry_ids[instance_id];
    const uint vertices_offset = params.scene.n_vertices[geom_id];
    const uint indices_offset = params.scene.n_faces[geom_id];

    const Material material =
        get_material(params.scene, prim_id, indices_offset);

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmax = optixGetRayTmax();
    const float2 barycentric = optixGetTriangleBarycentrics();

    SurfaceInfo surf_info(ray_origin, ray_direction, ray_tmax, barycentric,
                          params.scene, material, prim_id, vertices_offset,
                          indices_offset, instance_id, geom_id);

    // position
    if (params.output_mode == 0) { payload_ptr->color = surf_info.x; }
    // normal
    else if (params.output_mode == 1)
    {
        payload_ptr->color = 0.5f * (surf_info.n_s + 1.0f);
    }
    // texcoord
    else if (params.output_mode == 2)
    {
        payload_ptr->color = make_float3(surf_info.texcoord, 0.0f);
    }
    // barycentric
    else if (params.output_mode == 3)
    {
        payload_ptr->color = make_float3(surf_info.barycentric, 0.0f);
    }
    // clearcoat
    else if (params.output_mode == 4)
    {
        const float clearcoat =
            material.get_coat(params.scene.textures, surf_info.texcoord);
        payload_ptr->color = make_float3(clearcoat, clearcoat, clearcoat);
    }
    // specular
    else if (params.output_mode == 5)
    {
        const float specular = material.specular;
    }
    // specular color
    else if (params.output_mode == 6)
    {
        const float3 specular_color = material.get_specular_color(
            params.scene.textures, surf_info.texcoord);
        payload_ptr->color = specular_color;
    }
    // transmission
    else if (params.output_mode == 7)
    {
        const float transmission = material.get_transmission(
            params.scene.textures, surf_info.texcoord);
        payload_ptr->color =
            make_float3(transmission, transmission, transmission);
    }
    // diffuse color
    else if (params.output_mode == 8)
    {
        const float3 diffuse_color = material.get_diffuse_color(
            params.scene.textures, surf_info.texcoord);
        payload_ptr->color = diffuse_color;
    }
    // emission color
    else if (params.output_mode == 9)
    {
        const float3 emission_color = material.get_emission_color(
            params.scene.textures, surf_info.texcoord);
        payload_ptr->color = emission_color;
    }
}