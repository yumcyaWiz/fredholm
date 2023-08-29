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
    const float2 barycentric = optixGetTriangleBarycentrics();

    const uint geom_id = params.scene.geometry_ids[instance_id];
    const uint vertices_offset = params.scene.n_vertices[geom_id];
    const uint indices_offset = params.scene.n_faces[geom_id];
    const uint3 idx =
        params.scene.indices[indices_offset + prim_id] + vertices_offset;

    const Matrix3x4 object_to_world =
        params.scene.object_to_worlds[instance_id];
    const Matrix3x4 world_to_object =
        params.scene.world_to_objects[instance_id];

    const float3 v0 =
        transform_position(object_to_world, params.scene.vertices[idx.x]);
    const float3 v1 =
        transform_position(object_to_world, params.scene.vertices[idx.y]);
    const float3 v2 =
        transform_position(object_to_world, params.scene.vertices[idx.z]);
    const float3 x = (1.0f - barycentric.x - barycentric.y) * v0 +
                     barycentric.x * v1 + barycentric.y * v2;

    const float3 n0 =
        transform_normal(world_to_object, params.scene.normals[idx.x]);
    const float3 n1 =
        transform_normal(world_to_object, params.scene.normals[idx.y]);
    const float3 n2 =
        transform_normal(world_to_object, params.scene.normals[idx.z]);
    const float3 ns = normalize((1.0f - barycentric.x - barycentric.y) * n0 +
                                barycentric.x * n1 + barycentric.y * n2);

    const float2 tex0 = params.scene.texcoords[idx.x];
    const float2 tex1 = params.scene.texcoords[idx.y];
    const float2 tex2 = params.scene.texcoords[idx.z];
    const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                            barycentric.x * tex1 + barycentric.y * tex2;

    const uint material_id =
        params.scene.material_ids[indices_offset + prim_id];
    const Material& material = params.scene.materials[material_id];
    const float3 diffuse_color =
        material.get_diffuse_color(params.scene.textures, texcoord);

    if (params.output_mode == 0) { payload_ptr->color = x; }
    else if (params.output_mode == 1)
    {
        payload_ptr->color = 0.5f * (ns + 1.0f);
    }
    else if (params.output_mode == 2)
    {
        payload_ptr->color = make_float3(texcoord, 0.0f);
    }
    else if (params.output_mode == 3)
    {
        payload_ptr->color = make_float3(barycentric, 0.0f);
    }
    else if (params.output_mode == 4) { payload_ptr->color = diffuse_color; }
}