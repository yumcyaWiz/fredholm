#include <optix.h>

#include "helper_math.h"
#include "render_strategy/common/arhosek.cu"
#include "render_strategy/common/bsdf.cu"
#include "render_strategy/common/camera.cu"
#include "render_strategy/common/math.cu"
#include "render_strategy/common/sampling.cu"
#include "render_strategy/common/util.cu"
#include "render_strategy/pt/pt_shared.h"
#include "shared.h"

using namespace fredholm;

extern "C"
{
    CUDA_CONSTANT PtStrategyParams params;
}

struct RadiancePayload
{
    float3 origin;
    float3 direction;

    float3 throughput = make_float3(1.0f);
    float3 radiance = make_float3(0.0f);

    SamplerState sampler;

    bool done = false;
};

// trace radiance ray
static CUDA_INLINE CUDA_DEVICE void trace_radiance(
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    RadiancePayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);
}

extern "C" CUDA_KERNEL void __raygen__()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;

    float3 beauty = make_float3(params.output[image_idx]);

    RadiancePayload payload;
    for (int spp = 0; spp < params.n_samples; ++spp)
    {
        // initialize sampler
        const uint n_spp = spp + params.sample_count;
        payload.sampler.init(params.width, params.height, make_uint2(idx),
                             n_spp, params.seed);

        // generate initial ray from camera
        float2 u = sample_2d(payload.sampler);
        float2 uv = make_float2((2.0f * (idx.x + u.x) - dim.x) / dim.y,
                                (2.0f * (idx.y + u.y) - dim.y) / dim.y);
        // flip x
        uv.x = -uv.x;
        u = sample_2d(payload.sampler);
        float camera_pdf;
        // sample_ray_thinlens_camera(params.camera, uv, u, payload.origin,
        //                            payload.direction, camera_pdf);
        sample_ray_pinhole_camera(params.camera, uv, payload.origin,
                                  payload.direction, camera_pdf);

        // start ray tracing from the camera
        payload.radiance = make_float3(0);
        // payload.throughput =
        //     make_float3(dot(payload.direction, params.camera.forward) /
        //     camera_pdf);
        payload.throughput = make_float3(1.0f);
        payload.done = false;
        for (int ray_depth = 0; ray_depth < params.max_depth; ++ray_depth)
        {
            // russian roulette
            const float russian_roulette_prob =
                ray_depth == 0
                    ? 1.0f
                    : clamp(rgb_to_luminance(payload.throughput), 0.0f, 1.0f);
            if (sample_1d(payload.sampler) >= russian_roulette_prob) { break; }
            payload.throughput /= russian_roulette_prob;

            // trace ray and update payloads
            trace_radiance(params.ias_handle, payload.origin, payload.direction,
                           0.0f, FLT_MAX, &payload);

            // throughput nan check
            if (isnan(payload.throughput) || isinf(payload.throughput))
            {
                break;
            }

            if (payload.done) { break; }
        }

        // radiance nan check
        float3 radiance = make_float3(0.0f);
        if (!isnan(payload.radiance) && !isinf(payload.radiance))
        {
            radiance = payload.radiance;
        }

        // streaming average
        const float coef = 1.0f / (n_spp + 1.0f);
        beauty = coef * (n_spp * beauty + radiance);
    }

    // write results in render layers
    params.output[image_idx] = make_float4(beauty, 1.0f);
}

extern "C" CUDA_KERNEL void __miss__()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    // firsthit light case
    float3 le = make_float3(0.0f);
    if (params.scene.envmap.is_valid())
    {
        le = fetch_envmap(params.scene.envmap, payload->direction);
    }

    payload->radiance += payload->throughput * le;
    payload->done = true;
}

extern "C" CUDA_KERNEL void __anyhit__()
{
    // TODO: implement alpha test
}

extern "C" CUDA_KERNEL void __closesthit__()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    const uint prim_idx = optixGetPrimitiveIndex();
    const uint instance_idx = optixGetInstanceIndex();
    const uint geom_id = params.scene.geometry_ids[instance_idx];
    const uint indices_offset = params.scene.indices_offsets[geom_id];

    const Material material = get_material(params.scene, prim_idx, geom_id);

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmax = optixGetRayTmax();
    const float2 barycentric = optixGetTriangleBarycentrics();

    SurfaceInfo surf_info(ray_origin, ray_direction, ray_tmax, barycentric,
                          params.scene, material, prim_idx, indices_offset,
                          instance_idx, geom_id);

    ShadingParams shading_params(material, surf_info.texcoord,
                                 params.scene.textures);

    // Le
    if (material.has_emission())
    {
        payload->radiance += payload->throughput *
                             material.get_emission_color(params.scene.textures,
                                                         surf_info.texcoord);
        payload->done = true;
        return;
    }

    // init BSDF
    const float3 wo = world_to_local(-ray_direction, surf_info.tangent,
                                     surf_info.n_s, surf_info.bitangent);
    const BSDF bsdf = BSDF(wo, shading_params, surf_info.is_entering);

    // generate next ray direction
    {
        float3 f;
        float pdf;
        const float3 wi = bsdf.sample(wo, sample_1d(payload->sampler),
                                      sample_2d(payload->sampler), f, pdf);
        const float3 wi_world = local_to_world(
            wi, surf_info.tangent, surf_info.n_s, surf_info.bitangent);

        // update throughput
        payload->throughput *= f * abs_cos_theta(wi) / pdf;

        // advance ray
        payload->origin =
            ray_origin_offset(surf_info.x, surf_info.n_g, wi_world);
        payload->direction = wi_world;
    }
}