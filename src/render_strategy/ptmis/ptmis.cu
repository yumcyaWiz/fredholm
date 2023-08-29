#include <optix.h>

#include "helper_math.h"
#include "render_strategy/common/arhosek.cu"
#include "render_strategy/common/bsdf.cu"
#include "render_strategy/common/camera.cu"
#include "render_strategy/common/math.cu"
#include "render_strategy/common/sampling.cu"
#include "render_strategy/common/util.cu"
#include "render_strategy/ptmis/ptmis_shared.h"
#include "shared.h"

using namespace fredholm;

extern "C"
{
    CUDA_CONSTANT PTMISStrategyParams params;
}

struct RadiancePayload
{
    float3 origin = make_float3(0.0f);
    float3 direction = make_float3(0.0f);

    float3 throughput = make_float3(1.0f);
    float3 radiance = make_float3(0.0f);

    SamplerState sampler;

    bool firsthit = true;
    bool done = false;
};

struct ShadowPayload
{
    bool visible = false;  // light visibility
};

struct LightPayload
{
    float3 direction;  // ray direction

    float3 le = make_float3(0.0f);  // emission

    bool done = false;  // hit sky?
    float3 p;           // hit position
    float3 n;           // hit normal
    float area;         // triangle area
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
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 3, 0, u0, u1);
}

static CUDA_INLINE CUDA_DEVICE void trace_shadow(OptixTraversableHandle& handle,
                                                 const float3& ray_origin,
                                                 const float3& ray_direction,
                                                 float tmin, float tmax,
                                                 ShadowPayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax - SHADOW_RAY_EPS,
               0.0f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 3, 1, u0, u1);
}

static CUDA_INLINE CUDA_DEVICE void trace_light(OptixTraversableHandle& handle,
                                                const float3& ray_origin,
                                                const float3& ray_direction,
                                                float tmin, float tmax,
                                                LightPayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 2, 3, 2, u0, u1);
}

static CUDA_INLINE CUDA_DEVICE float3
sample_position_on_light(const SceneData& scene, float u, const float2& v,
                         float3& le, float3& n, float& pdf)
{
    // sample light
    // TODO: implement better light sampling
    const uint light_idx = clamp(static_cast<uint>(u * scene.n_area_lights), 0u,
                                 scene.n_area_lights - 1);
    const AreaLight& light = scene.area_lights[light_idx];

    // sample point on the light
    const float2 barycentric = sample_triangle(v);

    const Matrix3x4& object_to_world =
        scene.object_to_worlds[light.instance_idx];
    const Matrix3x4& world_to_object =
        scene.world_to_objects[light.instance_idx];

    const uint3& idx = light.indices;
    const float3 v0 =
        transform_position(object_to_world, scene.vertices[idx.x]);
    const float3 v1 =
        transform_position(object_to_world, scene.vertices[idx.y]);
    const float3 v2 =
        transform_position(object_to_world, scene.vertices[idx.z]);

    const float3 n0 = transform_normal(world_to_object, scene.normals[idx.x]);
    const float3 n1 = transform_normal(world_to_object, scene.normals[idx.y]);
    const float3 n2 = transform_normal(world_to_object, scene.normals[idx.z]);

    const float2 tex0 = scene.texcoords[idx.x];
    const float2 tex1 = scene.texcoords[idx.y];
    const float2 tex2 = scene.texcoords[idx.z];

    const float3 p = (1.0f - barycentric.x - barycentric.y) * v0 +
                     barycentric.x * v1 + barycentric.y * v2;
    n = normalize((1.0f - barycentric.x - barycentric.y) * n0 +
                  barycentric.x * n1 + barycentric.y * n2);
    const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                            barycentric.x * tex1 + barycentric.y * tex2;

    const float area = 0.5f * length(cross(v1 - v0, v2 - v0));

    const auto& material = scene.materials[light.material_id];
    le = material.get_emission_color(scene.textures, texcoord);
    pdf = 1.0f / (scene.n_area_lights * area);

    return p;
}

static CUDA_INLINE CUDA_DEVICE float3 sample_position_on_directional_light(
    const SceneData& scene, const float2& u, float3& le)
{
    constexpr float DIRECTIONAL_LIGHT_DISTANCE = 1e9f;

    le = scene.directional_light.le;

    // sample point on disk
    const float2 p_disk = sample_concentric_disk(u);

    // compute world space position
    const float disk_radius =
        DIRECTIONAL_LIGHT_DISTANCE *
        tan(deg_to_rad(0.5f * scene.directional_light.angle));
    float3 t, b;
    orthonormal_basis(scene.directional_light.dir, t, b);
    const float3 p = DIRECTIONAL_LIGHT_DISTANCE * scene.directional_light.dir +
                     disk_radius * (t * p_disk.x + b * p_disk.y);

    return p;
}

// balance heuristics
static CUDA_INLINE CUDA_DEVICE float compute_mis_weight(float pdf0, float pdf1)
{
    return pdf0 / (pdf0 + pdf1);
}

extern "C" CUDA_KERNEL void __raygen__()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;

    float3 beauty = make_float3(params.output[image_idx]);
    uint sample_count = uint(params.output[image_idx].w);

    for (int spp = 0; spp < params.n_samples; ++spp)
    {
        RadiancePayload payload;

        // initialize sampler
        const uint n_spp = spp + sample_count;
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
        sample_ray_thinlens_camera(params.camera, uv, u, payload.origin,
                                   payload.direction, camera_pdf);

        // start ray tracing from the camera
        // TODO: multiply cos / pdf
        payload.throughput = make_float3(1.0f);
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
    params.output[image_idx] =
        make_float4(beauty, sample_count + params.n_samples);
}

extern "C" CUDA_KERNEL void __miss__radiance()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    // firsthit light case
    if (payload->firsthit)
    {
        float3 le = make_float3(0.0f);
        if (params.scene.envmap.is_valid())
        {
            le = fetch_envmap(params.scene.envmap, payload->direction);
        }
        payload->radiance += payload->throughput * le;
    }

    payload->done = true;
}

extern "C" CUDA_KERNEL void __anyhit__radiance()
{
    // TODO: implement alpha test
}

extern "C" CUDA_KERNEL void __closesthit__radiance()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    const uint prim_idx = optixGetPrimitiveIndex();
    const uint instance_idx = optixGetInstanceIndex();
    const uint geom_id = params.scene.geometry_ids[instance_idx];
    const uint vertices_offset = params.scene.n_vertices[geom_id];
    const uint indices_offset = params.scene.n_faces[geom_id];

    const Material material =
        get_material(params.scene, prim_idx, indices_offset);

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmax = optixGetRayTmax();
    const float2 barycentric = optixGetTriangleBarycentrics();

    SurfaceInfo surf_info(ray_origin, ray_direction, ray_tmax, barycentric,
                          params.scene, material, prim_idx, vertices_offset,
                          indices_offset, instance_idx, geom_id);

    ShadingParams shading_params(material, surf_info.texcoord,
                                 params.scene.textures);

    if (payload->firsthit)
    {
        payload->firsthit = false;

        // first hit light case
        if (material.has_emission())
        {
            payload->radiance += payload->throughput *
                                 material.get_emission_color(
                                     params.scene.textures, surf_info.texcoord);
            payload->done = true;
            return;
        }
    }

    // init BSDF
    const float3 wo = world_to_local(-ray_direction, surf_info.tangent,
                                     surf_info.n_s, surf_info.bitangent);
    const BSDF bsdf = BSDF(wo, shading_params, surf_info.is_entering);

    // light sampling
    {
        const float3 shadow_ray_origin =
            ray_origin_offset(surf_info.x, surf_info.n_g);

        // sky
        if (params.scene.envmap.is_valid())
        {
            // TODO: implement IBL importance sampling
            const float3 wi =
                sample_cosine_weighted_hemisphere(sample_2d(payload->sampler));
            const float3 shadow_ray_direction = local_to_world(
                wi, surf_info.tangent, surf_info.n_s, surf_info.bitangent);

            ShadowPayload shadow_payload;
            trace_shadow(params.ias_handle, shadow_ray_origin,
                         shadow_ray_direction, 0.0f, FLT_MAX, &shadow_payload);

            if (shadow_payload.visible)
            {
                const float3 f = bsdf.eval(wo, wi);
                const float pdf = abs_cos_theta(wi) / M_PIf;
                const float pdf_bsdf = bsdf.eval_pdf(wo, wi);
                const float mis_weight = compute_mis_weight(pdf, pdf_bsdf);
                const float3 weight = payload->throughput * mis_weight * f *
                                      abs_cos_theta(wi) / pdf;
                const float3 le =
                    fetch_envmap(params.scene.envmap, shadow_ray_direction);
                payload->radiance += weight * le;
            }
        }

        // directional light
        if (params.scene.directional_light.is_valid())
        {
            float3 le;
            const float3 p = sample_position_on_directional_light(
                params.scene, sample_2d(payload->sampler), le);
            const float3 shadow_ray_direction =
                normalize(p - shadow_ray_origin);

            ShadowPayload shadow_payload;
            trace_shadow(params.ias_handle, shadow_ray_origin,
                         shadow_ray_direction, 0.0f, FLT_MAX, &shadow_payload);

            if (shadow_payload.visible)
            {
                const float3 wi =
                    world_to_local(shadow_ray_direction, surf_info.tangent,
                                   surf_info.n_s, surf_info.bitangent);
                const float3 f = bsdf.eval(wo, wi);
                const float pdf = 1.0f;
                const float pdf_bsdf = bsdf.eval_pdf(wo, wi);
                const float mis_weight = compute_mis_weight(pdf, pdf_bsdf);
                const float3 weight = payload->throughput * mis_weight * f *
                                      abs_cos_theta(wi) / pdf;
                payload->radiance += weight * le;
            }
        }

        // area light
        if (params.scene.n_area_lights > 0)
        {
            float3 le, n;
            float pdf_area;
            const float3 p = sample_position_on_light(
                params.scene, sample_1d(payload->sampler),
                sample_2d(payload->sampler), le, n, pdf_area);

            const float3 shadow_ray_direction =
                normalize(p - shadow_ray_origin);
            const float r = length(p - shadow_ray_origin);

            ShadowPayload shadow_payload;
            trace_shadow(params.ias_handle, shadow_ray_origin,
                         shadow_ray_direction, 0.0f, r, &shadow_payload);

            if (shadow_payload.visible && dot(-shadow_ray_direction, n) > 0.0f)
            {
                const float3 wi =
                    world_to_local(shadow_ray_direction, surf_info.tangent,
                                   surf_info.n_s, surf_info.bitangent);
                const float3 f = bsdf.eval(wo, wi);
                float pdf =
                    r * r / fabs(dot(-shadow_ray_direction, n)) * pdf_area;

                const float pdf_bsdf = bsdf.eval_pdf(wo, wi);
                const float mis_weight = compute_mis_weight(pdf, pdf_bsdf);
                const float3 weight = payload->throughput * mis_weight * f *
                                      abs_cos_theta(wi) / pdf;
                payload->radiance += weight * le;
            }
        }
    }

    // BSDF sampling
    {
        // TODO: create BSDFSample struct
        float3 f;
        float pdf;
        const float3 wi = bsdf.sample(wo, sample_1d(payload->sampler),
                                      sample_2d(payload->sampler), f, pdf);

        // update throughput
        payload->throughput *= f * abs_cos_theta(wi) / pdf;

        // trace light ray
        const float3 light_ray_direction = local_to_world(
            wi, surf_info.tangent, surf_info.n_s, surf_info.bitangent);
        const bool is_transmitted = dot(light_ray_direction, surf_info.n_g) < 0;
        const float3 light_ray_origin = ray_origin_offset(
            surf_info.x, is_transmitted ? -surf_info.n_g : surf_info.n_g);

        LightPayload light_payload;
        light_payload.direction = light_ray_direction;
        trace_light(params.ias_handle, light_ray_origin, light_ray_direction,
                    0.0f, FLT_MAX, &light_payload);

        if (light_payload.done)
        {
            // IBL importance sampling pdf
            const float pdf_light = abs_cos_theta(wi) / M_PIf;

            const float mis_weight = compute_mis_weight(pdf, pdf_light);
            const float3 weight = payload->throughput * mis_weight;
            payload->radiance += weight * light_payload.le;
            payload->done = true;
            return;
        }
        else
        {
            // hit area light
            const float r2 = dot(light_payload.p - light_ray_origin,
                                 light_payload.p - light_ray_origin);
            const float pdf_area =
                1.0f / (params.scene.n_area_lights * light_payload.area);
            const float pdf_light =
                r2 / fabs(dot(-light_ray_direction, light_payload.n)) *
                pdf_area;

            const float mis_weight = compute_mis_weight(pdf, pdf_light);
            const float3 weight = payload->throughput * mis_weight;
            payload->radiance += weight * light_payload.le;
        }

        const float3 wi_world = local_to_world(
            wi, surf_info.tangent, surf_info.n_s, surf_info.bitangent);

        // advance ray
        payload->origin =
            ray_origin_offset(surf_info.x, surf_info.n_g, wi_world);
        payload->direction = wi_world;
    }
}

extern "C" __global__ void __miss__shadow()
{
    ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
    payload->visible = true;
}

extern "C" __global__ void __anyhit__shadow()
{
    // TODO: implement alpha testing
}

extern "C" __global__ void __closesthit__shadow()
{
    ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
    payload->visible = false;
}

extern "C" __global__ void __miss__light()
{
    LightPayload* payload = get_payload_ptr<LightPayload>();
    payload->done = true;

    float3 le = make_float3(0.0f);
    if (params.scene.envmap.is_valid())
    {
        le = fetch_envmap(params.scene.envmap, payload->direction);
    }

    payload->le = le;
}

extern "C" __global__ void __anyhit__light()
{
    // TODO: implement alpha testing
}

extern "C" __global__ void __closesthit__light()
{
    LightPayload* payload = get_payload_ptr<LightPayload>();

    const uint prim_idx = optixGetPrimitiveIndex();
    const uint instance_idx = optixGetInstanceIndex();
    const uint geom_id = params.scene.geometry_ids[instance_idx];
    const uint vertices_offset = params.scene.n_vertices[geom_id];
    const uint indices_offset = params.scene.n_faces[geom_id];

    const Material material =
        get_material(params.scene, prim_idx, indices_offset);

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmax = optixGetRayTmax();
    const float2 barycentric = optixGetTriangleBarycentrics();

    SurfaceInfo surf_info(ray_origin, ray_direction, ray_tmax, barycentric,
                          params.scene, material, prim_idx, vertices_offset,
                          indices_offset, instance_idx, geom_id);

    if (material.has_emission() &&
        dot(-payload->direction, surf_info.n_s) > 0.0f)
    {
        payload->done = false;
        payload->le = material.get_emission_color(params.scene.textures,
                                                  surf_info.texcoord);
        payload->p = surf_info.x;
        payload->n = surf_info.n_s;
        payload->area = surf_info.area;
    }
    else
    {
        payload->done = false;
        payload->le = make_float3(0.0f);
    }
}