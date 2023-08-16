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
    __constant__ PtStrategyParams params;
}

struct RadiancePayload
{
    float3 origin;
    float3 direction;

    float3 throughput = make_float3(1);
    float3 radiance = make_float3(0);

    SamplerState sampler;

    bool done = false;
};

// trace radiance ray
static __forceinline__ __device__ void trace_radiance(
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    RadiancePayload* payload_ptr)
{
    unsigned int u0, u1;
    pack_ptr(payload_ptr, u0, u1);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, u0, u1);
}

static __forceinline__ __device__ bool has_emission(const Material& material)
{
    // return (material.emission_color.x > 0 || material.emission_color.y > 0 ||
    //         material.emission_color.z > 0 ||
    //         material.emission_texture_id != -1);
}

static __forceinline__ __device__ float3 get_emission(const Material& material,
                                                      const float2& texcoord)
{
    // return material.emission_texture_id >= 0
    //            ? make_float3(
    //                  tex2D<float4>(params.textures[material.emission_texture_id]
    //                                    .texture_object,
    //                                texcoord.x, texcoord.y))
    //            : material.emission_color;
}

static __forceinline__ __device__ void fill_surface_info(
    const float3& ray_origin, const float3& ray_direction, float ray_tmax,
    const float2& barycentric, uint prim_idx, uint instance_idx, uint geom_id,
    SurfaceInfo& info)
{
    info.t = ray_tmax;
    info.barycentric = barycentric;

    const uint indices_offset = params.scene.indices_offsets[geom_id];
    const uint3 idx = params.scene.indices[indices_offset + prim_idx];

    const Matrix3x4& object_to_world =
        params.scene.object_to_worlds[instance_idx];
    const Matrix3x4& world_to_object =
        params.scene.world_to_objects[instance_idx];

    const float3 v0 =
        transform_position(object_to_world, params.scene.vertices[idx.x]);
    const float3 v1 =
        transform_position(object_to_world, params.scene.vertices[idx.y]);
    const float3 v2 =
        transform_position(object_to_world, params.scene.vertices[idx.z]);
    // surface based robust hit position, Ray Tracing Gems Chapter 6
    info.x = (1.0f - info.barycentric.x - info.barycentric.y) * v0 +
             info.barycentric.x * v1 + info.barycentric.y * v2;
    info.n_g = normalize(cross(v1 - v0, v2 - v0));

    const float3 n0 =
        transform_normal(world_to_object, params.scene.normals[idx.x]);
    const float3 n1 =
        transform_normal(world_to_object, params.scene.normals[idx.y]);
    const float3 n2 =
        transform_normal(world_to_object, params.scene.normals[idx.z]);
    info.n_s = normalize((1.0f - info.barycentric.x - info.barycentric.y) * n0 +
                         info.barycentric.x * n1 + info.barycentric.y * n2);

    const float2 tex0 = params.scene.texcoords[idx.x];
    const float2 tex1 = params.scene.texcoords[idx.y];
    const float2 tex2 = params.scene.texcoords[idx.z];
    info.texcoord = (1.0f - info.barycentric.x - info.barycentric.y) * tex0 +
                    info.barycentric.x * tex1 + info.barycentric.y * tex2;

    // flip normal
    info.is_entering = dot(-ray_direction, info.n_g) > 0;
    info.n_s = info.is_entering ? info.n_s : -info.n_s;
    info.n_g = info.is_entering ? info.n_g : -info.n_g;

    orthonormal_basis(info.n_s, info.tangent, info.bitangent);
}

static __forceinline__ __device__ ShadingParams fill_shading_params(
    const Material& material, const SurfaceInfo& surf_info,
    const TextureHeader* textures, ShadingParams& shading_params)
{
    // diffuse
    shading_params.diffuse = material.diffuse;

    // diffuse roughness
    shading_params.diffuse_roughness = material.diffuse_roughness;

    // base color
    shading_params.base_color =
        material.base_color_texture_id >= 0
            ? make_float3(tex2D<float4>(
                  textures[material.base_color_texture_id].texture_object,
                  surf_info.texcoord.x, surf_info.texcoord.y))
            : material.base_color;

    // specular
    shading_params.specular = material.specular;

    // specular color
    shading_params.specular_color =
        material.specular_color_texture_id >= 0
            ? make_float3(tex2D<float4>(
                  textures[material.specular_color_texture_id].texture_object,
                  surf_info.texcoord.x, surf_info.texcoord.y))
            : material.specular_color;

    // specular roughness
    shading_params.specular_roughness = clamp(
        material.specular_roughness_texture_id >= 0
            ? tex2D<float4>(textures[material.specular_roughness_texture_id]
                                .texture_object,
                            surf_info.texcoord.x, surf_info.texcoord.y)
                  .x
            : material.specular_roughness,
        0.01f, 1.0f);

    // metalness
    shading_params.metalness =
        material.metalness_texture_id >= 0
            ? tex2D<float4>(
                  textures[material.metalness_texture_id].texture_object,
                  surf_info.texcoord.x, surf_info.texcoord.y)
                  .x
            : material.metalness;

    // metallic roughness
    if (material.metallic_roughness_texture_id >= 0)
    {
        float4 mr = tex2D<float4>(
            textures[material.metallic_roughness_texture_id].texture_object,
            surf_info.texcoord.x, surf_info.texcoord.y);
        shading_params.specular_roughness = clamp(mr.y, 0.01f, 1.0f);
        shading_params.metalness = clamp(mr.z, 0.0f, 1.0f);
    }

    // coat
    shading_params.coat = clamp(
        material.coat_texture_id >= 0
            ? tex2D<float4>(textures[material.coat_texture_id].texture_object,
                            surf_info.texcoord.x, surf_info.texcoord.y)
                  .x
            : material.coat,
        0.0f, 1.0f);

    // coat roughness
    shading_params.coat_roughness = clamp(
        material.coat_roughness_texture_id >= 0
            ? tex2D<float4>(
                  textures[material.coat_roughness_texture_id].texture_object,
                  surf_info.texcoord.x, surf_info.texcoord.y)
                  .y
            : material.coat_roughness,
        0.0f, 1.0f);

    // transmission
    shading_params.transmission = material.transmission;

    // transmission color
    shading_params.transmission_color = material.transmission_color;

    // sheen
    shading_params.sheen = material.sheen;

    // sheen color
    shading_params.sheen_color = material.sheen_color;

    // sheen roughness
    shading_params.sheen_roughness = material.sheen_roughness;

    // subsurface
    shading_params.subsurface = material.subsurface;

    // subsurface color
    shading_params.subsurface_color = material.subsurface_color;

    // thin walled
    shading_params.thin_walled = material.thin_walled;
}

static __forceinline__ __device__ float3 fetch_ibl(const float3& v)
{
    // const float2 thphi = cartesian_to_spherical(v);
    // return params.sky_intensity *
    //        make_float3(tex2D<float4>(params.ibl, thphi.y / (2.0f * M_PIf),
    //                                  thphi.x / M_PIf));
}

static __forceinline__ __device__ float3 evaluate_arhosek_sky(const float3& v)
{
    // const float2 thphi = cartesian_to_spherical(v);
    // const float gamma = acosf(dot(params.sun_direction, v));
    // return params.sky_intensity *
    //        make_float3(arhosek_tristim_skymodel_radiance(params.arhosek,
    //                                                      thphi.x, gamma, 0),
    //                    arhosek_tristim_skymodel_radiance(params.arhosek,
    //                                                      thphi.x, gamma, 1),
    //                    arhosek_tristim_skymodel_radiance(params.arhosek,
    //                                                      thphi.x, gamma, 2));
}

// TODO: need more nice way to suppress firefly
static __forceinline__ __device__ float3 regularize_weight(const float3& weight)
{
    return clamp(weight, make_float3(0.0f), make_float3(1.0f));
}

static __forceinline__ __device__ void init_sampler_state(
    const uint3& idx, unsigned int image_idx, unsigned int n_spp,
    SamplerState& state)
{
    state.pcg_state.state =
        xxhash32(image_idx + n_spp * params.width * params.height);
    state.pcg_state.inc = xxhash32(params.seed);

    state.sobol_state.index = image_idx + n_spp * params.width * params.height;
    state.sobol_state.dimension = 1;
    state.sobol_state.seed = xxhash32(params.seed);

    state.cmj_state.image_idx = image_idx;
    state.cmj_state.depth = 0;
    state.cmj_state.n_spp = n_spp;
    state.cmj_state.scramble = xxhash32(params.seed);

    state.blue_noise_state.pixel_i = idx.x;
    state.blue_noise_state.pixel_j = idx.y;
    state.blue_noise_state.index = n_spp;
    state.blue_noise_state.dimension = 0;
}

// Ray Tracing Gems Chapter 6
static __forceinline__ __device__ float3 ray_origin_offset(const float3& p,
                                                           const float3& n)
{
    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;
    const int3 of_i = make_int3(int_scale * n);
    const float3 p_i = make_float3(
        __int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return make_float3(fabsf(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                       fabsf(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                       fabsf(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const uint image_idx = idx.x + params.width * idx.y;
    uint n_spp = params.sample_count[image_idx];

    float3 beauty = make_float3(params.output[image_idx]);

    RadiancePayload payload;
    for (int spp = 0; spp < params.n_samples; ++spp)
    {
        // initialize sampler
        init_sampler_state(idx, image_idx, n_spp, payload.sampler);

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

        n_spp++;
    }

    // update total number of samples
    params.sample_count[image_idx] = n_spp;

    // write results in render layers
    params.output[image_idx] = make_float4(beauty, 1.0f);
}

extern "C" __global__ void __miss__radiance()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    // firsthit light case
    float3 le = make_float3(1.0f);
    // if (params.ibl) { le = fetch_ibl(payload->direction); }
    // else if (params.arhosek) { le = evaluate_arhosek_sky(payload->direction);
    // } else { le = params.bg_color; }

    payload->radiance += payload->throughput * le;

    payload->done = true;
}

extern "C" __global__ void __anyhit__radiance()
{
    // const HitGroupSbtRecordData* sbt =
    //     reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
    // const uint prim_idx = optixGetPrimitiveIndex();

    // // get material info
    // const uint material_id = sbt->material_ids[prim_idx];
    // const Material& material = params.materials[material_id];

    // // fill surface info
    // const float2 barycentric = optixGetTriangleBarycentrics();

    // // calc texcoord
    // const uint3 idx = sbt->indices[prim_idx];
    // const float2 tex0 = params.texcoords[idx.x];
    // const float2 tex1 = params.texcoords[idx.y];
    // const float2 tex2 = params.texcoords[idx.z];
    // const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
    //                         barycentric.x * tex1 + barycentric.y * tex2;

    // // fetch base color texture
    // if (material.base_color_texture_id >= 0)
    // {
    //     const float alpha =
    //         tex2D<float4>(
    //             params.textures[material.base_color_texture_id].texture_object,
    //             texcoord.x, texcoord.y)
    //             .w;

    //     // ignore intersection
    //     if (alpha < 0.5) { optixIgnoreIntersection(); }
    // }

    // // fetch alpha texture
    // if (material.alpha_texture_id >= 0)
    // {
    //     const float alpha =
    //         tex2D<float4>(
    //             params.textures[material.alpha_texture_id].texture_object,
    //             texcoord.x, texcoord.y)
    //             .x;

    //     // ignore intersection
    //     if (alpha < 0.5) { optixIgnoreIntersection(); }
    // }
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

    const uint prim_idx = optixGetPrimitiveIndex();
    const uint instance_idx = optixGetInstanceIndex();
    const uint geom_id = params.scene.geometry_ids[instance_idx];

    // get material info
    // const uint material_id = sbt->material_ids[prim_idx];
    // const Material& material = params.materials[material_id];

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmax = optixGetRayTmax();
    const float2 barycentric = optixGetTriangleBarycentrics();

    SurfaceInfo surf_info;
    fill_surface_info(ray_origin, ray_direction, ray_tmax, barycentric,
                      prim_idx, instance_idx, geom_id, surf_info);

    ShadingParams shading_params;
    // fill_shading_params(material, surf_info, params.textures,
    // shading_params);

    float3 tangent = surf_info.tangent;
    float3 normal = surf_info.n_s;
    float3 bitangent = surf_info.bitangent;

    // bump mapping(with height map)
    // if (material.heightmap_texture_id >= 0)
    // {
    //     const TextureHeader& heightmap =
    //         params.textures[material.heightmap_texture_id];
    //     const float du = 1.0f / heightmap.size.x;
    //     const float dv = 1.0f / heightmap.size.y;
    //     const float v =
    //         tex2D<float4>(heightmap.texture_object, surf_info.texcoord.x,
    //                       surf_info.texcoord.y)
    //             .x;
    //     const float dfdu =
    //         (tex2D<float4>(heightmap.texture_object, surf_info.texcoord.x +
    //         du,
    //                        surf_info.texcoord.y)
    //              .x -
    //          v);
    //     const float dfdv =
    //         (tex2D<float4>(heightmap.texture_object, surf_info.texcoord.x,
    //                        surf_info.texcoord.y + dv)
    //              .x -
    //          v);
    //     tangent = normalize(surf_info.tangent + dfdu * surf_info.n_s);
    //     bitangent = normalize(surf_info.bitangent + dfdv * surf_info.n_s);
    //     normal = normalize(cross(tangent, bitangent));
    // }

    // normal mapping
    // if (material.normalmap_texture_id >= 0)
    // {
    //     float3 value = make_float3(tex2D<float4>(
    //         params.textures[material.normalmap_texture_id].texture_object,
    //         surf_info.texcoord.x, surf_info.texcoord.y));
    //     value = 2.0f * value - 1.0f;
    //     normal = normalize(local_to_world(value, surf_info.tangent,
    //                                       surf_info.bitangent,
    //                                       surf_info.n_s));
    //     orthonormal_basis(normal, tangent, bitangent);
    // }

    // Le
    // if (has_emission(material))
    // {
    //     payload->radiance +=
    //         payload->throughput * get_emission(material, surf_info.texcoord);
    //     payload->done = true;
    //     return;
    // }

    // init BSDF
    const float3 wo =
        world_to_local(-ray_direction, tangent, normal, bitangent);
    const BSDF bsdf = BSDF(wo, shading_params, surf_info.is_entering);

    // generate next ray direction
    {
        float3 f;
        float pdf;
        const float3 wi = bsdf.sample(wo, sample_1d(payload->sampler),
                                      sample_2d(payload->sampler), f, pdf);
        const float3 wi_world = local_to_world(wi, tangent, normal, bitangent);

        // update throughput
        payload->throughput *= f * abs_cos_theta(wi) / pdf;

        // advance ray
        const bool is_transmitted = dot(wi_world, surf_info.n_g) < 0;
        payload->origin = ray_origin_offset(
            surf_info.x, is_transmitted ? -surf_info.n_g : surf_info.n_g);
        payload->direction = wi_world;
    }
}