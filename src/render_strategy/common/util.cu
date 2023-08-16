#pragma once

#include <optix.h>

#include "cuda_util.h"
#include "helper_math.h"

#define FLT_MAX 1e9f
#define SHADOW_RAY_EPS 0.001f

struct Ray
{
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = make_float3(0.0f, 0.0f, 0.0f);
    float tmin = 0.0f;
    float tmax = FLT_MAX;
};

// upper-32bit + lower-32bit -> 64bit
static __forceinline__ __device__ void* unpack_ptr(unsigned int i0,
                                                   unsigned int i1)
{
    const unsigned long long uptr =
        static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

// 64bit -> upper-32bit + lower-32bit
static __forceinline__ __device__ void pack_ptr(void* ptr, unsigned int& i0,
                                                unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

// u0, u1 is upper-32bit, lower-32bit of ptr of Payload
template <typename Payload>
static __forceinline__ __device__ Payload* get_payload_ptr()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<Payload*>(unpack_ptr(u0, u1));
}

// Ray Tracing Gems Chapter 6
static __forceinline__ __device__ float3 ray_origin_offset(const float3& p,
                                                           const float3& n,
                                                           const float3& wi)
{
    // flip normal
    const float3 t = copysignf(1.0f, dot(wi, n)) * n;

    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;
    const int3 of_i = make_int3(int_scale * t);
    const float3 p_i = make_float3(
        __int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return make_float3(fabsf(p.x) < origin ? p.x + float_scale * t.x : p_i.x,
                       fabsf(p.y) < origin ? p.y + float_scale * t.y : p_i.y,
                       fabsf(p.z) < origin ? p.z + float_scale * t.z : p_i.z);
}

// TODO: need more nice way to suppress firefly
static __forceinline__ __device__ float3 regularize_weight(const float3& weight)
{
    return clamp(weight, make_float3(0.0f), make_float3(1.0f));
}

static __forceinline__ __device__ void fill_surface_info(
    const float3& ray_origin, const float3& ray_direction, float ray_tmax,
    const float2& barycentric, const SceneData& scene, uint prim_idx,
    uint instance_idx, uint geom_id, SurfaceInfo& info)
{
    info.t = ray_tmax;
    info.barycentric = barycentric;

    const uint indices_offset = scene.indices_offsets[geom_id];
    const uint3 idx = scene.indices[indices_offset + prim_idx];

    const Matrix3x4& object_to_world = scene.object_to_worlds[instance_idx];
    const Matrix3x4& world_to_object = scene.world_to_objects[instance_idx];

    const float3 v0 =
        transform_position(object_to_world, scene.vertices[idx.x]);
    const float3 v1 =
        transform_position(object_to_world, scene.vertices[idx.y]);
    const float3 v2 =
        transform_position(object_to_world, scene.vertices[idx.z]);
    // surface based robust hit position, Ray Tracing Gems Chapter 6
    info.x = (1.0f - info.barycentric.x - info.barycentric.y) * v0 +
             info.barycentric.x * v1 + info.barycentric.y * v2;
    info.n_g = normalize(cross(v1 - v0, v2 - v0));

    const float3 n0 = transform_normal(world_to_object, scene.normals[idx.x]);
    const float3 n1 = transform_normal(world_to_object, scene.normals[idx.y]);
    const float3 n2 = transform_normal(world_to_object, scene.normals[idx.z]);
    info.n_s = normalize((1.0f - info.barycentric.x - info.barycentric.y) * n0 +
                         info.barycentric.x * n1 + info.barycentric.y * n2);

    const float2 tex0 = scene.texcoords[idx.x];
    const float2 tex1 = scene.texcoords[idx.y];
    const float2 tex2 = scene.texcoords[idx.z];
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