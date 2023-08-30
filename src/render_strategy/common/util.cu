#pragma once

#include <optix.h>

#include "arhosek.cu"
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
static CUDA_INLINE CUDA_DEVICE void* unpack_ptr(unsigned int i0,
                                                unsigned int i1)
{
    const unsigned long long uptr =
        static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

// 64bit -> upper-32bit + lower-32bit
static CUDA_INLINE CUDA_DEVICE void pack_ptr(void* ptr, unsigned int& i0,
                                             unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

// u0, u1 is upper-32bit, lower-32bit of ptr of Payload
template <typename Payload>
static CUDA_INLINE CUDA_DEVICE Payload* get_payload_ptr()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<Payload*>(unpack_ptr(u0, u1));
}

// Ray Tracing Gems Chapter 6
static CUDA_INLINE CUDA_DEVICE float3 ray_origin_offset(const float3& p,
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

// Ray Tracing Gems Chapter 6
static CUDA_INLINE CUDA_DEVICE float3 ray_origin_offset(const float3& p,
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
static CUDA_INLINE CUDA_DEVICE float3 regularize_weight(const float3& weight)
{
    return clamp(weight, make_float3(0.0f), make_float3(1.0f));
}

static CUDA_INLINE CUDA_DEVICE Material get_material(const SceneData& scene,
                                                     uint prim_id,
                                                     uint indices_offset)
{
    const uint material_id = scene.material_ids[indices_offset + prim_id];
    return scene.materials[material_id];
}

struct SurfaceInfo
{
    float t = 0.0f;                           // ray tmax
    float3 x = make_float3(0, 0, 0);          // shading position
    float3 n_g = make_float3(0, 0, 0);        // geometric normal in world space
    float3 n_s = make_float3(0, 0, 0);        // shading normal in world space
    float2 barycentric = make_float2(0, 0);   // barycentric coordinate
    float2 texcoord = make_float2(0, 0);      // texture coordinate
    float3 tangent = make_float3(0, 0, 0);    // tangent vector in world space
    float3 bitangent = make_float3(0, 0, 0);  // bitangent vector in world space
    float area = 0.0f;                        // triangle area
    bool is_entering = true;

    CUDA_INLINE CUDA_DEVICE SurfaceInfo() {}

    CUDA_INLINE CUDA_DEVICE
    SurfaceInfo(const float3& origin, const float3& direction, float tmax,
                const float2& barycentric, const SceneData& scene,
                const Material& material, uint prim_idx, uint vertices_offset,
                uint indices_offset, uint instance_idx, uint geom_id)
    {
        this->t = tmax;
        this->barycentric = barycentric;

        const uint3 idx =
            scene.indices[indices_offset + prim_idx] + vertices_offset;
        const Matrix3x4& object_to_world = scene.object_to_worlds[instance_idx];
        const Matrix3x4& world_to_object = scene.world_to_objects[instance_idx];

        const float3 v0 =
            transform_position(object_to_world, scene.vertices[idx.x]);
        const float3 v1 =
            transform_position(object_to_world, scene.vertices[idx.y]);
        const float3 v2 =
            transform_position(object_to_world, scene.vertices[idx.z]);
        // surface based robust hit position, Ray Tracing Gems Chapter 6
        x = (1.0f - barycentric.x - barycentric.y) * v0 + barycentric.x * v1 +
            barycentric.y * v2;
        n_g = normalize(cross(v1 - v0, v2 - v0));

        // shading normal
        const float3 n0 =
            transform_normal(world_to_object, scene.normals[idx.x]);
        const float3 n1 =
            transform_normal(world_to_object, scene.normals[idx.y]);
        const float3 n2 =
            transform_normal(world_to_object, scene.normals[idx.z]);
        n_s = normalize((1.0f - barycentric.x - barycentric.y) * n0 +
                        barycentric.x * n1 + barycentric.y * n2);

        // texcoord
        const float2 tex0 = scene.texcoords[idx.x];
        const float2 tex1 = scene.texcoords[idx.y];
        const float2 tex2 = scene.texcoords[idx.z];
        texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                   barycentric.x * tex1 + barycentric.y * tex2;

        area = 0.5f * length(cross(v1 - v0, v2 - v0));

        // flip normal
        is_entering = dot(-direction, n_g) > 0;
        n_s = is_entering > 0 ? n_s : -n_s;
        n_g = is_entering ? n_g : -n_g;

        // compute tangent and bitangent vector
        orthonormal_basis(n_s, tangent, bitangent);

        normal_mapping(scene.textures, material);
    }

   private:
    CUDA_INLINE CUDA_DEVICE void bump_mapping(const TextureHeader* textures,
                                              const Material& material)
    {
        if (material.heightmap_texture_id != FRED_INVALID_ID)
        {
            const TextureHeader& heightmap =
                textures[material.heightmap_texture_id];
            const float du = 1.0f / heightmap.width;
            const float dv = 1.0f / heightmap.height;
            const float v = heightmap.sample<uchar4>(texcoord).x;
            const float dfdu =
                heightmap.sample<uchar4>(texcoord + make_float2(du, 0)).x - v;
            const float dfdv =
                heightmap.sample<uchar4>(texcoord + make_float2(0, dv)).x - v;
            tangent = normalize(tangent + dfdu * n_s);
            bitangent = normalize(bitangent + dfdv * n_s);
            n_s = normalize(cross(tangent, bitangent));
        }
    }

    CUDA_INLINE CUDA_DEVICE void normal_mapping(const TextureHeader* textures,
                                                const Material& material)
    {
        if (material.normalmap_texture_id != FRED_INVALID_ID)
        {
            float3 v = make_float3(
                textures[material.normalmap_texture_id].sample<uchar4>(
                    texcoord));
            v = 2.0f * v - 1.0f;
            n_s = normalize(local_to_world(v, tangent, bitangent, n_s));
            orthonormal_basis(n_s, tangent, bitangent);
        }
    }
};

struct ShadingParams
{
    float diffuse = 1.0f;
    float3 base_color = make_float3(0, 0, 0);
    float diffuse_roughness = 0.0f;

    float specular = 1.0f;
    float3 specular_color = make_float3(0, 0, 0);
    float specular_roughness = 0.2f;

    float metalness = 0.0f;

    float coat = 0.0f;
    float3 coat_color = make_float3(1, 1, 1);
    float coat_roughness = 0.1f;

    float transmission = 0;
    float3 transmission_color = make_float3(1, 1, 1);

    float sheen = 0.0f;
    float3 sheen_color = make_float3(1.0f, 1.0f, 1.0f);
    float sheen_roughness = 0.3f;

    float subsurface = 0;
    float3 subsurface_color = make_float3(1.0f, 1.0f, 1.0f);

    float thin_walled = 0.0f;

    CUDA_INLINE CUDA_DEVICE ShadingParams(const Material& material,
                                          const float2& texcoord,
                                          const TextureHeader* textures)
    {
        // diffuse
        diffuse = clamp(material.diffuse, 0.0f, 1.0f);

        // diffuse roughness
        diffuse_roughness = clamp(material.diffuse_roughness, 0.01f, 1.0f);

        // diffuse color
        base_color = clamp(material.get_diffuse_color(textures, texcoord),
                           make_float3(0.01f), make_float3(0.99f));

        // specular
        specular = clamp(material.specular, 0.0f, 1.0f);

        // specular color
        specular_color = clamp(material.get_specular_color(textures, texcoord),
                               make_float3(0.01f), make_float3(0.99f));

        // specular roughness
        specular_roughness = clamp(
            material.get_specular_roughness(textures, texcoord), 0.01f, 1.0f);

        // metallic roughness
        // TODO: implement this in Material
        if (material.metallic_roughness_texture_id != FRED_INVALID_ID)
        {
            const float4 mr =
                textures[material.metallic_roughness_texture_id].sample<uchar4>(
                    texcoord);
            specular_roughness = clamp(srgb_to_linear(mr.y), 0.01f, 1.0f);
            metalness = clamp(srgb_to_linear(mr.z), 0.0f, 1.0f);
        }

        // metalness
        metalness =
            clamp(material.get_metalness(textures, texcoord), 0.0f, 1.0f);

        specular_roughness = 0.01f;
        metalness = 0.0f;

        // coat
        coat = clamp(material.get_coat(textures, texcoord), 0.0f, 1.0f);

        // coat roughness
        coat_roughness =
            clamp(material.get_coat_roughness(textures, texcoord), 0.01f, 1.0f);

        // transmission
        transmission =
            clamp(material.get_transmission(textures, texcoord), 0.0f, 1.0f);

        // transmission color
        transmission_color = material.transmission_color;

        // sheen
        sheen = clamp(material.sheen, 0.0f, 1.0f);

        // sheen roughness
        sheen_roughness = clamp(material.sheen_roughness, 0.01f, 1.0f);

        // subsurface
        subsurface = clamp(material.subsurface, 0.0f, 1.0f);

        // subsurface color
        subsurface_color = material.subsurface_color;

        // thin walled
        thin_walled = clamp(material.thin_walled, 0.0f, 1.0f);
    }
};

static CUDA_INLINE CUDA_DEVICE float3 fetch_envmap(const TextureHeader& envmap,
                                                   const float3& v)
{
    const float2 thphi = cartesian_to_spherical(v);
    const float2 uv = make_float2(thphi.y / (2.0f * M_PIf), thphi.x / M_PIf);
    return make_float3(envmap.sample<float3>(uv));
}

static CUDA_INLINE CUDA_DEVICE float3 fetch_arhosek(const ArhosekSky& arhosek,
                                                    const float3& v)
{
    const float2 thphi = cartesian_to_spherical(v);
    const float gamma = acos(dot(arhosek.sun_direction, v));
    return arhosek.intensity *
           make_float3(arhosek_tristim_skymodel_radiance(arhosek.state, thphi.x,
                                                         gamma, 0),
                       arhosek_tristim_skymodel_radiance(arhosek.state, thphi.x,
                                                         gamma, 1),
                       arhosek_tristim_skymodel_radiance(arhosek.state, thphi.x,
                                                         gamma, 2));
}