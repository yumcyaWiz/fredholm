#pragma once
#ifndef __CUDACC__
#include <cuda_runtime.h>
#include <optix.h>

#include "helper_math.h"
#endif

#include "arhosek.h"
#include "cuda_util.h"
#include "helper_math.h"
#include "optix_util.h"

namespace fredholm
{

struct Matrix3x4
{
    // 0 - (a00 a01 a02 a03)
    // 1 - (a10 a11 a12 a13)
    // 2 - (a20 a21 a22 a23)
    // 3 - (0   0   0   1  )
    float4 m[3];

    static CUDA_INLINE CUDA_HOST_DEVICE Matrix3x4 identity()
    {
        Matrix3x4 ret;
        ret.m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        ret.m[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        ret.m[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        return ret;
    }
};

CUDA_INLINE CUDA_HOST_DEVICE Matrix3x4 make_mat3x4(const float4& r0,
                                                   const float4& r1,
                                                   const float4& r2)
{
    Matrix3x4 m;
    m.m[0] = r0;
    m.m[1] = r1;
    m.m[2] = r2;
    return m;
}

CUDA_INLINE CUDA_HOST_DEVICE float3 transform_position(const Matrix3x4& m,
                                                       const float3& p)
{
    const float4 v = make_float4(p.x, p.y, p.z, 1.0f);
    return make_float3(dot(m.m[0], v), dot(m.m[1], v), dot(m.m[2], v));
}

CUDA_INLINE CUDA_HOST_DEVICE float3 transform_direction(const Matrix3x4& m,
                                                        const float3& v)
{
    const float4 t = make_float4(v.x, v.y, v.z, 0.0f);
    return make_float3(dot(m.m[0], t), dot(m.m[1], t), dot(m.m[2], t));
}

CUDA_INLINE CUDA_HOST_DEVICE float3 transform_normal(const Matrix3x4& m,
                                                     const float3& n)
{
    const float4 c0 = make_float4(m.m[0].x, m.m[1].x, m.m[2].x, 0.0f);
    const float4 c1 = make_float4(m.m[0].y, m.m[1].y, m.m[2].y, 0.0f);
    const float4 c2 = make_float4(m.m[0].z, m.m[1].z, m.m[2].z, 0.0f);
    const float4 t = make_float4(n.x, n.y, n.z, 0.0f);
    return make_float3(dot(c0, t), dot(c1, t), dot(c2, t));
}

// Duff, T., Burgess, J., Christensen, P., Hery, C., Kensler, A., Liani, M., &
// Villemin, R. (2017). Building an orthonormal basis, revisited. JCGT, 6(1).
CUDA_INLINE CUDA_HOST_DEVICE void orthonormal_basis(const float3& normal,
                                                    float3& tangent,
                                                    float3& bitangent)
{
    float sign = copysignf(1.0f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    tangent = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b,
                          -sign * normal.x);
    bitangent = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

CUDA_INLINE CUDA_DEVICE float3 world_to_local(const float3& v, const float3& t,
                                              const float3& n, const float3& b)
{
    return make_float3(dot(v, t), dot(v, n), dot(v, b));
}

CUDA_INLINE CUDA_DEVICE float3 local_to_world(const float3& v, const float3& t,
                                              const float3& n, const float3& b)
{
    return make_float3(v.x * t.x + v.y * n.x + v.z * b.x,
                       v.x * t.y + v.y * n.y + v.z * b.y,
                       v.x * t.z + v.y * n.z + v.z * b.z);
}

enum class RayType : unsigned int
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW = 1,
    RAY_TYPE_LIGHT = 2,
    RAY_TYPE_COUNT
};

struct CameraParams
{
    Matrix3x4 transform;
    float fov;
    float F;      // F number
    float focus;  // focus distance
};

struct PCGState
{
    unsigned long long state = 0;
    unsigned long long inc = 1;
};

struct SobolState
{
    unsigned long long index = 0;
    unsigned int dimension = 0;
    unsigned int seed = 0;
};

struct CMJState
{
    unsigned long long n_spp = 0;
    unsigned int scramble = 0;
    unsigned int depth = 0;
    unsigned int image_idx = 0;
};

struct BlueNoiseState
{
    int pixel_i = 0;
    int pixel_j = 0;
    int index = 0;
    int dimension = 0;
};

struct SamplerState
{
    PCGState pcg_state;
    SobolState sobol_state;
    CMJState cmj_state;
    BlueNoiseState blue_noise_state;
};

// similar to arnold standard surface
// https://autodesk.github.io/standard-surface/
struct Material
{
    float diffuse = 1.0f;
    float3 base_color = make_float3(1, 1, 1);
    int base_color_texture_id = -1;
    float diffuse_roughness = 0.0f;

    float specular = 1.0f;
    float3 specular_color = make_float3(1, 1, 1);
    int specular_color_texture_id = -1;
    float specular_roughness = 0.2f;
    int specular_roughness_texture_id = -1;

    float metalness = 0;
    int metalness_texture_id = -1;

    int metallic_roughness_texture_id = -1;

    float coat = 0;
    int coat_texture_id = -1;
    float3 coat_color = make_float3(1, 1, 1);
    float coat_roughness = 0.1;
    int coat_roughness_texture_id = -1;

    float transmission = 0;
    float3 transmission_color = make_float3(1, 1, 1);

    float sheen = 0.0f;
    float3 sheen_color = make_float3(1.0f, 1.0f, 1.0f);
    float sheen_roughness = 0.3f;

    float subsurface = 0;
    float3 subsurface_color = make_float3(1.0f, 1.0f, 1.0f);

    float thin_walled = 0.0f;

    float emission = 0;
    float3 emission_color = make_float3(0, 0, 0);
    int emission_texture_id = -1;

    int heightmap_texture_id = -1;
    int normalmap_texture_id = -1;
    int alpha_texture_id = -1;
};

struct TextureHeader
{
    uint width;
    uint height;
    CUdeviceptr data;

    // TODO: implement interpolation
    CUDA_INLINE CUDA_DEVICE float4 sample(const float2& uv) const
    {
        const int i = clamp(static_cast<int>(uv.x * width), 0,
                            static_cast<int>(width - 1));
        const int j = clamp(static_cast<int>(uv.y * height), 0,
                            static_cast<int>(height - 1));

        const uchar4* ptr = reinterpret_cast<const uchar4*>(data);
        const uchar4 v = ptr[j * width + i];
        constexpr float c = 1.0f / 255.0f;
        return make_float4(c * v.x, c * v.y, c * v.z, c * v.w);
    }
};

struct AreaLight
{
    uint3 indices;
    uint material_id;
    uint instance_idx;  // instance id
};

struct DirectionalLight
{
    float3 le;        // emission
    float3 dir;       // direction
    float angle = 0;  // angle size
};

// TODO: maybe this could be removed and use SceneDevice instead?
struct SceneData
{
    float3* vertices;
    uint3* indices;
    float3* normals;
    float2* texcoords;
    Material* materials;
    TextureHeader* textures;
    uint* material_ids;
    uint* indices_offsets;
    uint* geometry_ids;
    Matrix3x4* object_to_worlds;
    Matrix3x4* world_to_objects;
};

struct RenderLayer
{
    float4* beauty;
    float4* position;
    float* depth;
    float4* normal;
    float4* texcoord;
    float4* albedo;
};

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static CUDA_INLINE CUDA_HOST_DEVICE uint pcg32_random_r(PCGState* rng)
{
    unsigned long long oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// https://www.shadertoy.com/view/XlGcRh
static CUDA_INLINE CUDA_HOST_DEVICE uint xxhash32(uint p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

static CUDA_INLINE CUDA_HOST_DEVICE uint xxhash32(const uint3& p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

static CUDA_INLINE CUDA_HOST_DEVICE uint xxhash32(const uint4& p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

}  // namespace fredholm