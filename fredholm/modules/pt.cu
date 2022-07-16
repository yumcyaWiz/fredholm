#include <optix.h>

#include "bsdf.cu"
#include "fredholm/shared.h"
#include "math.cu"
#include "sampling.cu"
#include "sutil/vec_math.h"

#define RAY_EPS 0.001f

using namespace fredholm;

extern "C" {
__constant__ LaunchParams params;
}

struct RadiancePayload {
  float3 origin;
  float3 direction;

  float3 throughput = make_float3(1);
  float3 radiance = make_float3(0);

  RNGState rng;

  bool done = false;

  bool firsthit = true;
  float3 position = make_float3(0);
  float3 normal = make_float3(0);
  float depth = 0;
  float2 texcoord = make_float2(0);
  float3 albedo = make_float3(0);
};

struct ShadowPayload {
  bool visible = false;  // light visibility
};

struct LightPayload {
  float3 direction;
  float3 le = make_float3(0.0f);
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

// trace radiance ray
static __forceinline__ __device__ void trace_radiance(
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    RadiancePayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_RADIANCE), u0, u1);
}

static __forceinline__ __device__ void trace_shadow(
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    ShadowPayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             static_cast<unsigned int>(RayType::RAY_TYPE_SHADOW),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_SHADOW), u0, u1);
}

static __forceinline__ __device__ void trace_light(
    OptixTraversableHandle& handle, const float3& ray_origin,
    const float3& ray_direction, float tmin, float tmax,
    LightPayload* payload_ptr)
{
  unsigned int u0, u1;
  pack_ptr(payload_ptr, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f,
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             static_cast<unsigned int>(RayType::RAY_TYPE_LIGHT),
             static_cast<unsigned int>(RayType::RAY_TYPE_COUNT),
             static_cast<unsigned int>(RayType::RAY_TYPE_LIGHT), u0, u1);
}

static __forceinline__ __device__ bool has_emission(const Material& material)
{
  return (material.emission_color.x > 0 || material.emission_color.y > 0 ||
          material.emission_color.z > 0);
}

static __forceinline__ __device__ void sample_ray_pinhole_camera(
    const float2& uv, float3& origin, float3& direction, float& pdf)
{
  const float3 p_sensor = params.camera.origin + uv.x * params.camera.right +
                          uv.y * params.camera.up;
  const float3 p_pinhole =
      params.camera.origin + params.camera.f * params.camera.forward;

  origin = p_pinhole;
  direction = normalize(p_pinhole - p_sensor);
  pdf = 1.0f / dot(direction, params.camera.forward);
}

static __forceinline__ __device__ void fill_surface_info(
    const float3* vertices, const uint3* indices, const float3* normals,
    const float2* texcoords, const float3& ray_origin,
    const float3& ray_direction, float ray_tmax, const float2& barycentric,
    uint prim_idx, SurfaceInfo& info)
{
  info.t = ray_tmax;
  info.x = ray_origin + ray_tmax * ray_direction;
  info.barycentric = barycentric;

  const uint3 idx = indices[prim_idx];
  const float3 v0 = vertices[idx.x];
  const float3 v1 = vertices[idx.y];
  const float3 v2 = vertices[idx.z];
  info.n_g = normalize(cross(v1 - v0, v2 - v0));

  const float3 n0 = normals[idx.x];
  const float3 n1 = normals[idx.y];
  const float3 n2 = normals[idx.z];
  info.n_s = normalize((1.0f - info.barycentric.x - info.barycentric.y) * n0 +
                       info.barycentric.x * n1 + info.barycentric.y * n2);

  const float2 tex0 = texcoords[idx.x];
  const float2 tex1 = texcoords[idx.y];
  const float2 tex2 = texcoords[idx.z];
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
    const cudaTextureObject_t* textures, ShadingParams& shading_params)
{
  // base color
  shading_params.base_color =
      material.base_color_texture_id >= 0
          ? make_float3(tex2D<float4>(textures[material.base_color_texture_id],
                                      surf_info.texcoord.x,
                                      surf_info.texcoord.y))
          : material.base_color;

  // specular
  shading_params.specular = material.specular;

  // specular color
  shading_params.specular_color =
      material.specular_color_texture_id >= 0
          ? make_float3(
                tex2D<float4>(textures[material.specular_color_texture_id],
                              surf_info.texcoord.x, surf_info.texcoord.y))
          : material.specular_color;

  // specular roughness
  shading_params.specular_roughness =
      material.specular_roughness_texture_id >= 0
          ? tex2D<float4>(textures[material.specular_roughness_texture_id],
                          surf_info.texcoord.x, surf_info.texcoord.y)
                .x
          : material.specular_roughness;

  // metalness
  shading_params.metalness =
      material.metalness_texture_id >= 0
          ? tex2D<float4>(textures[material.metalness_texture_id],
                          surf_info.texcoord.x, surf_info.texcoord.y)
                .x
          : material.metalness;

  // coat
  shading_params.coat = material.coat;

  // coat roughness
  shading_params.coat_roughness = material.coat_roughness;

  // transmission
  shading_params.transmission = material.transmission;

  // transmission color
  shading_params.transmission_color = material.transmission_color;
}

static __forceinline__ __device__ float3
sample_position_on_light(const float u, const float2& v, const float3* vertices,
                         const uint3* indices, const float3* normals,
                         float3& le, float3& n, float& pdf)
{
  // sample light
  const uint light_idx =
      clamp(static_cast<uint>(params.n_lights * u), 0u, params.n_lights - 1);
  const Light& light = params.lights[light_idx];

  // sample point on the light
  const float2 barycentric = sample_triangle(v);

  const uint3 idx = light.indices;
  const float3 v0 = vertices[idx.x];
  const float3 v1 = vertices[idx.y];
  const float3 v2 = vertices[idx.z];
  const float3 n0 = normals[idx.x];
  const float3 n1 = normals[idx.y];
  const float3 n2 = normals[idx.z];

  const float3 p = (1.0f - barycentric.x - barycentric.y) * v0 +
                   barycentric.x * v1 + barycentric.y * v2;
  n = (1.0f - barycentric.x - barycentric.y) * n0 + barycentric.x * n1 +
      barycentric.y * n2;
  const float area = 0.5f * length(cross(v1 - v0, v2 - v0));

  le = light.le;
  pdf = 1.0f / (params.n_lights * area);

  return p;
}

static __forceinline__ __device__ float3 fetch_ibl(const float3& v)
{
  const float2 thphi = cartesian_to_spherical(v);
  return make_float3(
      tex2D<float4>(params.ibl, thphi.y / (2.0f * M_PIf), thphi.x / M_PIf));
}

static __forceinline__ __device__ float compute_mis_weight(float pdf0,
                                                           float pdf1)
{
  return pdf0 / (pdf0 + pdf1);
}

extern "C" __global__ void __raygen__rg()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  const uint image_idx = idx.x + params.width * idx.y;

  // set RNG state
  RadiancePayload payload;
  payload.rng.state = params.rng_states[image_idx].state;
  payload.rng.inc = params.rng_states[image_idx].inc;

  uint n_spp = params.sample_count[image_idx];
  float3 beauty = make_float3(params.render_layer.beauty[image_idx]);
  float3 position = make_float3(params.render_layer.position[image_idx]);
  float3 normal = make_float3(params.render_layer.normal[image_idx]);
  float depth = params.render_layer.depth[image_idx];
  float2 texcoord = make_float2(params.render_layer.texcoord[image_idx]);
  float3 albedo = make_float3(params.render_layer.albedo[image_idx]);

  for (int spp = 0; spp < params.n_samples; ++spp) {
    // generate initial ray from camera
    float2 uv =
        make_float2((2.0f * (idx.x + frandom(payload.rng)) - dim.x) / dim.y,
                    (2.0f * (idx.y + frandom(payload.rng)) - dim.y) / dim.y);
    // flip x
    uv.x = -uv.x;
    float camera_pdf;
    sample_ray_pinhole_camera(uv, payload.origin, payload.direction,
                              camera_pdf);

    // start ray tracing from the camera
    payload.radiance = make_float3(0);
    payload.throughput =
        make_float3(dot(payload.direction, params.camera.forward) / camera_pdf);
    payload.done = false;
    for (int ray_depth = 0; ray_depth < params.max_depth; ++ray_depth) {
      // russian roulette
      const float russian_roulette_prob =
          ray_depth == 0
              ? 1.0f
              : clamp(rgb_to_luminance(payload.throughput), 0.0f, 1.0f);
      if (frandom(payload.rng) >= russian_roulette_prob) { break; }
      payload.throughput /= russian_roulette_prob;

      // trace ray and update payloads
      trace_radiance(params.ias_handle, payload.origin, payload.direction, 0.0f,
                     1e9f, &payload);

      // throughput nan check
      if (isnan(payload.throughput) || isinf(payload.throughput)) { break; }

      if (payload.done) { break; }
    }

    // radiance nan check
    float3 radiance = make_float3(0.0f);
    if (!isnan(payload.radiance) && !isinf(payload.radiance)) {
      radiance = payload.radiance;
    }

    // take streaming average
    const float coef = 1.0f / (n_spp + 1.0f);
    beauty = coef * (n_spp * beauty + radiance);
    position = coef * (n_spp * position + payload.position);
    normal = coef * (n_spp * normal + payload.normal);
    depth = coef * (n_spp * depth + payload.depth);
    texcoord = coef * (n_spp * texcoord + payload.texcoord);
    albedo = coef * (n_spp * albedo + payload.albedo);

    n_spp++;
  }

  // update total number of samples
  params.sample_count[image_idx] = n_spp;

  // save RNG state for next render call
  params.rng_states[image_idx].state = payload.rng.state;
  params.rng_states[image_idx].inc = payload.rng.inc;

  // write results in render layers
  params.render_layer.beauty[image_idx] = make_float4(beauty, 1.0f);
  params.render_layer.position[image_idx] = make_float4(position, 1.0f);
  params.render_layer.normal[image_idx] = make_float4(normal, 1.0f);
  params.render_layer.depth[image_idx] = depth;
  params.render_layer.texcoord[image_idx] = make_float4(texcoord, 0.0f, 1.0f);
  params.render_layer.albedo[image_idx] = make_float4(albedo, 1.0f);
}

extern "C" __global__ void __miss__radiance()
{
  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

  // firsthit light case
  if (payload->firsthit) {
    float3 le;
    if (params.ibl) {
      le = fetch_ibl(payload->direction);
    } else {
      le = params.bg_color;
    }

    payload->radiance += payload->throughput * le;
  }

  payload->done = true;
}

extern "C" __global__ void __miss__shadow()
{
  ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
  payload->visible = true;
}

extern "C" __global__ void __miss__light()
{
  LightPayload* payload = get_payload_ptr<LightPayload>();

  if (params.ibl) {
    payload->le = fetch_ibl(payload->direction);
  } else {
    payload->le = params.bg_color;
  }
}

extern "C" __global__ void __anyhit__radiance()
{
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
  const uint prim_idx = optixGetPrimitiveIndex();

  // get material info
  const uint material_id = sbt->material_ids[prim_idx];
  const Material& material = params.materials[material_id];

  // fill surface info
  const float2 barycentric = optixGetTriangleBarycentrics();

  // calc texcoord
  const uint3 idx = sbt->indices[prim_idx];
  const float2 tex0 = sbt->texcoords[idx.x];
  const float2 tex1 = sbt->texcoords[idx.y];
  const float2 tex2 = sbt->texcoords[idx.z];
  const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                          barycentric.x * tex1 + barycentric.y * tex2;

  // fetch base color texture
  if (material.base_color_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.base_color_texture_id],
                      texcoord.x, texcoord.y)
            .w;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }

  // fetch alpha texture
  if (material.alpha_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.alpha_texture_id], texcoord.x,
                      texcoord.y)
            .x;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }
}

extern "C" __global__ void __anyhit__shadow()
{
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
  const uint prim_idx = optixGetPrimitiveIndex();

  // get material info
  const uint material_id = sbt->material_ids[prim_idx];
  const Material& material = params.materials[material_id];

  // fill surface info
  const float2 barycentric = optixGetTriangleBarycentrics();

  // calc texcoord
  const uint3 idx = sbt->indices[prim_idx];
  const float2 tex0 = sbt->texcoords[idx.x];
  const float2 tex1 = sbt->texcoords[idx.y];
  const float2 tex2 = sbt->texcoords[idx.z];
  const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                          barycentric.x * tex1 + barycentric.y * tex2;

  // fetch base color texture
  if (material.base_color_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.base_color_texture_id],
                      texcoord.x, texcoord.y)
            .w;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }

  // fetch alpha texture
  if (material.alpha_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.alpha_texture_id], texcoord.x,
                      texcoord.y)
            .x;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }
}

extern "C" __global__ void __anyhit__light()
{
  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
  const uint prim_idx = optixGetPrimitiveIndex();

  // get material info
  const uint material_id = sbt->material_ids[prim_idx];
  const Material& material = params.materials[material_id];

  // fill surface info
  const float2 barycentric = optixGetTriangleBarycentrics();

  // calc texcoord
  const uint3 idx = sbt->indices[prim_idx];
  const float2 tex0 = sbt->texcoords[idx.x];
  const float2 tex1 = sbt->texcoords[idx.y];
  const float2 tex2 = sbt->texcoords[idx.z];
  const float2 texcoord = (1.0f - barycentric.x - barycentric.y) * tex0 +
                          barycentric.x * tex1 + barycentric.y * tex2;

  // fetch base color texture
  if (material.base_color_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.base_color_texture_id],
                      texcoord.x, texcoord.y)
            .w;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }

  // fetch alpha texture
  if (material.alpha_texture_id >= 0) {
    const float alpha =
        tex2D<float4>(params.textures[material.alpha_texture_id], texcoord.x,
                      texcoord.y)
            .x;

    // ignore intersection
    if (alpha < 0.5) { optixIgnoreIntersection(); }
  }
}

extern "C" __global__ void __closesthit__radiance()
{
  RadiancePayload* payload = get_payload_ptr<RadiancePayload>();

  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
  const uint prim_idx = optixGetPrimitiveIndex();

  // get material info
  const uint material_id = sbt->material_ids[prim_idx];
  const Material& material = params.materials[material_id];

  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float ray_tmax = optixGetRayTmax();
  const float2 barycentric = optixGetTriangleBarycentrics();

  SurfaceInfo surf_info;
  fill_surface_info(sbt->vertices, sbt->indices, sbt->normals, sbt->texcoords,
                    ray_origin, ray_direction, ray_tmax, barycentric, prim_idx,
                    surf_info);

  ShadingParams shading_params;
  fill_shading_params(material, surf_info, params.textures, shading_params);

  // fill position, normal, depth, albedo
  if (payload->firsthit) {
    payload->position = surf_info.x;
    payload->normal = surf_info.n_s;
    payload->depth = surf_info.t;
    payload->texcoord = surf_info.texcoord;
    payload->albedo = shading_params.base_color;
    payload->firsthit = false;

    // first hit light case
    if (has_emission(material)) {
      payload->radiance += payload->throughput * material.emission_color;
      payload->done = true;
      return;
    }
  }

  // normal mapping
  float3 tangent = surf_info.tangent;
  float3 normal = surf_info.n_s;
  float3 bitangent = surf_info.bitangent;
  if (material.normalmap_texture_id >= 0) {
    float3 value = make_float3(
        tex2D<float4>(params.textures[material.normalmap_texture_id],
                      surf_info.texcoord.x, surf_info.texcoord.y));
    value = normalize(0.5f * (value + 1.0f));
    normal = local_to_world(value, surf_info.tangent, surf_info.n_s,
                            surf_info.bitangent);
    orthonormal_basis(normal, tangent, bitangent);
  }

  const float3 wo = world_to_local(-ray_direction, tangent, normal, bitangent);
  const BSDF bsdf = BSDF(shading_params, surf_info.is_entering);

  // light sampling
  {
    // sky
    if (params.ibl) {
      // TODO: implement IBL importance sampling
      const float3 wi = sample_cosine_weighted_hemisphere(
          make_float2(frandom(payload->rng), frandom(payload->rng)));
      const float3 shadow_ray_origin = surf_info.x + RAY_EPS * surf_info.n_g;
      const float3 shadow_ray_direction =
          local_to_world(wi, tangent, normal, bitangent);

      ShadowPayload shadow_payload;
      trace_shadow(params.ias_handle, shadow_ray_origin, shadow_ray_direction,
                   0.0f, 1e9f, &shadow_payload);

      if (shadow_payload.visible) {
        const float3 f = bsdf.eval(wo, wi);
        const float pdf = abs_cos_theta(wi) / M_PIf;
        payload->radiance += payload->throughput * f * abs_cos_theta(wi) *
                             fetch_ibl(shadow_ray_direction) / pdf;
      }
    } else {
      const float3 wi = sample_cosine_weighted_hemisphere(
          make_float2(frandom(payload->rng), frandom(payload->rng)));
      const float3 shadow_ray_origin = surf_info.x + RAY_EPS * surf_info.n_g;
      const float3 shadow_ray_direction =
          local_to_world(wi, tangent, normal, bitangent);

      ShadowPayload shadow_payload;
      trace_shadow(params.ias_handle, shadow_ray_origin, shadow_ray_direction,
                   0.0f, 1e9f, &shadow_payload);

      if (shadow_payload.visible) {
        const float3 f = bsdf.eval(wo, wi);
        const float pdf = abs_cos_theta(wi) / M_PIf;
        payload->radiance +=
            payload->throughput * f * abs_cos_theta(wi) * params.bg_color / pdf;
      }
    }

    // area light
    if (params.n_lights > 0) {
      float3 le, n;
      float pdf_area;
      const float3 p = sample_position_on_light(
          frandom(payload->rng),
          make_float2(frandom(payload->rng), frandom(payload->rng)),
          sbt->vertices, sbt->indices, sbt->normals, le, n, pdf_area);

      const float3 shadow_ray_origin = surf_info.x + RAY_EPS * surf_info.n_g;
      const float3 shadow_ray_direction = normalize(p - shadow_ray_origin);
      const float r = length(p - shadow_ray_origin);

      ShadowPayload shadow_payload;
      trace_shadow(params.ias_handle, shadow_ray_origin, shadow_ray_direction,
                   0.0f, r - RAY_EPS, &shadow_payload);

      if (shadow_payload.visible) {
        const float3 wi =
            world_to_local(shadow_ray_direction, tangent, normal, bitangent);
        const float3 f = bsdf.eval(wo, wi);
        const float pdf =
            r * r / fabs(dot(-shadow_ray_direction, n)) * pdf_area;
        payload->radiance +=
            payload->throughput * f * abs_cos_theta(wi) * le / pdf;
      }
    }
  }

  // BSDF sampling
  {
    const float4 u = make_float4(frandom(payload->rng), frandom(payload->rng),
                                 frandom(payload->rng), frandom(payload->rng));
    const float2 v = make_float2(frandom(payload->rng), frandom(payload->rng));
    float3 f;
    float pdf;
    const float3 wi = bsdf.sample(wo, u, v, f, pdf);

    const float3 light_ray_origin = surf_info.x + RAY_EPS * surf_info.n_g;
    const float3 light_ray_direction =
        local_to_world(wi, tangent, normal, bitangent);

    LightPayload light_payload;
    light_payload.direction = light_ray_direction;
    trace_light(params.ias_handle, light_ray_origin, light_ray_direction, 0.0f,
                1e9f, &light_payload);

    payload->radiance +=
        payload->throughput * f * abs_cos_theta(wi) * light_payload.le / pdf;
  }

  // generate next ray direction
  {
    const float4 u = make_float4(frandom(payload->rng), frandom(payload->rng),
                                 frandom(payload->rng), frandom(payload->rng));
    const float2 v = make_float2(frandom(payload->rng), frandom(payload->rng));
    float3 f;
    float pdf;
    const float3 wi = bsdf.sample(wo, u, v, f, pdf);
    const float3 wi_world = local_to_world(wi, tangent, normal, bitangent);

    // update throughput
    payload->throughput *= f * abs_cos_theta(wi) / pdf;

    // advance ray
    payload->origin = surf_info.x;
    payload->direction = wi_world;

    // adjust ray origin to prevent self-intersection
    const bool is_transmitted = dot(wi_world, surf_info.n_g) < 0;
    if (is_transmitted) {
      payload->origin -= RAY_EPS * surf_info.n_g;
    } else {
      payload->origin += RAY_EPS * surf_info.n_g;
    }
  }
}

extern "C" __global__ void __closesthit__shadow()
{
  ShadowPayload* payload = get_payload_ptr<ShadowPayload>();
  payload->visible = false;
}

extern "C" __global__ void __closesthit__light()
{
  LightPayload* payload = get_payload_ptr<LightPayload>();

  const HitGroupSbtRecordData* sbt =
      reinterpret_cast<HitGroupSbtRecordData*>(optixGetSbtDataPointer());
  const uint prim_idx = optixGetPrimitiveIndex();

  // get material info
  const uint material_id = sbt->material_ids[prim_idx];
  const Material& material = params.materials[material_id];

  if (has_emission(material)) {
    payload->le = material.emission_color;
  } else {
    payload->le = make_float3(0.0f);
  }
}