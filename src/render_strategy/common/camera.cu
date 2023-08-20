#pragma once

#include "helper_math.h"
#include "sampling.cu"
#include "shared.h"

using namespace fredholm;

static CUDA_INLINE CUDA_DEVICE void sample_ray_pinhole_camera(
    const CameraParams& params, const float2& uv, float3& origin,
    float3& direction, float& pdf)
{
    const float f = 1.0f / tanf(0.5f * params.fov);
    const float3 p_sensor = make_float3(uv.x, uv.y, 0);
    const float3 p_pinhole = make_float3(0, 0, f);

    origin = transform_position(params.transform, p_pinhole);
    float3 dir = normalize(p_pinhole - p_sensor);
    // TODO: remove this adhoc fix
    dir.z *= -1.0f;
    direction = transform_direction(params.transform, dir);
    pdf = 1.0f / abs(dir.z);
}

static CUDA_INLINE CUDA_DEVICE void sample_ray_thinlens_camera(
    const CameraParams& params, const float2& uv, const float2& u,
    float3& origin, float3& direction, float& pdf)
{
    const float f = 1.0f / tanf(0.5f * params.fov);

    const float b = params.focus;
    const float a = 1.0f / (1.0f + f - 1.0f / b);
    const float lens_radius = 2.0f * f / params.F;

    const float3 p_sensor = make_float3(uv.x, uv.y, 0);
    const float3 p_lens_center = make_float3(0, 0, f);

    const float2 p_disk = lens_radius * sample_concentric_disk(u);
    const float3 p_lens = p_lens_center + make_float3(p_disk.x, p_disk.y, 0);
    const float3 sensor_to_lens = normalize(p_lens - p_sensor);

    const float3 sensor_to_lens_center = normalize(p_lens_center - p_sensor);
    const float3 p_object =
        p_sensor + ((a + b) / sensor_to_lens_center.z) * sensor_to_lens_center;

    origin = transform_position(params.transform, p_lens);
    float3 dir = normalize(p_object - p_lens);
    // TODO: remove this adhoc fix
    dir.z *= -1.0f;
    direction = transform_direction(params.transform, dir);
    const float pdf_area = 1.0f / (M_PIf * lens_radius * lens_radius);
    // pdf = length2(p_lens - p_sensor) / abs(sensor_to_lens.z) * pdf_area;
    pdf = 1.0f / (dir.z * dir.z);
}