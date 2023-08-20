#include "helper_math.h"

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
static __forceinline__ __device__ float rgb_to_luminance(const float3& rgb)
{
    return dot(rgb, make_float3(0.2126729f, 0.7151522f, 0.0721750f));
}

static __forceinline__ __device__ float3 linear_to_srgb(const float3& rgb)
{
    float3 ret;
    ret.x = rgb.x < 0.0031308 ? 12.92 * rgb.x
                              : 1.055 * pow(rgb.x, 1.0f / 2.4f) - 0.055;
    ret.y = rgb.y < 0.0031308 ? 12.92 * rgb.y
                              : 1.055 * pow(rgb.y, 1.0f / 2.4f) - 0.055;
    ret.z = rgb.z < 0.0031308 ? 12.92 * rgb.z
                              : 1.055 * pow(rgb.z, 1.0f / 2.4f) - 0.055;
    return ret;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
static __forceinline__ __device__ float3 aces_tone_mapping(const float3& color)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e),
                 make_float3(0.0f), make_float3(1.0f));
}

static __forceinline__ __device__ float step(float edge, float x)
{
    return (x < edge) ? 0.0f : 1.0f;
}

static __forceinline__ __device__ float3 step(float edge, const float3& v)
{
    return make_float3(step(edge, v.x), step(edge, v.y), step(edge, v.z));
}

// static __forceinline__ __device__ float smoothstep(float edge0, float edge1,
//                                                    float x)
// {
//     if (x < edge0)
//         return 0.0f;
//     else if (x > edge1)
//         return 1.0f;
//     else
//     {
//         x = (x - edge0) / (edge1 - edge0);
//         return x * x * (3.0f - 2.0f * x);
//     }
// }

static __forceinline__ __device__ float3 smoothstep(float edge0, float edge1,
                                                    const float3& v)
{
    return make_float3(smoothstep(edge0, edge1, v.x),
                       smoothstep(edge0, edge1, v.y),
                       smoothstep(edge0, edge1, v.z));
}

// Uchimura 2017, "HDR theory and practice"
// Math: https://www.desmos.com/calculator/gslcdxvipg
// Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
static __forceinline__ __device__ float3 uchimura(const float3& x, float P,
                                                  float a, float m, float l,
                                                  float c, float b)
{
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float3 w0 = 1.0f - smoothstep(0.0, m, x);
    float3 w2 = step(m + l0, x);
    float3 w1 = 1.0f - w0 - w2;

    float3 T =
        m * make_float3(powf(x.x / m, c), powf(x.y / m, c), powf(x.z / m, c)) +
        b;
    float3 S = P - (P - S1) * expf(CP * (x - S0));
    float3 L = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

static __forceinline__ __device__ float3 uchimura(const float3& x)
{
    const float P = 1.0;   // max display brightness
    const float a = 1.0;   // contrast
    const float m = 0.22;  // linear section start
    const float l = 0.4;   // linear section length
    const float c = 1.33;  // black
    const float b = 0.0;   // pedestal

    return uchimura(x, P, a, m, l, c, b);
}

// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
static __forceinline__ __device__ float compute_EV100(float aperture,
                                                      float shutter_time,
                                                      float ISO)
{
    return log2f(aperture * aperture / shutter_time * 100.0 / ISO);
}

static __forceinline__ __device__ float convert_EV100_to_exposure(float EV100)
{
    float maxLuminance = 1.2 * powf(2.0, EV100);
    return 1.0f / maxLuminance;
}

extern "C" __global__ void tone_mapping_kernel(uint width, uint height,
                                               float chromatic_aberration,
                                               float ISO, const float4* input,
                                               float4* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    const int image_idx = i + width * j;

    // chromatic aberration
    const float2 uv = make_float2(static_cast<float>(i) / width,
                                  static_cast<float>(j) / height);
    const float2 d =
        (uv - make_float2(0.5f)) / (width * height) * chromatic_aberration;

    const float2 uv_r =
        clamp(uv - 0.0f * d, make_float2(0.0f), make_float2(1.0f));
    const float2 uv_g =
        clamp(uv - 1.0f * d, make_float2(0.0f), make_float2(1.0f));
    const float2 uv_b =
        clamp(uv - 2.0f * d, make_float2(0.0f), make_float2(1.0f));

    const int image_idx_r = uv_r.x * width + width * (uv_r.y * height);
    const int image_idx_g = uv_g.x * width + width * (uv_g.y * height);
    const int image_idx_b = uv_b.x * width + width * (uv_b.y * height);

    float3 color = make_float3(input[image_idx_r].x, input[image_idx_g].y,
                               input[image_idx_b].z);

    // exposure adjustment
    const float EV100 = compute_EV100(1.0f, 1.0f, ISO);
    const float exposure = convert_EV100_to_exposure(EV100);
    color *= exposure;

    // tone mapping
    // color = aces_tone_mapping(color);
    color = uchimura(color);

    // linear to sRGB conversion
    color = linear_to_srgb(color);

    output[image_idx] = make_float4(color, 1.0f);
}