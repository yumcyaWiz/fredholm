#include "cwl/util.h"
#include "kernels/post-process.h"
#include "sutil/vec_math.h"

void __host__ post_process_kernel_launch(
    const float4* beauty_in, const float4* denoised_in,
    float4* beauty_high_luminance, float4* denoised_high_luminance,
    float4* beauty_temp, float4* denoised_temp, int width, int height,
    float bloom_threshold, float bloom_sigma, float ISO, float4* beauty_out,
    float4* denoised_out)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(max(width / threads_per_block.x, 1),
                    max(height / threads_per_block.y, 1));

  // extract high luminance pixels
  bloom_kernel_0<<<blocks, threads_per_block>>>(
      beauty_in, denoised_in, width, height, bloom_threshold,
      beauty_high_luminance, denoised_high_luminance);
  CUDA_SYNC_CHECK();

  // gaussian blur
  bloom_kernel_1<<<blocks, threads_per_block>>>(
      beauty_in, denoised_in, beauty_high_luminance, denoised_high_luminance,
      width, height, bloom_sigma, beauty_temp, denoised_temp);
  CUDA_SYNC_CHECK();

  // tone mapping
  tone_mapping_kernel<<<blocks, threads_per_block>>>(
      beauty_temp, denoised_temp, width, height, ISO, beauty_out, denoised_out);
}

void __host__ tone_mapping_kernel_launch(const float4* beauty_in,
                                         const float4* denoised_in, int width,
                                         int height, float ISO,
                                         float4* beauty_out,
                                         float4* denoised_out)
{
  const dim3 threads_per_block(16, 16);
  const dim3 blocks(max(width / threads_per_block.x, 1),
                    max(height / threads_per_block.y, 1));
  tone_mapping_kernel<<<blocks, threads_per_block>>>(
      beauty_in, denoised_in, width, height, ISO, beauty_out, denoised_out);
}

__global__ void bloom_kernel_0(const float4* beauty_in,
                               const float4* denoised_in, int width, int height,
                               float bloom_threshold, float4* beauty_out,
                               float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  const float4 beauty = beauty_in[image_idx];
  const float4 denoised = denoised_in[image_idx];

  const float beauty_luminance = rgb_to_luminance(make_float3(beauty));
  const float denoised_luminance = rgb_to_luminance(make_float3(denoised));

  beauty_out[image_idx] =
      beauty_luminance > bloom_threshold ? beauty : make_float4(0.0f);
  denoised_out[image_idx] =
      denoised_luminance > bloom_threshold ? denoised : make_float4(0.0f);
}

__global__ void bloom_kernel_1(const float4* beauty_in,
                               const float4* denoised_in,
                               const float4* beauty_high_luminance,
                               const float4* denoised_high_luminance, int width,
                               int height, float bloom_sigma,
                               float4* beauty_out, float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  const float4 b0 = beauty_in[image_idx];
  const float4 d0 = denoised_in[image_idx];

  const int K = 16;
  const float sigma = 1.0f;

  float4 b_sum = make_float4(0.0f);
  float4 d_sum = make_float4(0.0f);
  float w_sum = 0.0f;
  for (int v = -K; v <= K; ++v) {
    for (int u = -K; u <= K; ++u) {
      const int x = clamp(i + u, 0, width - 1);
      const int y = clamp(j + v, 0, height - 1);

      const float4 b1 = beauty_high_luminance[x + width * y];
      const float4 d1 = denoised_high_luminance[x + width * y];

      const float dist2 = u * u + v * v;
      const float h = expf(-dist2 / (2.0f * bloom_sigma));

      b_sum += h * b1;
      d_sum += h * d1;
      w_sum += h;
    }
  }

  beauty_out[image_idx] = b0 + b_sum / w_sum;
  denoised_out[image_idx] = d0 + d_sum / w_sum;
}

__global__ void tone_mapping_kernel(const float4* beauty_in,
                                    const float4* denoised_in, int width,
                                    int height, float ISO, float4* beauty_out,
                                    float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  // chromatic aberration
  const float2 uv = make_float2(static_cast<float>(i) / width,
                                static_cast<float>(j) / height);
  const float2 d = (uv - make_float2(0.5f)) * 0.0000075f * 0.2f;

  const float2 uv_r =
      clamp(uv - 0.0f * d, make_float2(0.0f), make_float2(1.0f));
  const float2 uv_g =
      clamp(uv - 1.0f * d, make_float2(0.0f), make_float2(1.0f));
  const float2 uv_b =
      clamp(uv - 2.0f * d, make_float2(0.0f), make_float2(1.0f));

  const int image_idx_r = uv_r.x * width + width * uv_r.y * height;
  const int image_idx_g = uv_g.x * width + width * uv_g.y * height;
  const int image_idx_b = uv_b.x * width + width * uv_b.y * height;

  // beauty
  float3 color = make_float3(beauty_in[image_idx_r].x, beauty_in[image_idx_g].y,
                             beauty_in[image_idx_b].z);
  const float EV100 = compute_EV100(1.0f, 1.0f, ISO);
  const float exposure = convert_EV100_to_exposure(EV100);
  color *= exposure;
  // color = aces_tone_mapping(color);
  color = uchimura(color);
  color = linear_to_srgb(color);
  beauty_out[image_idx] = make_float4(color, 1.0f);

  // denoised
  color = make_float3(denoised_in[image_idx_r].x, denoised_in[image_idx_g].y,
                      denoised_in[image_idx_b].z);
  color *= exposure;
  // color = aces_tone_mapping(color);
  color = uchimura(color);
  color = linear_to_srgb(color);
  denoised_out[image_idx] = make_float4(color, 1.0f);
}