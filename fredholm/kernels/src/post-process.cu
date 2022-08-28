#include "kernels/post-process.h"
#include "sutil/vec_math.h"

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
                               float4* beauty_out, float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  const float4 beauty = beauty_in[image_idx];
  const float4 denoised = denoised_in[image_idx];

  const float beauty_luminance = rgb_to_luminance(make_float3(beauty));
  const float denoised_luminance = rgb_to_luminance(make_float3(denoised));

  beauty_out[image_idx] = beauty_luminance > 1.0f ? beauty : make_float4(0.0f);
  denoised_out[image_idx] =
      denoised_luminance > 1.0f ? denoised : make_float4(0.0f);
}

__global__ void bloom_kernel_1(const float4* beauty_in,
                               const float4* denoised_in,
                               const float4* beauty_high_luminance,
                               const float4* denoised_high_luminance, int width,
                               int height, float4* beauty_out,
                               float4* denoised_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int image_idx = i + width * j;

  const float4 b0 = beauty_in[image_idx];
  const float4 d0 = denoised_in[image_idx];

  const int K = 8;
  const float sigma = 1.0f;

  float4 b_sum = make_float4(0.0f);
  float4 d_sum = make_float4(0.0f);
  float w_sum = 0.0f;
  for (int v = -K; v <= K; ++v) {
    for (int u = -K; u <= K; ++u) {
      const int x = clamp(i + u, 0, width - 1);
      const int y = clamp(j + v, 0, height - 1);

      const float4 b1 = beauty_high_luminance[x + width * j];
      const float4 d1 = denoised_high_luminance[x + width * j];

      const float dist2 = u * u + v * v;
      const float h = expf(-dist2 / (2.0f * sigma));

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

  float3 color = make_float3(beauty_in[image_idx]);

  // beauty
  const float EV100 = compute_EV100(1.0f, 1.0f, ISO);
  const float exposure = convert_EV100_to_exposure(EV100);
  color *= exposure;
  // color = aces_tone_mapping(color);
  color = uchimura(color);
  color = linear_to_srgb(color);
  beauty_out[image_idx] = make_float4(color, 1.0f);

  // denoised
  color = make_float3(denoised_in[image_idx]);
  color *= exposure;
  // color = aces_tone_mapping(color);
  color = uchimura(color);
  color = linear_to_srgb(color);
  denoised_out[image_idx] = make_float4(color, 1.0f);
}