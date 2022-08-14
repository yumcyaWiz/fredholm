#include "kernels/post-process.h"

__global__ void post_process_kernel(const float4* beauty_in,
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
  color = aces_tone_mapping(color);
  color = linear_to_srgb(color);
  beauty_out[image_idx] = make_float4(color, 1.0f);

  // denoised
  color = make_float3(denoised_in[image_idx]);
  color *= exposure;
  color = aces_tone_mapping(color);
  color = linear_to_srgb(color);
  denoised_out[image_idx] = make_float4(color, 1.0f);
}