#pragma once

__global__ void post_process_kernel(const float4* beauty_in,
                                    const float4* denoised_in, int width,
                                    int height, float ISO, float4* beauty_out,
                                    float4* denoised_out);