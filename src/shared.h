#pragma once
#include <cuda_runtime.h>

struct Params {
  float4* image;
  unsigned int image_width;
  unsigned int image_height;
};