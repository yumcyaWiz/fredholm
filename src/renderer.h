#pragma once
#include <optix.h>

#include "cuda_util.h"
#include "optix_util.h"

namespace fredholm
{

class Renderer
{
   public:
    Renderer(CUdevice device_id) : device{device_id} {}

   private:
    uint32_t width = 0;
    uint32_t height = 0;

    CUDADevice device;
    OptixDeviceContext context;
};

}  // namespace fredholm