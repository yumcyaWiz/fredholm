#pragma once
#include <memory>

#include "cuda_util.h"

namespace fredholm
{

class PostProcess
{
   public:
    PostProcess()
    {
        kernel = std::make_unique<CUDAKernel>("tone_mapping_kernel.ptx",
                                              "tone_mapping_kernel");
    }

    ~PostProcess()
    {
        if (kernel) kernel.reset();
    }

    void run(uint32_t width, uint32_t height, float4* input, float4* output)
    {
        uint w = width;
        uint h = height;
        float chromatic_aberration = 1.0f;
        float ISO = 100.0f;

        const void* args[] = {&w,   &h,     &chromatic_aberration,
                              &ISO, &input, &output};
        constexpr int threads_per_block = 16;
        kernel->launch(max(width / threads_per_block, 1),
                       max(height / threads_per_block, 1), 1, threads_per_block,
                       threads_per_block, 1, args);
        cuda_check(cuCtxSynchronize());
    }

   private:
    std::unique_ptr<CUDAKernel> kernel = nullptr;
};

}  // namespace fredholm