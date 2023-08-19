#pragma once
#include <memory>

#include "cuda_util.h"
#include "post_process/shared.h"

namespace fredholm
{

class PostProcess
{
   public:
    PostProcess()
    {
        kernel = std::make_unique<CUDAKernel>(
            std::filesystem::path(CMAKE_CURRENT_SOURCE_DIR) /
                "tone_mapping_kernel.ptx",
            "tone_mapping_kernel");
        cuMemAlloc(&params_buffer, sizeof(PostProcessParams));
    }

    ~PostProcess()
    {
        if (output) output.reset();
        if (kernel) kernel.reset();
        if (params_buffer != 0) cuda_check(cuMemFree(params_buffer));
    }

    void run(uint32_t width, uint32_t height, float4* input)
    {
        PostProcessParams params;
        params.width = width;
        params.height = height;
        params.input = input;
        params.output = reinterpret_cast<float4*>(output->get_device_ptr());
        cuda_check(
            cuMemcpyHtoD(params_buffer, &params, sizeof(PostProcessParams)));

        const void* args[] = {&params_buffer};
        constexpr int threads_per_block = 16;
        kernel->launch(max(width / threads_per_block, 1),
                       max(height / threads_per_block, 1), 1, threads_per_block,
                       threads_per_block, 1, args);
        cuda_check(cuCtxSynchronize());
    }

   private:
    std::unique_ptr<CUDABuffer<float4>> output;
    std::unique_ptr<CUDAKernel> kernel;
    CUdeviceptr params_buffer = 0;
};

}  // namespace fredholm