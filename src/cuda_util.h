#pragma once

#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_INLINE __forceinline__

#ifndef __CUDACC__
#include <cuda.h>

#include <format>
#include <iostream>
#include <source_location>

#include "gl_util.h"
#include "helper_math.h"
//
#include <cudaGL.h>

namespace fredholm
{

inline void cuda_check(
    const CUresult &result,
    const std::source_location &loc = std::source_location::current())
{
    if (result == CUDA_SUCCESS) return;

    const char *errorName = nullptr;
    cuGetErrorName(result, &errorName);
    const char *errorString = nullptr;
    cuGetErrorString(result, &errorString);

    throw std::runtime_error(
        std::format("{}({}:{}) {}: {}\n", loc.file_name(), loc.line(),
                    loc.column(), loc.function_name(), errorName, errorString));
}

template <typename T>
class CUDABuffer
{
   private:
    CUdeviceptr dptr = 0;
    uint32_t size = 0;

   public:
    CUDABuffer() {}

    CUDABuffer(uint32_t size) : size(size)
    {
        cuda_check(cuMemAlloc(&dptr, sizeof(T) * size));
    }

    CUDABuffer(const T *hptr, uint32_t size) : CUDABuffer(size)
    {
        copy_h_to_d(hptr);
    }

    CUDABuffer(const CUDABuffer &) = delete;

    CUDABuffer(CUDABuffer &&other) : dptr(other.dptr), size(other.size)
    {
        other.dptr = 0;
    }

    ~CUDABuffer() { cuda_check(cuMemFree(dptr)); }

    CUdeviceptr get_device_ptr() { return dptr; }

    void copy_h_to_d(const T *hptr) const
    {
        cuda_check(cuMemcpyHtoD(dptr, hptr, sizeof(T) * size));
    }

    void copy_d_to_h(T *hptr) const
    {
        cuda_check(cuMemcpyDtoH(hptr, dptr, sizeof(T) * size));
    }
};

template <typename T>
class CUDAGLBuffer
{
   private:
    GLBuffer buffer;
    CUdeviceptr dptr = 0;
    CUgraphicsResource resource = nullptr;
    uint32_t size = 0;

   public:
    CUDAGLBuffer() {}

    CUDAGLBuffer(uint32_t size) : size(size)
    {
        cuda_check(cuGraphicsGLRegisterBuffer(
            &resource, buffer.getName(),
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
        cuda_check(cuGraphicsMapResources(1, &resource, 0));
        size_t s = 0;
        cuda_check(cuGraphicsResourceGetMappedPointer(&dptr, &s, resource));
    }

    CUDAGLBuffer(const CUDAGLBuffer &) = delete;

    CUDAGLBuffer(CUDAGLBuffer &&other) : dptr(other.dptr), size(other.size)
    {
        other.dptr = 0;
        other.size = 0;
    }

    ~CUDAGLBuffer()
    {
        cuda_check(cuGraphicsUnmapResources(1, &resource, 0));
        cuda_check(cuGraphicsUnregisterResource(resource));
    }

    const CUdeviceptr &get_device_ptr() const { return dptr; }

    void copy_h_to_d(const T *hptr) const
    {
        cuda_check(cuMemcpyHtoD(dptr, hptr, sizeof(T) * size));
    }

    void copy_d_to_h(T *hptr) const
    {
        cuda_check(cuMemcpyDtoH(hptr, dptr, sizeof(T) * size));
    }
};

class CUDADevice
{
   private:
    CUdevice device = 0;
    CUcontext context = nullptr;

   public:
    CUDADevice(CUdevice device) : device(device)
    {
        // check device availability
        int nDevices = 0;
        cuda_check(cuDeviceGetCount(&nDevices));
        if (device >= nDevices)
        {
            throw std::runtime_error(
                std::format("device {} is not available\n", device));
        }

        cuda_check(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
        cuCtxPushCurrent(context);
    }

    CUDADevice(const CUDADevice &) = delete;

    CUDADevice(CUDADevice &&other)
        : device(other.device), context(other.context)
    {
        other.device = 0;
        other.context = nullptr;
    }

    ~CUDADevice()
    {
        cuCtxPopCurrent(&context);
        cuCtxDestroy(context);
    }

    CUcontext get_context() const { return context; }

    void synchronize() const { cuda_check(cuCtxSynchronize()); }
};

class CUDAKernel
{
   private:
    CUmodule module = nullptr;
    CUfunction function = nullptr;

   public:
    CUDAKernel(const std::string &filename, const std::string &kernelName)
    {
        cuda_check(cuModuleLoad(&module, filename.c_str()));
        cuda_check(cuModuleGetFunction(&function, module, kernelName.c_str()));
    }

    CUDAKernel(const CUDAKernel &) = delete;

    CUDAKernel(CUDAKernel &&other)
        : module(other.module), function(other.function)
    {
        other.module = nullptr;
        other.function = nullptr;
    }

    ~CUDAKernel() { cuda_check(cuModuleUnload(module)); }

    void launch(const int gridX, const int gridY, const int gridZ,
                const int blockX, const int blockY, const int blockZ,
                const void *args[]) const
    {
        cuda_check(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                                  blockZ, 0, nullptr, const_cast<void **>(args),
                                  nullptr));
    }
};

}  // namespace fredholm

#endif