#pragma once

#include <cstring>
#include <vector>

#include "cwl/util.h"
//
#include "oglw/buffer.h"
//
#include <cuda.h>
#include <cudaGL.h>

namespace cwl
{

// RAII buffer object which is on device
template <typename T>
class CUDABuffer
{
   public:
    CUDABuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
    {
        if (buffer_size == 0) return;
        cudaCheckError(cuMemAlloc(&m_d_ptr, m_buffer_size * sizeof(T)));
    }

    CUDABuffer(uint32_t buffer_size, uint32_t value)
        : CUDABuffer<T>(buffer_size)
    {
        if (buffer_size == 0) return;
        cudaCheckError(cuMemsetD32(
            m_d_ptr, value, m_buffer_size * sizeof(T) / sizeof(uint32_t)));
    }

    CUDABuffer(const std::vector<T>& values) : CUDABuffer<T>(values.size())
    {
        if (values.size() == 0) return;
        copy_from_host_to_device(values);
    }

    CUDABuffer(const CUDABuffer<T>& other) = delete;

    CUDABuffer(CUDABuffer<T>&& other)
        : m_d_ptr(other.m_d_ptr), m_buffer_size(other.m_buffer_size)
    {
        other.m_d_ptr = nullptr;
        other.m_buffer_size = 0;
    }

    ~CUDABuffer() { cudaCheckError(cuMemFree(m_d_ptr)); }

    void clear() const
    {
        cudaCheckError(cuMemsetD32(
            m_d_ptr, 0, m_buffer_size * sizeof(T) / sizeof(uint32_t)));
    }

    void copy_from_host_to_device(const std::vector<T>& value) const
    {
        cudaCheckError(
            cuMemcpyHtoD(m_d_ptr, value.data(), m_buffer_size * sizeof(T)));
    }

    void copy_from_device_to_host(std::vector<T>& value) const
    {
        value.resize(m_buffer_size);
        cudaCheckError(
            cuMemcpyDtoH(value.data(), m_d_ptr, m_buffer_size * sizeof(T)));
    }

    T* get_device_ptr() { return reinterpret_cast<T*>(m_d_ptr); }

    const T* get_const_device_ptr() const
    {
        return reinterpret_cast<const T*>(m_d_ptr);
    }

    uint32_t get_size() const { return m_buffer_size; }

    uint32_t get_size_in_bytes() const { return m_buffer_size * sizeof(T); }

   private:
    CUdeviceptr m_d_ptr = 0;
    uint32_t m_buffer_size = 0;
};

template <typename T>
struct CUDAGLBuffer {
    CUDAGLBuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
    {
        // create gl buffer
        std::vector<T> data(m_buffer_size);
        memset(data.data(), 0, m_buffer_size * sizeof(T));
        m_buffer.setData(data, GL_STATIC_DRAW);

        // get cuda device ptr from OpenGL texture
        cudaCheckError(
            cuGraphicsGLRegisterBuffer(&m_resource, m_buffer.getName(),
                                       CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE));
        cudaCheckError(cuGraphicsMapResources(1, &m_resource, 0));

        size_t size = 0;
        cudaCheckError(
            cuGraphicsResourceGetMappedPointer(&m_d_buffer, &size, m_resource));
    }

    CUDAGLBuffer(const CUDAGLBuffer& other) = delete;

    CUDAGLBuffer(CUDAGLBuffer&& other)
        : m_buffer(std::move(other.m_buffer)),
          m_resource(other.m_resource),
          m_d_buffer(other.m_d_buffer)
    {
    }

    ~CUDAGLBuffer()
    {
        cudaCheckError(cuGraphicsUnmapResources(1, &m_resource, 0));
        cudaCheckError(cuGraphicsUnregisterResource(m_resource));
    }

    void clear()
    {
        cudaCheckError(cuMemsetD32(
            m_d_buffer, 0, m_buffer_size * sizeof(T) / sizeof(uint32_t)));
    }

    void copy_from_device_to_host(std::vector<T>& value)
    {
        value.resize(m_buffer_size);
        cudaCheckError(
            cuMemcpyDtoH(value.data(), m_d_buffer, m_buffer_size * sizeof(T)));
    }

    const oglw::Buffer<T>& get_gl_buffer() const { return m_buffer; }

    T* get_device_ptr() const { return reinterpret_cast<T*>(m_d_buffer); }

    oglw::Buffer<T> m_buffer;
    uint32_t m_buffer_size = 0;
    CUgraphicsResource m_resource = nullptr;
    CUdeviceptr m_d_buffer = 0;
};

}  // namespace cwl