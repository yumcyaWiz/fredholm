#pragma once

#include <cuda_gl_interop.h>

#include "device/util.h"
#include "gcss/buffer.h"
#include "glm/glm.hpp"

namespace app
{

template <typename T>
struct CUDAGLBuffer {
  CUDAGLBuffer(uint32_t width, uint32_t height)
  {
    // create gl buffer
    std::vector<T> data(width * height);
    m_buffer.setData(data, GL_STATIC_DRAW);

    // get cuda device ptr from OpenGL texture
    CUDA_CHECK(
        cudaGraphicsGLRegisterBuffer(&m_resource, m_buffer.getName(),
                                     cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_resource));

    size_t size;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&m_d_buffer), &size, m_resource));
  }

  CUDAGLBuffer(const CUDAGLBuffer& other) = delete;

  CUDAGLBuffer(CUDAGLBuffer&& other)
      : m_buffer(std::move(other.m_buffer)),
        m_resource(other.m_resource),
        m_d_buffer(other.m_d_buffer)
  {
  }

  ~CUDAGLBuffer() noexcept(false)
  {
    // TODO: CUDA call (cudaGraphicsUnmapResources(1, &m_resource) ) failed with
    // error: 'invalid OpenGL or DirectX context'
    // this is because OpenGL context is destroyed before this call?
    // CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_resource));
    // CUDA_CHECK(cudaGraphicsUnregisterResource(m_resource));
  }

  const gcss::Buffer<T>& get_gl_buffer() const { return m_buffer; }

  T* get_device_ptr() const { return m_d_buffer; }

  gcss::Buffer<T> m_buffer;
  cudaGraphicsResource* m_resource;
  T* m_d_buffer;
};

};  // namespace app