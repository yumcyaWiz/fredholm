#pragma once
#include <cuda_gl_interop.h>

#include <cstring>
#include <vector>

#include "cwl/util.h"
#include "oglw/buffer.h"

namespace cwl
{

// RAII buffer object which is on device
template <typename T>
class CUDABuffer
{
 public:
  CUDABuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_ptr),
                          m_buffer_size * sizeof(T)));
  }

  CUDABuffer(uint32_t buffer_size, int value) : CUDABuffer<T>(buffer_size)
  {
    CUDA_CHECK(cudaMemset(m_d_ptr, value, m_buffer_size * sizeof(T)));
  }

  CUDABuffer(const std::vector<T>& values) : CUDABuffer<T>(values.size())
  {
    copy_from_host_to_device(values);
  }

  CUDABuffer(const CUDABuffer<T>& other) = delete;

  CUDABuffer(CUDABuffer<T>&& other)
      : m_d_ptr(other.m_d_ptr), m_buffer_size(other.m_buffer_size)
  {
    other.m_d_ptr = nullptr;
    other.m_buffer_size = 0;
  }

  ~CUDABuffer() noexcept(false)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_ptr)));
  }

  void copy_from_host_to_device(const std::vector<T>& value)
  {
    CUDA_CHECK(cudaMemcpy(m_d_ptr, value.data(), m_buffer_size * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_from_device_to_host(std::vector<T>& value)
  {
    value.resize(m_buffer_size);
    CUDA_CHECK(cudaMemcpy(value.data(), m_d_ptr, m_buffer_size * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  T* get_device_ptr() const { return m_d_ptr; }

  uint32_t get_size() const { return m_buffer_size; }

  uint32_t get_size_in_bytes() const { return m_buffer_size * sizeof(T); }

 private:
  T* m_d_ptr;
  uint32_t m_buffer_size;
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
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_resource));
    CUDA_CHECK(cudaGraphicsUnregisterResource(m_resource));
  }

  void clear()
  {
    CUDA_CHECK(cudaMemset(m_d_buffer, 0, m_buffer_size * sizeof(T)));
  }

  void copy_from_device_to_host(std::vector<T>& value)
  {
    value.resize(m_buffer_size);
    CUDA_CHECK(cudaMemcpy(value.data(), m_d_buffer, m_buffer_size * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  const oglw::Buffer<T>& get_gl_buffer() const { return m_buffer; }

  T* get_device_ptr() const { return m_d_buffer; }

  oglw::Buffer<T> m_buffer;
  uint32_t m_buffer_size;
  cudaGraphicsResource* m_resource;
  T* m_d_buffer;
};

}  // namespace cwl