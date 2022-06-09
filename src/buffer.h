#pragma once

#include "util.h"

template <typename T>
class Buffer
{
 public:
  Buffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    // allocate on host
    m_host_ptr = malloc(m_buffer_size * sizeof(T));

    // allocate on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                          m_buffer_size * sizeof(T)));
  }

  ~Buffer()
  {
    // free memory on host
    free(m_host_ptr);

    // free memory on device
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  uint32_t get_size() const { return m_buffer_size; }

  void copy_from_host_to_device()
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_host_ptr), m_device_ptr,
                          m_buffer_size, cudaMemcpyHostToDevice));
  }

  void copy_from_device_to_host()
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_device_ptr), m_host_ptr,
                          m_buffer_size, cudaMemcpyDeviceToHost));
  }

  T* get_host_ptr() const { return m_host_ptr; }
  T* get_device_ptr() const { return m_device_ptr; }

 private:
  uint32_t m_buffer_size = 0;

  T* m_host_ptr = nullptr;
  T* m_device_ptr = nullptr;
};