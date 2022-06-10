#pragma once

#include "types.h"
#include "util.h"

template <typename T>
class Buffer : public DeviceAndHostObject<T>
{
 public:
  Buffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    // allocate on host
    m_host_ptr = reinterpret_cast<T*>(malloc(m_buffer_size * sizeof(T)));

    // allocate on device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                          m_buffer_size * sizeof(T)));
  }

  ~Buffer() noexcept(false)
  {
    // free memory on host
    free(m_host_ptr);

    // free memory on device
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  T* get_host_ptr() const override { return m_host_ptr; }

  T* get_device_ptr() const override { return m_device_ptr; }

  uint32_t get_size() const { return m_buffer_size; }

  T get_value(uint32_t index) const { return m_host_ptr[index]; }

  uint32_t get_size_in_bytes() const override
  {
    return m_buffer_size * sizeof(T);
  }

  void copy_from_host_to_device() override
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_host_ptr), m_device_ptr,
                          m_buffer_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_from_device_to_host() override
  {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(m_device_ptr), m_host_ptr,
                          m_buffer_size * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  uint32_t m_buffer_size = 0;

  T* m_host_ptr = nullptr;
  T* m_device_ptr = nullptr;
};