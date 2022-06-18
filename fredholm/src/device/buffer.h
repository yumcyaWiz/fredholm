#pragma once
#include <cstring>
#include <vector>

#include "device/types.h"
#include "device/util.h"

namespace fredholm
{

// RAII buffer object which is on device
template <typename T>
class DeviceBuffer
{
 public:
  DeviceBuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_ptr),
                          m_buffer_size * sizeof(T)));
  }

  DeviceBuffer(uint32_t buffer_size, int value) : DeviceBuffer<T>(buffer_size)
  {
    CUDA_CHECK(cudaMemset(m_device_ptr, value, m_buffer_size * sizeof(T)));
  }

  DeviceBuffer(const std::vector<T>& values) : DeviceBuffer<T>(values.size())
  {
    copy_from_host_to_device(values);
  }

  DeviceBuffer(const DeviceBuffer<T>& other) = delete;

  DeviceBuffer(DeviceBuffer<T>&& other)
      : m_device_ptr(other.m_device_ptr), m_buffer_size(other.m_buffer_size)
  {
    other.m_device_ptr = nullptr;
    other.m_buffer_size = 0;
  }

  ~DeviceBuffer() noexcept(false)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_ptr)));
  }

  void copy_from_host_to_device(const std::vector<T>& value)
  {
    CUDA_CHECK(cudaMemcpy(m_device_ptr, value.data(), m_buffer_size * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  T* get_device_ptr() const { return m_device_ptr; }

  uint32_t get_size() const { return m_buffer_size; }

  uint32_t get_size_in_bytes() const { return m_buffer_size * sizeof(T); }

 private:
  T* m_device_ptr;
  uint32_t m_buffer_size;
};

}  // namespace fredholm